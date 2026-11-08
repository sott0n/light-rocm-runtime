#include "executor/qwen/stateful_graph_context.hpp"
#include "executor/stateful_graph/compiled_scheduler.hpp"
#include "executor/stateful_graph/device_scheduler.hpp"

namespace sg = lrrt::stateful_graph;
namespace qd = lrrt::qwen::stateful_graph;

namespace {

__device__ qd::ProfileStage profile_stage(qd::Opcode opcode) {
  switch (opcode) {
  case qd::Opcode::attention_rmsnorm_f32:
    return qd::ProfileStage::attention_norm;
  case qd::Opcode::qkv_projection_f32:
    return qd::ProfileStage::qkv_projection;
  case qd::Opcode::rope_kv_cache_f32:
    return qd::ProfileStage::rope_kv_cache;
  case qd::Opcode::grouped_query_attention_f32:
    return qd::ProfileStage::attention;
  case qd::Opcode::attention_output_residual_f32:
    return qd::ProfileStage::attention_output;
  case qd::Opcode::mlp_rmsnorm_f32:
    return qd::ProfileStage::mlp_norm;
  case qd::Opcode::gate_up_silu_f32:
    return qd::ProfileStage::gate_up_silu;
  case qd::Opcode::down_matvec_f32:
    return qd::ProfileStage::down_projection;
  case qd::Opcode::residual_advance:
    return qd::ProfileStage::residual_advance;
  case qd::Opcode::final_rmsnorm_f32:
    return qd::ProfileStage::final_norm;
  case qd::Opcode::lm_head_top1_partition_f32:
    return qd::ProfileStage::lm_head_top1;
  case qd::Opcode::top1_reduce:
    return qd::ProfileStage::top1_reduce;
  case qd::Opcode::embedding_feedback_f32:
    return qd::ProfileStage::embedding_feedback;
  }
  return qd::ProfileStage::count;
}

__device__ __forceinline__ void no_profile_initialize(void *) {}

__device__ __forceinline__ void no_profile_complete(const sg::Stage &,
                                                    const sg::Task *, void *) {}

__device__ bool valid_qwen_task(const sg::Task &task,
                                const qd::Context *context) {
  switch (static_cast<qd::Opcode>(task.opcode)) {
  case qd::Opcode::attention_rmsnorm_f32:
  case qd::Opcode::mlp_rmsnorm_f32:
  case qd::Opcode::final_rmsnorm_f32:
  case qd::Opcode::down_matvec_f32:
    return task.begin < task.end && task.end <= context->hidden_size;
  case qd::Opcode::qkv_projection_f32:
    return task.begin < task.end && task.end <= qd::kPackedQkvSize;
  case qd::Opcode::rope_kv_cache_f32:
    return task.begin == 0 && task.end == qd::kRotaryHeads;
  case qd::Opcode::grouped_query_attention_f32:
    return task.begin < task.end && task.end <= qd::kAttentionHeads;
  case qd::Opcode::attention_output_residual_f32:
    return task.begin < task.end && task.end <= context->hidden_size;
  case qd::Opcode::residual_advance:
    return task.begin < task.end && task.end <= context->hidden_size &&
           task.argument0 != 0;
  case qd::Opcode::gate_up_silu_f32:
    return task.begin < task.end && task.end <= context->intermediate_size;
  case qd::Opcode::lm_head_top1_partition_f32:
    return task.begin < task.end && task.end <= context->tail->vocab_size &&
           task.argument0 < context->tail->partition_count;
  case qd::Opcode::top1_reduce:
    return task.begin == 0 && task.end == context->tail->partition_count;
  case qd::Opcode::embedding_feedback_f32:
    return task.begin == 0 && task.end == context->hidden_size;
  }
  return false;
}

__device__ bool valid_qwen_contexts(const sg::State *state,
                                    const qd::Context *contexts) {
  (void)state;
  const qd::Context *context = contexts;
  if (contexts == nullptr || context->layer_count == 0 ||
      context->hidden_size != qd::kHiddenSize ||
      context->query_size != qd::kQuerySize ||
      context->kv_size != qd::kKvSize ||
      context->intermediate_size != qd::kIntermediateSize ||
      context->tail == nullptr || context->tail->final_norm_weight == nullptr ||
      context->tail->lm_head_weight == nullptr ||
      context->tail->token_embeddings == nullptr ||
      context->tail->final_normalized == nullptr ||
      context->tail->partial_max_values == nullptr ||
      context->tail->partial_max_indices == nullptr ||
      context->tail->decode_state == nullptr ||
      context->tail->output_tokens == nullptr ||
      context->tail->vocab_size == 0 || context->tail->partition_count == 0 ||
      context->tail->partition_count > qd::kMaximumLmHeadPartitions ||
      context->tail->embedding_count != context->tail->vocab_size ||
      context->tail->decode_state->maximum_tokens == 0 ||
      context->tail->decode_state->maximum_tokens > context->cache_capacity ||
      context->tail->decode_state->token_position !=
          context->tail->decode_state->kv_length ||
      context->tail->decode_state->token_position >= context->cache_capacity) {
    return false;
  }

  for (uint32_t layer = 1; layer < context->layer_count; ++layer) {
    const qd::Context &candidate = contexts[layer];
    if (candidate.hidden_size != context->hidden_size ||
        candidate.query_size != context->query_size ||
        candidate.kv_size != context->kv_size ||
        candidate.intermediate_size != context->intermediate_size ||
        candidate.cache_capacity != context->cache_capacity ||
        candidate.layer_count != context->layer_count ||
        candidate.tail != context->tail) {
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ float wave_sum(float value) {
  for (uint32_t offset = 16; offset != 0; offset >>= 1u) {
    value += __shfl_down(value, offset, 32);
  }
  return value;
}

template <uint32_t InputSize, uint32_t PrefetchColumns>
__device__ __forceinline__ void
prefetched_two_row_matvec(const float *input, const float *first_weight,
                          const float *second_weight, bool has_second_row,
                          uint32_t lane, float &first_sum, float &second_sum) {
  constexpr uint32_t kColumnStride = 32u * PrefetchColumns;
  static_assert(InputSize % kColumnStride == 0u);
  for (uint32_t column = lane; column < InputSize; column += kColumnStride) {
    float inputs[PrefetchColumns];
    float first_weights[PrefetchColumns];
    float second_weights[PrefetchColumns];
#pragma unroll
    for (uint32_t index = 0; index < PrefetchColumns; ++index) {
      const uint32_t prefetched_column = column + 32u * index;
      inputs[index] = input[prefetched_column];
      first_weights[index] = first_weight[prefetched_column];
      second_weights[index] =
          has_second_row ? second_weight[prefetched_column] : 0.0f;
    }
#pragma unroll
    for (uint32_t index = 0; index < PrefetchColumns; ++index) {
      first_sum = fmaf(inputs[index], first_weights[index], first_sum);
      if (has_second_row) {
        second_sum = fmaf(inputs[index], second_weights[index], second_sum);
      }
    }
  }
}

__device__ void matvec_range_wave(const float *input, const float *weight,
                                  float *output, uint32_t input_size,
                                  const sg::Task &task) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wave = threadIdx.x >> 5u;
  constexpr uint32_t kWavesPerWorker = sg::kWorkerThreads / 32u;
  for (uint32_t row = task.begin + wave; row < task.end;
       row += kWavesPerWorker) {
    float sum = 0.0f;
    const float *row_weight = weight + static_cast<size_t>(row) * input_size;
    for (uint32_t column = lane; column < input_size; column += 32u) {
      sum = fmaf(input[column], row_weight[column], sum);
    }
    sum = wave_sum(sum);
    if (lane == 0) {
      output[row] = sum;
    }
  }
}

// The compiled resident launch has fewer workgroups than the 32-way down
// projection. Redistribute the complete output range over all resident waves
// instead of making a subset of workers execute a second task serially. Each
// wave accumulates two independent rows so both weight streams can remain in
// flight while sharing the input load.
__device__ void down_matvec_resident_stage(const qd::Context *context,
                                           const sg::Stage &stage,
                                           const sg::Task *tasks,
                                           uint32_t worker,
                                           uint32_t active_workers) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wave = threadIdx.x >> 5u;
  constexpr uint32_t kWavesPerWorker = sg::kWorkerThreads / 32u;
  const uint32_t first_row = tasks[stage.task_offset].begin;
  const uint32_t end_row = tasks[stage.task_offset + stage.task_count - 1u].end;
  const uint32_t resident_wave = worker * kWavesPerWorker + wave;
  const uint32_t resident_waves = active_workers * kWavesPerWorker;

  for (uint32_t row = first_row + resident_wave; row < end_row;
       row += 2u * resident_waves) {
    const uint32_t second_row = row + resident_waves;
    const bool has_second_row = second_row < end_row;
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    const float *first_weight =
        context->down_weight +
        static_cast<size_t>(row) * context->intermediate_size;
    const float *second_weight =
        has_second_row
            ? context->down_weight +
                  static_cast<size_t>(second_row) * context->intermediate_size
            : first_weight;
    // Four 38-column groups cover the 4,864-element input without a tail.
    prefetched_two_row_matvec<qd::kIntermediateSize, 38u>(
        context->activated, first_weight, second_weight, has_second_row, lane,
        first_sum, second_sum);
    first_sum = wave_sum(first_sum);
    second_sum = wave_sum(second_sum);
    if (lane == 0) {
      context->projected[row] = first_sum;
      if (has_second_row) {
        context->projected[second_row] = second_sum;
      }
    }
  }
}

__device__ __forceinline__ bool better_top1(float value, uint32_t index,
                                            float best_value,
                                            uint32_t best_index) {
  return value > best_value || (value == best_value && index < best_index);
}

__device__ void lm_head_top1_partition(const qd::Context *context,
                                       const sg::Task &task, float *reduction,
                                       uint32_t *wave_indices) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wave = threadIdx.x >> 5u;
  constexpr uint32_t kWavesPerWorker = sg::kWorkerThreads / 32u;
  float best_value = -INFINITY;
  uint32_t best_index = UINT32_MAX;
  for (uint32_t row = task.begin + wave; row < task.end;
       row += kWavesPerWorker) {
    float sum = 0.0f;
    const float *row_weight = context->tail->lm_head_weight +
                              static_cast<size_t>(row) * context->hidden_size;
    for (uint32_t column = lane; column < context->hidden_size; column += 32u) {
      sum = fmaf(context->tail->final_normalized[column], row_weight[column],
                 sum);
    }
    sum = wave_sum(sum);
    if (lane == 0 && better_top1(sum, row, best_value, best_index)) {
      best_value = sum;
      best_index = row;
    }
  }
  if (lane == 0) {
    reduction[wave] = best_value;
    wave_indices[wave] = best_index;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    best_value = reduction[0];
    best_index = wave_indices[0];
    for (uint32_t candidate = 1; candidate < kWavesPerWorker; ++candidate) {
      if (better_top1(reduction[candidate], wave_indices[candidate], best_value,
                      best_index)) {
        best_value = reduction[candidate];
        best_index = wave_indices[candidate];
      }
    }
    context->tail->partial_max_values[task.argument0] = best_value;
    context->tail->partial_max_indices[task.argument0] = best_index;
  }
}

// Resident LM-head partitions contain enough rows that a single independent
// weight stream per wave leaves memory latency exposed. Accumulate two rows in
// parallel while loading the normalized hidden value once, then retain the
// same per-wave and per-workgroup Top-1 reduction contract as the host path.
__device__ void lm_head_top1_resident_partition(const qd::Context *context,
                                                const sg::Task &task) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wave = threadIdx.x >> 5u;
  constexpr uint32_t kWavesPerWorker = sg::kWorkerThreads / 32u;
  __shared__ float reduction[kWavesPerWorker];
  __shared__ uint32_t wave_indices[kWavesPerWorker];
  float best_value = -INFINITY;
  uint32_t best_index = UINT32_MAX;

  for (uint32_t row = task.begin + wave; row < task.end;
       row += 2u * kWavesPerWorker) {
    const uint32_t second_row = row + kWavesPerWorker;
    const bool has_second_row = second_row < task.end;
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    const float *first_weight = context->tail->lm_head_weight +
                                static_cast<size_t>(row) * context->hidden_size;
    const float *second_weight =
        has_second_row
            ? context->tail->lm_head_weight +
                  static_cast<size_t>(second_row) * context->hidden_size
            : first_weight;
    prefetched_two_row_matvec<qd::kHiddenSize, 14u>(
        context->tail->final_normalized, first_weight, second_weight,
        has_second_row, lane, first_sum, second_sum);
    first_sum = wave_sum(first_sum);
    second_sum = wave_sum(second_sum);
    if (lane == 0) {
      if (better_top1(first_sum, row, best_value, best_index)) {
        best_value = first_sum;
        best_index = row;
      }
      if (has_second_row &&
          better_top1(second_sum, second_row, best_value, best_index)) {
        best_value = second_sum;
        best_index = second_row;
      }
    }
  }

  if (lane == 0) {
    reduction[wave] = best_value;
    wave_indices[wave] = best_index;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    best_value = reduction[0];
    best_index = wave_indices[0];
    for (uint32_t candidate = 1; candidate < kWavesPerWorker; ++candidate) {
      if (better_top1(reduction[candidate], wave_indices[candidate], best_value,
                      best_index)) {
        best_value = reduction[candidate];
        best_index = wave_indices[candidate];
      }
    }
    context->tail->partial_max_values[task.argument0] = best_value;
    context->tail->partial_max_indices[task.argument0] = best_index;
  }
}

__device__ void reduce_top1(const qd::Context *context) {
  if (threadIdx.x != 0) {
    return;
  }
  float best_value = context->tail->partial_max_values[0];
  uint32_t best_index = context->tail->partial_max_indices[0];
  for (uint32_t candidate = 1; candidate < context->tail->partition_count;
       ++candidate) {
    const float value = context->tail->partial_max_values[candidate];
    const uint32_t index = context->tail->partial_max_indices[candidate];
    if (better_top1(value, index, best_value, best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  context->tail->decode_state->sampled_logit = best_value;
  context->tail->decode_state->sampled_token = best_index;
}

__device__ void embedding_feedback(qd::Context *context, sg::State *state) {
  qd::DecodeState *decode = context->tail->decode_state;
  if (threadIdx.x == 0) {
    const uint32_t output_index = decode->generated_tokens;
    const uint32_t token = decode->sampled_token;
    context->tail->output_tokens[output_index] = token;
    decode->generated_tokens = output_index + 1u;

    const bool reached_eos = token == decode->eos_token;
    const bool reached_limit =
        decode->generated_tokens >= decode->maximum_tokens;
    if (reached_eos || reached_limit) {
      decode->stop_reason =
          static_cast<uint32_t>(reached_eos ? qd::StopReason::eos_token
                                            : qd::StopReason::maximum_tokens);
      state->predicate_value = 1;
    } else {
      ++decode->token_position;
      ++decode->kv_length;
    }
  }
  __syncthreads();

  if (state->predicate_value == 0) {
    const size_t embedding_offset =
        static_cast<size_t>(decode->sampled_token) * context->hidden_size;
    for (uint32_t index = threadIdx.x; index < context->hidden_size;
         index += sg::kWorkerThreads) {
      context->hidden[index] =
          context->tail->token_embeddings[embedding_offset + index];
    }
  }
}

__device__ void qkv_projection_range_wave(const qd::Context *context,
                                          const sg::Task &task) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wave = threadIdx.x >> 5u;
  constexpr uint32_t kWavesPerWorker = sg::kWorkerThreads / 32u;
  for (uint32_t packed_row = task.begin + wave; packed_row < task.end;
       packed_row += kWavesPerWorker) {
    const float *weight = context->q_weight;
    const float *bias = context->q_bias;
    float *output = context->query;
    uint32_t row = packed_row;
    if (packed_row >= context->query_size + context->kv_size) {
      weight = context->v_weight;
      bias = context->v_bias;
      output = context->value;
      row -= context->query_size + context->kv_size;
    } else if (packed_row >= context->query_size) {
      weight = context->k_weight;
      bias = context->k_bias;
      output = context->key;
      row -= context->query_size;
    }

    float sum = 0.0f;
    const float *row_weight =
        weight + static_cast<size_t>(row) * context->hidden_size;
    for (uint32_t column = lane; column < context->hidden_size; column += 32u) {
      sum =
          fmaf(context->attention_normalized[column], row_weight[column], sum);
    }
    sum = wave_sum(sum);
    if (lane == 0) {
      output[row] = sum + bias[row];
    }
  }
}

struct QkvRowView {
  const float *weight;
  const float *bias;
  float *output;
  uint32_t row;
};

__device__ __forceinline__ QkvRowView qkv_row_view(const qd::Context *context,
                                                   uint32_t packed_row) {
  QkvRowView view{context->q_weight, context->q_bias, context->query,
                  packed_row};
  if (packed_row >= context->query_size + context->kv_size) {
    view.weight = context->v_weight;
    view.bias = context->v_bias;
    view.output = context->value;
    view.row -= context->query_size + context->kv_size;
  } else if (packed_row >= context->query_size) {
    view.weight = context->k_weight;
    view.bias = context->k_bias;
    view.output = context->key;
    view.row -= context->query_size;
  }
  return view;
}

__device__ void qkv_projection_resident_stage(const qd::Context *context,
                                              const sg::Stage &stage,
                                              const sg::Task *tasks,
                                              uint32_t worker,
                                              uint32_t active_workers) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wave = threadIdx.x >> 5u;
  constexpr uint32_t kWavesPerWorker = sg::kWorkerThreads / 32u;
  const uint32_t first_row = tasks[stage.task_offset].begin;
  const uint32_t end_row = tasks[stage.task_offset + stage.task_count - 1u].end;
  const uint32_t resident_wave = worker * kWavesPerWorker + wave;
  const uint32_t resident_waves = active_workers * kWavesPerWorker;

  for (uint32_t packed_row = first_row + resident_wave; packed_row < end_row;
       packed_row += 2u * resident_waves) {
    const uint32_t second_packed_row = packed_row + resident_waves;
    const bool has_second_row = second_packed_row < end_row;
    const QkvRowView first = qkv_row_view(context, packed_row);
    const QkvRowView second =
        has_second_row ? qkv_row_view(context, second_packed_row) : first;
    const float *first_weight =
        first.weight + static_cast<size_t>(first.row) * context->hidden_size;
    const float *second_weight =
        second.weight + static_cast<size_t>(second.row) * context->hidden_size;
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    prefetched_two_row_matvec<qd::kHiddenSize, 14u>(
        context->attention_normalized, first_weight, second_weight,
        has_second_row, lane, first_sum, second_sum);
    first_sum = wave_sum(first_sum);
    second_sum = wave_sum(second_sum);
    if (lane == 0) {
      first.output[first.row] = first_sum + first.bias[first.row];
      if (has_second_row) {
        second.output[second.row] = second_sum + second.bias[second.row];
      }
    }
  }
}

__device__ void gate_up_silu_range_wave(const qd::Context *context,
                                        const sg::Task &task) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wave = threadIdx.x >> 5u;
  constexpr uint32_t kWavesPerWorker = sg::kWorkerThreads / 32u;
  for (uint32_t row = task.begin + wave; row < task.end;
       row += kWavesPerWorker) {
    float gate = 0.0f;
    float up = 0.0f;
    const size_t row_offset = static_cast<size_t>(row) * context->hidden_size;
    // Two 14-column groups cover the 896-element row while keeping 42
    // independent input and weight loads in flight.
    prefetched_two_row_matvec<qd::kHiddenSize, 14u>(
        context->mlp_normalized, context->gate_weight + row_offset,
        context->up_weight + row_offset, true, lane, gate, up);
    gate = wave_sum(gate);
    up = wave_sum(up);
    if (lane == 0) {
      context->activated[row] = (gate / (1.0f + expf(-gate))) * up;
    }
  }
}

__device__ void rope_kv_cache(const sg::State *state,
                              const qd::Context *context) {
  (void)state;
  const uint32_t position = context->tail->decode_state->token_position;
  const size_t rope_offset = static_cast<size_t>(position) * qd::kRotarySize;
  for (uint32_t pair = threadIdx.x; pair < qd::kRotaryHeads * qd::kRotarySize;
       pair += sg::kWorkerThreads) {
    const uint32_t head = pair / qd::kRotarySize;
    const uint32_t frequency = pair % qd::kRotarySize;
    const bool is_query = head < qd::kAttentionHeads;
    const uint32_t local_head = is_query ? head : head - qd::kAttentionHeads;
    float *vector = is_query ? context->query : context->key;
    const uint32_t first = local_head * qd::kHeadSize + frequency;
    const uint32_t second = first + qd::kRotarySize;
    const float first_value = vector[first];
    const float second_value = vector[second];
    const float cosine = context->rope_cos[rope_offset + frequency];
    const float sine = context->rope_sin[rope_offset + frequency];
    const float rotated_first = first_value * cosine - second_value * sine;
    const float rotated_second = second_value * cosine + first_value * sine;
    vector[first] = rotated_first;
    vector[second] = rotated_second;
    if (!is_query) {
      const size_t cache_offset =
          static_cast<size_t>(position) * context->kv_size;
      context->key_cache[cache_offset + first] = rotated_first;
      context->key_cache[cache_offset + second] = rotated_second;
    }
  }

  const size_t cache_offset = static_cast<size_t>(position) * context->kv_size;
  for (uint32_t index = threadIdx.x; index < context->kv_size;
       index += sg::kWorkerThreads) {
    context->value_cache[cache_offset + index] = context->value[index];
  }
}

__device__ void grouped_query_attention(const sg::Task &task,
                                        const sg::State *state,
                                        qd::Context *context) {
  (void)state;
  constexpr uint32_t kQueriesPerKvHead = qd::kAttentionHeads / qd::kKvHeads;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wave = threadIdx.x >> 5u;
  constexpr uint32_t kWavesPerWorker = sg::kWorkerThreads / 32u;
  for (uint32_t query_head = task.begin + wave; query_head < task.end;
       query_head += kWavesPerWorker) {
    const uint32_t kv_head = query_head / kQueriesPerKvHead;
    const uint32_t query_offset = query_head * qd::kHeadSize;
    const uint32_t kv_offset = kv_head * qd::kHeadSize;
    const uint32_t first = query_offset + lane;
    const uint32_t second = first + 32u;
    float weighted_first = 0.0f;
    float weighted_second = 0.0f;
    float maximum = -INFINITY;
    float denominator = 0.0f;

    const uint32_t kv_length = context->tail->decode_state->kv_length;
    for (uint32_t position = 0; position <= kv_length; ++position) {
      const size_t cache_offset =
          static_cast<size_t>(position) * context->kv_size + kv_offset;
      float dot_product =
          context->query[first] * context->key_cache[cache_offset + lane] +
          context->query[second] *
              context->key_cache[cache_offset + lane + 32u];
      dot_product = wave_sum(dot_product);
      const float score = __shfl(dot_product, 0, 32) *
                          rsqrtf(static_cast<float>(qd::kHeadSize));
      const float next_maximum = fmaxf(maximum, score);
      const float previous_scale = expf(maximum - next_maximum);
      const float score_scale = expf(score - next_maximum);
      denominator = denominator * previous_scale + score_scale;
      weighted_first = weighted_first * previous_scale +
                       score_scale * context->value_cache[cache_offset + lane];
      weighted_second =
          weighted_second * previous_scale +
          score_scale * context->value_cache[cache_offset + lane + 32u];
      maximum = next_maximum;
    }

    context->query[first] = weighted_first / denominator;
    context->query[second] = weighted_second / denominator;
  }
}

__device__ void attention_output_residual_range_wave(const qd::Context *context,
                                                     const sg::Task &task) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wave = threadIdx.x >> 5u;
  constexpr uint32_t kWavesPerWorker = sg::kWorkerThreads / 32u;
  for (uint32_t row = task.begin + wave; row < task.end;
       row += kWavesPerWorker) {
    float sum = 0.0f;
    const float *row_weight =
        context->out_weight + static_cast<size_t>(row) * context->query_size;
    for (uint32_t column = lane; column < context->query_size; column += 32u) {
      sum = fmaf(context->query[column], row_weight[column], sum);
    }
    sum = wave_sum(sum);
    if (lane == 0) {
      context->hidden[row] += sum;
    }
  }
}

__device__ void attention_output_resident_stage(const qd::Context *context,
                                                const sg::Stage &stage,
                                                const sg::Task *tasks,
                                                uint32_t worker,
                                                uint32_t active_workers) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wave = threadIdx.x >> 5u;
  constexpr uint32_t kWavesPerWorker = sg::kWorkerThreads / 32u;
  const uint32_t first_row = tasks[stage.task_offset].begin;
  const uint32_t end_row = tasks[stage.task_offset + stage.task_count - 1u].end;
  const uint32_t resident_wave = worker * kWavesPerWorker + wave;
  const uint32_t resident_waves = active_workers * kWavesPerWorker;

  for (uint32_t row = first_row + resident_wave; row < end_row;
       row += 2u * resident_waves) {
    const uint32_t second_row = row + resident_waves;
    const bool has_second_row = second_row < end_row;
    const float *first_weight =
        context->out_weight + static_cast<size_t>(row) * context->query_size;
    const float *second_weight =
        has_second_row ? context->out_weight + static_cast<size_t>(second_row) *
                                                   context->query_size
                       : first_weight;
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    prefetched_two_row_matvec<qd::kHiddenSize, 14u>(
        context->query, first_weight, second_weight, has_second_row, lane,
        first_sum, second_sum);
    first_sum = wave_sum(first_sum);
    second_sum = wave_sum(second_sum);
    if (lane == 0) {
      context->hidden[row] += first_sum;
      if (has_second_row) {
        context->hidden[second_row] += second_sum;
      }
    }
  }
}

__device__ void execute_task(const sg::Task &task, sg::State *state,
                             qd::Context *context, float *reduction,
                             uint32_t *wave_indices) {
  const qd::Opcode opcode = static_cast<qd::Opcode>(task.opcode);
  if (opcode == qd::Opcode::attention_rmsnorm_f32 ||
      opcode == qd::Opcode::mlp_rmsnorm_f32 ||
      opcode == qd::Opcode::final_rmsnorm_f32) {
    float sum_squares = 0.0f;
    for (uint32_t index = task.begin + threadIdx.x; index < task.end;
         index += sg::kWorkerThreads) {
      const float value = context->hidden[index];
      sum_squares += value * value;
    }
    reduction[threadIdx.x] = sum_squares;
    __syncthreads();
    for (uint32_t stride = sg::kWorkerThreads / 2u; stride != 0;
         stride >>= 1u) {
      if (threadIdx.x < stride) {
        reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      }
      __syncthreads();
    }
    const float epsilon = __uint_as_float(task.argument0);
    const float inverse_rms = rsqrtf(
        reduction[0] / static_cast<float>(task.end - task.begin) + epsilon);
    const float *norm_weight = context->tail->final_norm_weight;
    float *normalized = context->tail->final_normalized;
    if (opcode == qd::Opcode::attention_rmsnorm_f32) {
      norm_weight = context->attention_norm_weight;
      normalized = context->attention_normalized;
    } else if (opcode == qd::Opcode::mlp_rmsnorm_f32) {
      norm_weight = context->mlp_norm_weight;
      normalized = context->mlp_normalized;
    }
    for (uint32_t index = task.begin + threadIdx.x; index < task.end;
         index += sg::kWorkerThreads) {
      normalized[index] =
          context->hidden[index] * inverse_rms * norm_weight[index];
    }
    return;
  }

  if (opcode == qd::Opcode::qkv_projection_f32) {
    qkv_projection_range_wave(context, task);
    return;
  }

  if (opcode == qd::Opcode::rope_kv_cache_f32) {
    rope_kv_cache(state, context);
    return;
  }

  if (opcode == qd::Opcode::grouped_query_attention_f32) {
    grouped_query_attention(task, state, context);
    return;
  }

  if (opcode == qd::Opcode::attention_output_residual_f32) {
    attention_output_residual_range_wave(context, task);
    return;
  }

  if (opcode == qd::Opcode::gate_up_silu_f32) {
    gate_up_silu_range_wave(context, task);
    return;
  }

  if (opcode == qd::Opcode::down_matvec_f32) {
    matvec_range_wave(context->activated, context->down_weight,
                      context->projected, context->intermediate_size, task);
    return;
  }

  if (opcode == qd::Opcode::lm_head_top1_partition_f32) {
    lm_head_top1_partition(context, task, reduction, wave_indices);
    return;
  }

  if (opcode == qd::Opcode::top1_reduce) {
    reduce_top1(context);
    return;
  }

  if (opcode == qd::Opcode::embedding_feedback_f32) {
    embedding_feedback(context, state);
    return;
  }

  for (uint32_t index = task.begin + threadIdx.x; index < task.end;
       index += sg::kWorkerThreads) {
    context->hidden[index] += context->projected[index];
  }
  if (threadIdx.x == 0) {
    const bool token_boundary =
        task.argument1 == 0 || (state->step + 1u) % context->layer_count == 0;
    if (token_boundary && task.argument1 == 0) {
      state->predicate_value += task.argument0;
    }
    state->branch_value = static_cast<uint32_t>(token_boundary);
  }
}

struct QwenWorkload {
  __device__ static __forceinline__ bool valid_context(const sg::State *state,
                                                       void *opaque_context) {
    auto *contexts = static_cast<qd::Context *>(opaque_context);
    return valid_qwen_contexts(state, contexts);
  }

  __device__ static __forceinline__ bool valid_task(const sg::Task &task,
                                                    void *opaque_context) {
    auto *contexts = static_cast<qd::Context *>(opaque_context);
    return valid_qwen_task(task, contexts);
  }

  __device__ static __forceinline__ void
  execute(const sg::Task &task, sg::State *state, void *opaque_context) {
    __shared__ float reduction[sg::kWorkerThreads];
    __shared__ uint32_t wave_indices[sg::kWorkerThreads / 32u];
    auto *contexts = static_cast<qd::Context *>(opaque_context);
    qd::Context *context = &contexts[state->step % contexts[0].layer_count];
    execute_task(task, state, context, reduction, wave_indices);
  }

  __device__ static __forceinline__ void
  execute_stage(const sg::Stage &stage, const sg::Task *tasks, sg::State *state,
                void *opaque_context, uint32_t worker,
                uint32_t active_workers) {
    auto *contexts = static_cast<qd::Context *>(opaque_context);
    qd::Context *context = &contexts[state->step % contexts[0].layer_count];
    const sg::Task &first_task = tasks[stage.task_offset];
    if (static_cast<qd::Opcode>(first_task.opcode) ==
        qd::Opcode::qkv_projection_f32) {
      qkv_projection_resident_stage(context, stage, tasks, worker,
                                    active_workers);
      return;
    }
    if (static_cast<qd::Opcode>(first_task.opcode) ==
        qd::Opcode::attention_output_residual_f32) {
      attention_output_resident_stage(context, stage, tasks, worker,
                                      active_workers);
      return;
    }
    if (stage.task_count == qd::kDownProjectionPartitions &&
        static_cast<qd::Opcode>(first_task.opcode) ==
            qd::Opcode::down_matvec_f32) {
      down_matvec_resident_stage(context, stage, tasks, worker, active_workers);
      return;
    }
    if (static_cast<qd::Opcode>(first_task.opcode) ==
        qd::Opcode::lm_head_top1_partition_f32) {
      for (uint32_t task_index = worker; task_index < stage.task_count;
           task_index += active_workers) {
        lm_head_top1_resident_partition(context,
                                        tasks[stage.task_offset + task_index]);
      }
      return;
    }
    for (uint32_t task_index = worker; task_index < stage.task_count;
         task_index += active_workers) {
      execute(tasks[stage.task_offset + task_index], state, opaque_context);
    }
  }

  __device__ static __forceinline__ void
  initialize_profile(void *opaque_context) {
    no_profile_initialize(opaque_context);
  }

  __device__ static __forceinline__ void
  complete_profile_stage(const sg::Stage &stage, const sg::Task *tasks,
                         void *opaque_context) {
    no_profile_complete(stage, tasks, opaque_context);
  }
};

struct QwenStageProfileWorkload : QwenWorkload {
  __device__ static __forceinline__ void
  initialize_profile(void *opaque_context) {
    auto *contexts = static_cast<qd::Context *>(opaque_context);
    qd::StageProfile *profile = contexts[0].tail->stage_profile;
    if (profile != nullptr) {
      profile->previous_tick = static_cast<uint64_t>(wall_clock64());
    }
  }

  __device__ static __forceinline__ void
  complete_profile_stage(const sg::Stage &stage, const sg::Task *tasks,
                         void *opaque_context) {
    auto *contexts = static_cast<qd::Context *>(opaque_context);
    qd::StageProfile *profile = contexts[0].tail->stage_profile;
    if (profile == nullptr) {
      return;
    }
    const uint64_t now = static_cast<uint64_t>(wall_clock64());
    const qd::ProfileStage stage_id =
        profile_stage(static_cast<qd::Opcode>(tasks[stage.task_offset].opcode));
    const uint32_t index = static_cast<uint32_t>(stage_id);
    if (index < qd::kProfileStageCount) {
      profile->stage_ticks[index] += now - profile->previous_tick;
      ++profile->stage_counts[index];
    }
    profile->previous_tick = now;
  }
};

// Preserves the Qwen Program transitions and compiled stage barriers while
// replacing arithmetic tasks with only the state writes needed to follow the
// same control-flow path. Host-side event timing of this workload measures the
// compiled scheduler floor without relying on architecture-specific counters.
struct QwenSchedulerProfileWorkload {
  __device__ static __forceinline__ bool valid_context(const sg::State *state,
                                                       void *opaque_context) {
    auto *contexts = static_cast<qd::Context *>(opaque_context);
    return valid_qwen_contexts(state, contexts);
  }

  __device__ static __forceinline__ bool valid_task(const sg::Task &task,
                                                    void *opaque_context) {
    auto *contexts = static_cast<qd::Context *>(opaque_context);
    return valid_qwen_task(task, contexts);
  }

  __device__ static __forceinline__ void
  execute(const sg::Task &task, sg::State *state, void *opaque_context) {
    __shared__ float reduction[sg::kWorkerThreads];
    __shared__ uint32_t wave_indices[sg::kWorkerThreads / 32u];
    auto *contexts = static_cast<qd::Context *>(opaque_context);
    qd::Context *context = &contexts[state->step % contexts[0].layer_count];
    const qd::Opcode opcode = static_cast<qd::Opcode>(task.opcode);
    if (opcode == qd::Opcode::residual_advance ||
        opcode == qd::Opcode::embedding_feedback_f32) {
      execute_task(task, state, context, reduction, wave_indices);
    }
  }

  __device__ static __forceinline__ void
  execute_stage(const sg::Stage &stage, const sg::Task *tasks, sg::State *state,
                void *opaque_context, uint32_t worker,
                uint32_t active_workers) {
    for (uint32_t task_index = worker; task_index < stage.task_count;
         task_index += active_workers) {
      execute(tasks[stage.task_offset + task_index], state, opaque_context);
    }
  }

  __device__ static __forceinline__ void
  initialize_profile(void *opaque_context) {
    no_profile_initialize(opaque_context);
  }

  __device__ static __forceinline__ void
  complete_profile_stage(const sg::Stage &stage, const sg::Task *tasks,
                         void *opaque_context) {
    no_profile_complete(stage, tasks, opaque_context);
  }
};

} // namespace

extern "C" __global__ void qwen_stateful_graph_resident_kernel(
    sg::State *state, const sg::Task *tasks, const sg::Program *programs,
    sg::MaskState *phase_states, uint32_t program_count,
    uint32_t total_task_count, uint32_t phase_state_count, void *opaque_context,
    const sg::Stage *stages, const sg::CompiledProgram *compiled_programs,
    uint32_t stage_count, uint32_t reserved) {
  (void)stages;
  (void)compiled_programs;
  (void)stage_count;
  (void)reserved;
  sg::device::run_resident<QwenWorkload>(state, tasks, programs, phase_states,
                                         program_count, total_task_count,
                                         phase_state_count, opaque_context);
}

extern "C" __global__ void qwen_stateful_graph_compiled_kernel(
    sg::State *state, const sg::Task *tasks, const sg::Program *programs,
    sg::MaskState *phase_states, uint32_t program_count,
    uint32_t total_task_count, uint32_t phase_state_count, void *opaque_context,
    const sg::Stage *stages, const sg::CompiledProgram *compiled_programs,
    uint32_t stage_count, uint32_t reserved) {
  (void)reserved;
  sg::device::run_compiled_resident<QwenWorkload>(
      state, tasks, programs, phase_states, program_count, total_task_count,
      phase_state_count, opaque_context, stages, compiled_programs,
      stage_count);
}

extern "C" __global__ void qwen_stateful_graph_scheduler_profile_kernel(
    sg::State *state, const sg::Task *tasks, const sg::Program *programs,
    sg::MaskState *phase_states, uint32_t program_count,
    uint32_t total_task_count, uint32_t phase_state_count, void *opaque_context,
    const sg::Stage *stages, const sg::CompiledProgram *compiled_programs,
    uint32_t stage_count, uint32_t reserved) {
  (void)reserved;
  sg::device::run_compiled_resident<QwenSchedulerProfileWorkload>(
      state, tasks, programs, phase_states, program_count, total_task_count,
      phase_state_count, opaque_context, stages, compiled_programs,
      stage_count);
}

extern "C" __global__ void qwen_stateful_graph_stage_profile_kernel(
    sg::State *state, const sg::Task *tasks, const sg::Program *programs,
    sg::MaskState *phase_states, uint32_t program_count,
    uint32_t total_task_count, uint32_t phase_state_count, void *opaque_context,
    const sg::Stage *stages, const sg::CompiledProgram *compiled_programs,
    uint32_t stage_count, uint32_t reserved) {
  (void)reserved;
  sg::device::run_compiled_resident<QwenStageProfileWorkload>(
      state, tasks, programs, phase_states, program_count, total_task_count,
      phase_state_count, opaque_context, stages, compiled_programs,
      stage_count);
}

extern "C" __global__ void
qwen_stateful_graph_matched_step_kernel(sg::State *state, const sg::Task *tasks,
                                        uint32_t task_count, uint32_t reserved,
                                        void *opaque_context) {
  (void)reserved;
  const sg::Stage stage{0, task_count};
  QwenWorkload::execute_stage(stage, tasks, state, opaque_context, blockIdx.x,
                              gridDim.x);
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0 &&
      static_cast<qd::Opcode>(tasks[0].opcode) ==
          qd::Opcode::residual_advance) {
    ++state->step;
  }
}

extern "C" __global__ void qwen_stateful_graph_matched_profile_step_kernel(
    sg::State *state, const sg::Task *tasks, uint32_t task_count,
    uint32_t profile_index, void *opaque_context) {
  auto *contexts = static_cast<qd::Context *>(opaque_context);
  qd::StageProfile *profile = contexts[0].tail->stage_profile;
  qd::StageProfileEntry *entry =
      profile != nullptr && profile_index < profile->entry_count
          ? &profile->entries[profile_index]
          : nullptr;
  if (threadIdx.x == 0 && entry != nullptr) {
    const uint64_t begin = static_cast<uint64_t>(wall_clock64());
    atomicMin(reinterpret_cast<unsigned long long *>(&entry->begin_tick),
              static_cast<unsigned long long>(begin));
  }

  const sg::Stage stage{0, task_count};
  QwenWorkload::execute_stage(stage, tasks, state, opaque_context, blockIdx.x,
                              gridDim.x);
  __syncthreads();

  if (threadIdx.x == 0) {
    if (blockIdx.x == 0 && static_cast<qd::Opcode>(tasks[0].opcode) ==
                               qd::Opcode::residual_advance) {
      ++state->step;
    }
    if (entry != nullptr) {
      const uint64_t end = static_cast<uint64_t>(wall_clock64());
      atomicMax(reinterpret_cast<unsigned long long *>(&entry->end_tick),
                static_cast<unsigned long long>(end));
    }
  }
}

extern "C" __global__ void
qwen_stateful_graph_host_step_kernel(sg::State *state, const sg::Task *tasks,
                                     uint32_t task_count, uint32_t reserved,
                                     void *opaque_context) {
  (void)reserved;
  if (blockIdx.x >= task_count) {
    return;
  }
  auto *contexts = static_cast<qd::Context *>(opaque_context);
  qd::Context *context = &contexts[state->step % contexts[0].layer_count];
  __shared__ float reduction[sg::kWorkerThreads];
  __shared__ uint32_t wave_indices[sg::kWorkerThreads / 32u];
  execute_task(tasks[blockIdx.x], state, context, reduction, wave_indices);
  __syncthreads();
  if (threadIdx.x == 0 && static_cast<qd::Opcode>(tasks[blockIdx.x].opcode) ==
                              qd::Opcode::residual_advance) {
    ++state->step;
  }
}

extern "C" __global__ void qwen_stateful_graph_host_profile_step_kernel(
    sg::State *state, const sg::Task *tasks, uint32_t task_count,
    uint32_t profile_index, void *opaque_context) {
  if (blockIdx.x >= task_count) {
    return;
  }
  auto *contexts = static_cast<qd::Context *>(opaque_context);
  qd::Context *context = &contexts[state->step % contexts[0].layer_count];
  qd::StageProfile *profile = contexts[0].tail->stage_profile;
  qd::StageProfileEntry *entry =
      profile != nullptr && profile_index < profile->entry_count
          ? &profile->entries[profile_index]
          : nullptr;
  if (threadIdx.x == 0 && entry != nullptr) {
    const uint64_t begin = static_cast<uint64_t>(wall_clock64());
    atomicMin(reinterpret_cast<unsigned long long *>(&entry->begin_tick),
              static_cast<unsigned long long>(begin));
  }

  __shared__ float reduction[sg::kWorkerThreads];
  __shared__ uint32_t wave_indices[sg::kWorkerThreads / 32u];
  execute_task(tasks[blockIdx.x], state, context, reduction, wave_indices);
  __syncthreads();

  if (threadIdx.x == 0) {
    if (static_cast<qd::Opcode>(tasks[blockIdx.x].opcode) ==
        qd::Opcode::residual_advance) {
      ++state->step;
    }
    if (entry != nullptr) {
      const uint64_t end = static_cast<uint64_t>(wall_clock64());
      atomicMax(reinterpret_cast<unsigned long long *>(&entry->end_tick),
                static_cast<unsigned long long>(end));
    }
  }
}
