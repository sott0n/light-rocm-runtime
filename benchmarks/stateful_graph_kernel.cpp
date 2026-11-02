#include "executor/stateful_graph/qwen_decode_abi.hpp"

namespace sg = lrrt::stateful_graph;
namespace qd = lrrt::stateful_graph::qwen_decode;

namespace {

__device__ __forceinline__ uint32_t load_status(const sg::State *state) {
  return *reinterpret_cast<const volatile uint32_t *>(&state->status);
}

__device__ __forceinline__ uint32_t load_epoch(sg::State *state) {
  return atomicAdd(&state->epoch, 0u);
}

__device__ __forceinline__ uint32_t load_transition_barrier(sg::State *state) {
  return atomicAdd(&state->transition_barrier, 0u);
}

__device__ __forceinline__ uint32_t load_mask(uint32_t *mask) {
  return atomicAdd(mask, 0u);
}

__device__ __forceinline__ uint64_t load_completed_mask(sg::State *state,
                                                        uint32_t task_count) {
  const uint64_t low = load_mask(&state->completed_mask);
  const uint64_t high =
      task_count > 32 ? load_mask(&state->completed_mask_high) : 0;
  return low | (high << 32u);
}

__device__ __forceinline__ uint64_t task_dependencies(const sg::Task &task) {
  return static_cast<uint64_t>(task.dependency_mask) |
         (static_cast<uint64_t>(task.dependency_mask_high) << 32u);
}

__device__ __forceinline__ uint64_t all_tasks_mask(uint32_t task_count) {
  return task_count == sg::kMaximumTasks
             ? UINT64_MAX
             : (static_cast<uint64_t>(1) << task_count) - 1u;
}

__device__ bool valid_task(const sg::Task &task, const qd::Context *context) {
  switch (static_cast<qd::Opcode>(task.opcode)) {
  case qd::Opcode::attention_rmsnorm_f32:
  case qd::Opcode::mlp_rmsnorm_f32:
  case qd::Opcode::down_matvec_f32:
    return task.begin < task.end && task.end <= context->hidden_size;
  case qd::Opcode::qkv_projection_f32:
    return task.begin < task.end && task.end <= qd::kPackedQkvSize;
  case qd::Opcode::residual_advance:
    return task.begin < task.end && task.end <= context->hidden_size &&
           task.argument0 != 0;
  case qd::Opcode::gate_up_silu_f32:
    return task.begin < task.end && task.end <= context->intermediate_size;
  }
  return false;
}

__device__ bool validate_program(const sg::State *state, const sg::Task *tasks,
                                 const qd::Context *context) {
  if (state->abi_version != sg::kAbiVersion || state->task_count == 0 ||
      state->task_count > sg::kMaximumTasks || state->max_steps == 0 ||
      (state->transition_barrier >> 16u) == 0 ||
      (state->transition_barrier & UINT16_MAX) != 0 || context == nullptr ||
      context->hidden_size != qd::kHiddenSize ||
      context->query_size != qd::kQuerySize ||
      context->kv_size != qd::kKvSize ||
      context->intermediate_size != qd::kIntermediateSize) {
    return false;
  }

  const uint64_t valid_mask = all_tasks_mask(state->task_count);
  uint64_t reachable_mask = 0;
  for (uint32_t task_index = 0; task_index < state->task_count; ++task_index) {
    const sg::Task &task = tasks[task_index];
    const uint64_t task_mask = static_cast<uint64_t>(1) << task_index;
    const uint64_t dependencies = task_dependencies(task);
    if (!valid_task(task, context) || (dependencies & ~valid_mask) != 0 ||
        (dependencies & task_mask) != 0) {
      return false;
    }
  }

  // A small bitmask topological walk rejects cycles before workers enter the
  // persistent scheduling loop.
  for (uint32_t pass = 0; pass < state->task_count; ++pass) {
    for (uint32_t task_index = 0; task_index < state->task_count;
         ++task_index) {
      const uint64_t task_mask = static_cast<uint64_t>(1) << task_index;
      const uint64_t dependencies = task_dependencies(tasks[task_index]);
      if ((reachable_mask & task_mask) == 0 &&
          (dependencies & reachable_mask) == dependencies) {
        reachable_mask |= task_mask;
      }
    }
  }
  return reachable_mask == valid_mask;
}

__device__ __forceinline__ float wave_sum(float value) {
  for (uint32_t offset = 16; offset != 0; offset >>= 1u) {
    value += __shfl_down(value, offset, 32);
  }
  return value;
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
    for (uint32_t column = lane; column < context->hidden_size; column += 32u) {
      const float input = context->mlp_normalized[column];
      gate = fmaf(input, context->gate_weight[row_offset + column], gate);
      up = fmaf(input, context->up_weight[row_offset + column], up);
    }
    gate = wave_sum(gate);
    up = wave_sum(up);
    if (lane == 0) {
      context->activated[row] = (gate / (1.0f + expf(-gate))) * up;
    }
  }
}

__device__ void execute_task(const sg::Task &task, sg::State *state,
                             qd::Context *context, float *reduction) {
  const qd::Opcode opcode = static_cast<qd::Opcode>(task.opcode);
  if (opcode == qd::Opcode::attention_rmsnorm_f32 ||
      opcode == qd::Opcode::mlp_rmsnorm_f32) {
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
    const float *norm_weight = opcode == qd::Opcode::attention_rmsnorm_f32
                                   ? context->attention_norm_weight
                                   : context->mlp_norm_weight;
    float *normalized = opcode == qd::Opcode::attention_rmsnorm_f32
                            ? context->attention_normalized
                            : context->mlp_normalized;
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

  if (opcode == qd::Opcode::gate_up_silu_f32) {
    gate_up_silu_range_wave(context, task);
    return;
  }

  if (opcode == qd::Opcode::down_matvec_f32) {
    matvec_range_wave(context->activated, context->down_weight,
                      context->projected, context->intermediate_size, task);
    return;
  }

  for (uint32_t index = task.begin + threadIdx.x; index < task.end;
       index += sg::kWorkerThreads) {
    context->hidden[index] += context->projected[index];
  }
  if (threadIdx.x == 0) {
    state->predicate_value += task.argument0;
  }
}

} // namespace

extern "C" __global__ void
stateful_graph_resident_kernel(sg::State *state, const sg::Task *tasks,
                               void *opaque_context) {
  auto *context = static_cast<qd::Context *>(opaque_context);
  const uint32_t worker = blockIdx.x;

  if (worker == 0 && threadIdx.x == 0) {
    if (!validate_program(state, tasks, context)) {
      state->error_code = 1;
      __threadfence();
      atomicExch(&state->status,
                 static_cast<uint32_t>(sg::Status::invalid_program));
    } else {
      atomicExch(&state->status, static_cast<uint32_t>(sg::Status::running));
    }
  }

  while (load_status(state) == static_cast<uint32_t>(sg::Status::ready)) {
    __builtin_amdgcn_s_sleep(1);
  }
  if (load_status(state) != static_cast<uint32_t>(sg::Status::running)) {
    return;
  }

  __shared__ uint32_t selected_task;
  __shared__ uint32_t acknowledged_epoch;
  __shared__ float reduction[sg::kWorkerThreads];
  if (threadIdx.x == 0) {
    acknowledged_epoch = UINT32_MAX;
  }
  __syncthreads();
  const uint64_t valid_mask = all_tasks_mask(state->task_count);
  while (true) {
    if (threadIdx.x == 0) {
      selected_task = sg::kNoTask;
      const uint32_t epoch_before = load_epoch(state);
      if ((epoch_before & 1u) != 0) {
        if (acknowledged_epoch != epoch_before) {
          acknowledged_epoch = epoch_before;
          atomicAdd(&state->transition_barrier, 1u);
        }
      } else {
        const uint32_t claimed_mask = load_mask(&state->claimed_mask);
        const uint32_t claimed_mask_high =
            state->task_count > 32 ? load_mask(&state->claimed_mask_high) : 0;
        const uint64_t completed_mask =
            load_completed_mask(state, state->task_count);
        const uint32_t epoch_after = load_epoch(state);
        if (epoch_before == epoch_after) {
          for (uint32_t offset = 0; offset < state->task_count; ++offset) {
            const uint32_t task_index = (worker + offset) % state->task_count;
            const uint32_t mask_word = task_index >> 5u;
            const uint32_t task_mask = 1u << (task_index & 31u);
            const sg::Task &task = tasks[task_index];
            const uint32_t claimed =
                mask_word == 0 ? claimed_mask : claimed_mask_high;
            const uint64_t dependencies = task_dependencies(task);
            if ((claimed & task_mask) == 0 &&
                (dependencies & completed_mask) == dependencies) {
              uint32_t *claimed_word = mask_word == 0
                                           ? &state->claimed_mask
                                           : &state->claimed_mask_high;
              const uint32_t previous = atomicOr(claimed_word, task_mask);
              if ((previous & task_mask) == 0) {
                selected_task = task_index;
                break;
              }
            }
          }
        }
      }
    }
    __syncthreads();

    if (selected_task == sg::kNoTask) {
      if (threadIdx.x == 0) {
        __builtin_amdgcn_s_sleep(1);
      }
    } else {
      const uint32_t mask_word = selected_task >> 5u;
      const uint32_t task_mask = 1u << (selected_task & 31u);
      execute_task(tasks[selected_task], state, context, reduction);
      __syncthreads();

      if (threadIdx.x == 0) {
        __threadfence();
        uint32_t *completed_word = mask_word == 0 ? &state->completed_mask
                                                  : &state->completed_mask_high;
        const uint32_t previous = atomicOr(completed_word, task_mask);
        const bool all_complete =
            state->task_count <= 32
                ? (previous | task_mask) == static_cast<uint32_t>(valid_mask)
                : load_completed_mask(state, state->task_count) == valid_mask;
        if (all_complete) {
          // Odd epochs close claiming while masks transition. Workers only
          // claim tasks from a stable even epoch snapshot.
          const uint32_t transition_epoch = atomicAdd(&state->epoch, 1u) + 1u;
          const uint32_t next_step = state->step + 1u;
          state->step = next_step;
          state->executed_tasks += state->task_count;

          const bool predicate = state->predicate_value >= state->stop_value;
          if (predicate) {
            state->status = static_cast<uint32_t>(sg::Status::complete);
          } else if (next_step >= state->max_steps) {
            state->status = static_cast<uint32_t>(sg::Status::max_steps);
          } else {
            acknowledged_epoch = transition_epoch;
            atomicAdd(&state->transition_barrier, 1u);
            const uint32_t worker_count = state->transition_barrier >> 16u;
            while ((load_transition_barrier(state) & UINT16_MAX) !=
                   worker_count) {
              __builtin_amdgcn_s_sleep(1);
            }

            // Every admitted worker is now outside task selection. Reset both
            // words before publishing the next even epoch.
            state->completed_mask = 0;
            state->completed_mask_high = 0;
            state->claimed_mask = 0;
            state->claimed_mask_high = 0;
            atomicAnd(&state->transition_barrier, UINT32_C(0xffff0000));
          }
          __threadfence();
          atomicAdd(&state->epoch, 1u);
        }
      }
    }
    __syncthreads();

    if (load_status(state) != static_cast<uint32_t>(sg::Status::running))
      return;
  }
}

extern "C" __global__ void
stateful_graph_host_step_kernel(sg::State *state, const sg::Task *tasks,
                                uint32_t task_count, uint32_t reserved,
                                void *opaque_context) {
  (void)reserved;
  if (blockIdx.x >= task_count) {
    return;
  }
  auto *context = static_cast<qd::Context *>(opaque_context);
  __shared__ float reduction[sg::kWorkerThreads];
  execute_task(tasks[blockIdx.x], state, context, reduction);
}
