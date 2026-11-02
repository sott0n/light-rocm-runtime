#include "executor/stateful_graph/qwen_ffn_abi.hpp"

namespace sg = lrrt::stateful_graph;
namespace qf = lrrt::stateful_graph::qwen_ffn;

namespace {

__device__ __forceinline__ uint32_t load_status(const sg::State *state) {
  return *reinterpret_cast<const volatile uint32_t *>(&state->status);
}

__device__ __forceinline__ uint32_t load_claimed_mask(sg::State *state) {
  return atomicAdd(&state->claimed_mask, 0u);
}

__device__ __forceinline__ uint32_t load_completed_mask(sg::State *state) {
  return atomicAdd(&state->completed_mask, 0u);
}

__device__ __forceinline__ uint32_t all_tasks_mask(uint32_t task_count) {
  return task_count == sg::kMaximumTasks ? UINT32_MAX : (1u << task_count) - 1u;
}

__device__ bool valid_task(const sg::Task &task, const qf::Context *context) {
  switch (static_cast<qf::Opcode>(task.opcode)) {
  case qf::Opcode::rmsnorm_f32:
  case qf::Opcode::down_matvec_f32:
    return task.begin < task.end && task.end <= context->hidden_size;
  case qf::Opcode::residual_advance:
    return task.begin < task.end && task.end <= context->hidden_size &&
           task.argument0 != 0;
  case qf::Opcode::gate_up_silu_f32:
    return task.begin < task.end && task.end <= context->intermediate_size;
  }
  return false;
}

__device__ bool validate_program(const sg::State *state, const sg::Task *tasks,
                                 const qf::Context *context) {
  if (state->abi_version != sg::kAbiVersion || state->task_count == 0 ||
      state->task_count > sg::kMaximumTasks || state->max_steps == 0 ||
      context == nullptr || context->hidden_size != qf::kHiddenSize ||
      context->intermediate_size != qf::kIntermediateSize) {
    return false;
  }

  const uint32_t valid_mask = all_tasks_mask(state->task_count);
  uint32_t reachable_mask = 0;
  for (uint32_t task_index = 0; task_index < state->task_count; ++task_index) {
    const sg::Task &task = tasks[task_index];
    const uint32_t task_mask = 1u << task_index;
    if (!valid_task(task, context) ||
        (task.dependency_mask & ~valid_mask) != 0 ||
        (task.dependency_mask & task_mask) != 0) {
      return false;
    }
  }

  // A small bitmask topological walk rejects cycles before workers enter the
  // persistent scheduling loop.
  for (uint32_t pass = 0; pass < state->task_count; ++pass) {
    for (uint32_t task_index = 0; task_index < state->task_count;
         ++task_index) {
      const uint32_t task_mask = 1u << task_index;
      if ((reachable_mask & task_mask) == 0 &&
          (tasks[task_index].dependency_mask & reachable_mask) ==
              tasks[task_index].dependency_mask) {
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

__device__ void gate_up_silu_range_wave(const qf::Context *context,
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
      const float input = context->normalized[column];
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
                             qf::Context *context, float *reduction) {
  const qf::Opcode opcode = static_cast<qf::Opcode>(task.opcode);
  if (opcode == qf::Opcode::rmsnorm_f32) {
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
    for (uint32_t index = task.begin + threadIdx.x; index < task.end;
         index += sg::kWorkerThreads) {
      context->normalized[index] =
          context->hidden[index] * inverse_rms * context->norm_weight[index];
    }
    return;
  }

  if (opcode == qf::Opcode::gate_up_silu_f32) {
    gate_up_silu_range_wave(context, task);
    return;
  }

  if (opcode == qf::Opcode::down_matvec_f32) {
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
  auto *context = static_cast<qf::Context *>(opaque_context);
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
  __shared__ float reduction[sg::kWorkerThreads];
  const uint32_t valid_mask = all_tasks_mask(state->task_count);
  while (true) {
    if (threadIdx.x == 0) {
      selected_task = sg::kNoTask;
      // claimed_mask publishes a new epoch only after completed_mask resets.
      const uint32_t claimed_mask = load_claimed_mask(state);
      const uint32_t completed_mask = load_completed_mask(state);
      for (uint32_t offset = 0; offset < state->task_count; ++offset) {
        const uint32_t task_index = (worker + offset) % state->task_count;
        const uint32_t task_mask = 1u << task_index;
        const sg::Task &task = tasks[task_index];
        if ((claimed_mask & task_mask) == 0 &&
            (task.dependency_mask & completed_mask) == task.dependency_mask) {
          const uint32_t previous = atomicOr(&state->claimed_mask, task_mask);
          if ((previous & task_mask) == 0) {
            selected_task = task_index;
            break;
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
      const uint32_t task_mask = 1u << selected_task;
      execute_task(tasks[selected_task], state, context, reduction);
      __syncthreads();

      if (threadIdx.x == 0) {
        __threadfence();
        const uint32_t previous = atomicOr(&state->completed_mask, task_mask);
        if ((previous | task_mask) == valid_mask) {
          const uint32_t next_step = state->step + 1u;
          state->step = next_step;
          state->executed_tasks += state->task_count;

          const bool predicate = state->predicate_value >= state->stop_value;
          if (predicate) {
            state->status = static_cast<uint32_t>(sg::Status::complete);
          } else if (next_step >= state->max_steps) {
            state->status = static_cast<uint32_t>(sg::Status::max_steps);
          } else {
            // Keep claimed_mask closed while completed_mask is reset.
            // Publishing an empty claimed mask makes the next epoch's roots
            // claimable.
            state->completed_mask = 0;
            __threadfence();
            atomicExch(&state->claimed_mask, 0u);
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
  auto *context = static_cast<qf::Context *>(opaque_context);
  __shared__ float reduction[sg::kWorkerThreads];
  execute_task(tasks[blockIdx.x], state, context, reduction);
}
