#include "executor/stateful_graph/stateful_graph_abi.hpp"

namespace sg = lrrt::stateful_graph;

namespace {

__device__ __forceinline__ uint32_t load_status(const sg::State *state) {
  return *reinterpret_cast<const volatile uint32_t *>(&state->status);
}

__device__ __forceinline__ uint32_t load_epoch(const sg::State *state) {
  return *reinterpret_cast<const volatile uint32_t *>(&state->epoch);
}

__device__ void execute_task(const sg::Task &task, uint32_t *values) {
  if (task.opcode != static_cast<uint32_t>(sg::Opcode::increment_u32)) {
    return;
  }
  for (uint32_t index = task.begin + threadIdx.x; index < task.end;
       index += sg::kWorkerThreads) {
    values[index] += task.argument0;
  }
}

} // namespace

extern "C" __global__ void stateful_graph_resident_kernel(sg::State *state,
                                                          const sg::Task *tasks,
                                                          uint32_t *values) {
  const uint32_t worker = blockIdx.x;
  if (worker >= state->task_count) {
    return;
  }

  if (worker == 0 && threadIdx.x == 0) {
    if (state->abi_version != sg::kAbiVersion || state->task_count == 0 ||
        state->max_steps == 0) {
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

  const sg::Task task = tasks[worker];
  if (task.opcode != static_cast<uint32_t>(sg::Opcode::increment_u32) ||
      task.begin >= task.end) {
    if (threadIdx.x == 0) {
      state->error_code = 2;
      __threadfence();
      atomicExch(&state->status,
                 static_cast<uint32_t>(sg::Status::invalid_program));
    }
    return;
  }

  while (true) {
    const uint32_t epoch = load_epoch(state);
    execute_task(task, values);
    __syncthreads();

    if (threadIdx.x == 0) {
      __threadfence();
      const uint32_t previous = atomicAdd(&state->arrivals, 1u);
      if (previous + 1u == state->task_count) {
        state->arrivals = 0;
        const uint32_t next_step = state->step + 1u;
        state->step = next_step;
        state->executed_tasks += state->task_count;

        const bool predicate = values[0] >= state->stop_value;
        if (predicate) {
          state->status = static_cast<uint32_t>(sg::Status::complete);
        } else if (next_step >= state->max_steps) {
          state->status = static_cast<uint32_t>(sg::Status::max_steps);
        }
        __threadfence();
        atomicAdd(&state->epoch, 1u);
      } else {
        while (load_epoch(state) == epoch &&
               load_status(state) ==
                   static_cast<uint32_t>(sg::Status::running)) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
    }
    __syncthreads();

    if (load_status(state) != static_cast<uint32_t>(sg::Status::running)) {
      return;
    }
  }
}

extern "C" __global__ void
stateful_graph_host_step_kernel(const sg::Task *tasks, uint32_t task_count,
                                uint32_t reserved, uint32_t *values) {
  (void)reserved;
  if (blockIdx.x >= task_count) {
    return;
  }
  execute_task(tasks[blockIdx.x], values);
}
