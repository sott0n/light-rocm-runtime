#include "executor/stateful_graph/stateful_graph_abi.hpp"

namespace sg = lrrt::stateful_graph;

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

__device__ bool validate_program(const sg::State *state,
                                 const sg::Task *tasks) {
  if (state->abi_version != sg::kAbiVersion || state->task_count == 0 ||
      state->task_count > sg::kMaximumTasks || state->max_steps == 0) {
    return false;
  }

  const uint32_t valid_mask = all_tasks_mask(state->task_count);
  uint32_t reachable_mask = 0;
  for (uint32_t task_index = 0; task_index < state->task_count; ++task_index) {
    const sg::Task &task = tasks[task_index];
    const uint32_t task_mask = 1u << task_index;
    if (task.opcode != static_cast<uint32_t>(sg::Opcode::increment_u32) ||
        task.begin >= task.end || (task.dependency_mask & ~valid_mask) != 0 ||
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

  if (worker == 0 && threadIdx.x == 0) {
    if (!validate_program(state, tasks)) {
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
      execute_task(tasks[selected_task], values);
      __syncthreads();

      if (threadIdx.x == 0) {
        __threadfence();
        const uint32_t previous = atomicOr(&state->completed_mask, task_mask);
        if ((previous | task_mask) == valid_mask) {
          const uint32_t next_step = state->step + 1u;
          state->step = next_step;
          state->executed_tasks += state->task_count;

          const bool predicate = values[0] >= state->stop_value;
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
stateful_graph_host_step_kernel(const sg::Task *tasks, uint32_t task_count,
                                uint32_t reserved, uint32_t *values) {
  (void)reserved;
  if (blockIdx.x >= task_count) {
    return;
  }
  execute_task(tasks[blockIdx.x], values);
}
