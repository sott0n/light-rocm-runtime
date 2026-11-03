#ifndef LRRT_STATEFUL_GRAPH_DEVICE_SCHEDULER_HPP_
#define LRRT_STATEFUL_GRAPH_DEVICE_SCHEDULER_HPP_

#include "executor/stateful_graph/stateful_graph_abi.hpp"

namespace lrrt::stateful_graph::device {

// Per-workgroup storage supplied by a resident kernel. Workload-specific
// shared memory stays outside the scheduler and is owned by the task executor.
struct SchedulerScratch {
  uint32_t validation_status;
  uint32_t selected_task;
  uint32_t selected_program;
  uint32_t selected_generation;
};

namespace detail {

__device__ __forceinline__ uint32_t load_status(const State *state) {
  return *reinterpret_cast<const volatile uint32_t *>(&state->status);
}

__device__ __forceinline__ uint32_t load_phase(State *state) {
  return atomicAdd(&state->phase, 0u);
}

__device__ __forceinline__ uint32_t load_mask(uint32_t *mask) {
  return atomicAdd(mask, 0u);
}

__device__ __forceinline__ uint64_t load_completed_mask(MaskState *mask_state,
                                                        uint32_t task_count) {
  const uint64_t low = load_mask(&mask_state->completed_mask);
  const uint64_t high =
      task_count > 32 ? load_mask(&mask_state->completed_mask_high) : 0;
  return low | (high << 32u);
}

__device__ __forceinline__ uint64_t task_dependencies(const Task &task) {
  return static_cast<uint64_t>(task.dependency_mask) |
         (static_cast<uint64_t>(task.dependency_mask_high) << 32u);
}

__device__ __forceinline__ uint64_t all_tasks_mask(uint32_t task_count) {
  return task_count == kMaximumTasks
             ? UINT64_MAX
             : (static_cast<uint64_t>(1) << task_count) - 1u;
}

template <typename Workload>
__device__ __forceinline__ bool
validate_programs(const State *state, const Task *tasks,
                  const Program *programs, MaskState *phase_states,
                  uint32_t program_count, uint32_t total_task_count,
                  uint32_t phase_state_count, void *opaque_context) {
  if (state->abi_version != kAbiVersion || program_count == 0 ||
      total_task_count == 0 || program_count > kMaximumPrograms ||
      phase_program(state->phase) >= program_count ||
      phase_generation(state->phase) >= phase_state_count ||
      state->max_steps == 0 || tasks == nullptr || programs == nullptr ||
      phase_states == nullptr || phase_state_count == 0 ||
      state->worker_count == 0 ||
      state->worker_count > kMaximumResidentWorkers ||
      !Workload::valid_context(state, opaque_context)) {
    return false;
  }

  for (uint32_t program_index = 0; program_index < program_count;
       ++program_index) {
    const Program &program = programs[program_index];
    if (program.task_count == 0 || program.task_count > kMaximumTasks ||
        program.task_offset > total_task_count ||
        program.task_count > total_task_count - program.task_offset ||
        program.next_program >= program_count ||
        (program.flags & ~(kProgramAdvancesStep | kProgramEvaluatesPredicate |
                           kProgramBranchesOnState)) != 0 ||
        ((program.flags & kProgramBranchesOnState) != 0 &&
         program.branch_program >= program_count)) {
      return false;
    }

    const Task *program_tasks = tasks + program.task_offset;
    const uint64_t valid_mask = all_tasks_mask(program.task_count);
    uint64_t reachable_mask = 0;
    for (uint32_t task_index = 0; task_index < program.task_count;
         ++task_index) {
      const Task &task = program_tasks[task_index];
      const uint64_t task_mask = static_cast<uint64_t>(1) << task_index;
      const uint64_t dependencies = task_dependencies(task);
      if (!Workload::valid_task(task, opaque_context) ||
          (dependencies & ~valid_mask) != 0 ||
          (dependencies & task_mask) != 0) {
        return false;
      }
    }

    // A small bitmask topological walk rejects cycles before workers enter the
    // persistent scheduling loop.
    for (uint32_t pass = 0; pass < program.task_count; ++pass) {
      for (uint32_t task_index = 0; task_index < program.task_count;
           ++task_index) {
        const uint64_t task_mask = static_cast<uint64_t>(1) << task_index;
        const uint64_t dependencies =
            task_dependencies(program_tasks[task_index]);
        if ((reachable_mask & task_mask) == 0 &&
            (dependencies & reachable_mask) == dependencies) {
          reachable_mask |= task_mask;
        }
      }
    }
    if (reachable_mask != valid_mask) {
      return false;
    }
  }
  return true;
}

} // namespace detail

// Runs a dependency-scheduled graph until its device predicate, step bound, or
// validation terminates it. Workload supplies valid_context(), valid_task(),
// and execute() so this control loop contains no model semantics.
template <typename Workload>
__device__ __forceinline__ void
run_resident(State *state, const Task *tasks, const Program *programs,
             MaskState *phase_states, uint32_t program_count,
             uint32_t total_task_count, uint32_t phase_state_count,
             void *opaque_context) {
  __shared__ SchedulerScratch scratch_storage;
  SchedulerScratch *scratch = &scratch_storage;
  const uint32_t worker = blockIdx.x;
  if (worker == 0) {
    if (threadIdx.x == 0) {
      scratch->validation_status =
          detail::validate_programs<Workload>(
              state, tasks, programs, phase_states, program_count,
              total_task_count, phase_state_count, opaque_context)
              ? static_cast<uint32_t>(Status::running)
              : static_cast<uint32_t>(Status::invalid_program);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      if (scratch->validation_status ==
          static_cast<uint32_t>(Status::invalid_program)) {
        state->error_code = 1;
      }
      __threadfence();
      atomicExch(&state->status, scratch->validation_status);
    }
    __syncthreads();
  } else {
    while (detail::load_status(state) == static_cast<uint32_t>(Status::ready)) {
      __builtin_amdgcn_s_sleep(1);
    }
  }

  if (detail::load_status(state) != static_cast<uint32_t>(Status::running)) {
    return;
  }

  while (true) {
    if (threadIdx.x == 0) {
      scratch->selected_task = kNoTask;
      const uint32_t phase_before = detail::load_phase(state);
      const uint32_t generation = phase_generation(phase_before);
      const uint32_t program_index = phase_program(phase_before);
      const Program program = programs[program_index];
      const Task *program_tasks = tasks + program.task_offset;
      MaskState *mask_state = &phase_states[generation];
      const uint32_t claimed_mask =
          detail::load_mask(&mask_state->claimed_mask);
      const uint32_t claimed_mask_high =
          program.task_count > 32
              ? detail::load_mask(&mask_state->claimed_mask_high)
              : 0;
      const uint64_t completed_mask =
          detail::load_completed_mask(mask_state, program.task_count);
      const uint32_t phase_after = detail::load_phase(state);
      if (phase_before == phase_after) {
        for (uint32_t offset = 0; offset < program.task_count; ++offset) {
          const uint32_t task_index = (worker + offset) % program.task_count;
          const uint32_t mask_word = task_index >> 5u;
          const uint32_t task_mask = 1u << (task_index & 31u);
          const Task &task = program_tasks[task_index];
          const uint32_t claimed =
              mask_word == 0 ? claimed_mask : claimed_mask_high;
          const uint64_t dependencies = detail::task_dependencies(task);
          if ((claimed & task_mask) == 0 &&
              (dependencies & completed_mask) == dependencies) {
            uint32_t *claimed_word = mask_word == 0
                                         ? &mask_state->claimed_mask
                                         : &mask_state->claimed_mask_high;
            const uint32_t previous = atomicOr(claimed_word, task_mask);
            if ((previous & task_mask) == 0) {
              scratch->selected_task = task_index;
              scratch->selected_program = program_index;
              scratch->selected_generation = generation;
              break;
            }
          }
        }
      }
    }
    __syncthreads();

    if (scratch->selected_task == kNoTask) {
      if (threadIdx.x == 0) {
        __builtin_amdgcn_s_sleep(1);
      }
    } else {
      const Program program = programs[scratch->selected_program];
      const Task *program_tasks = tasks + program.task_offset;
      MaskState *mask_state = &phase_states[scratch->selected_generation];
      const uint32_t mask_word = scratch->selected_task >> 5u;
      const uint32_t task_mask = 1u << (scratch->selected_task & 31u);
      Workload::execute(program_tasks[scratch->selected_task], state,
                        opaque_context);
      __syncthreads();

      if (threadIdx.x == 0) {
        __threadfence();
        uint32_t *completed_word = mask_word == 0
                                       ? &mask_state->completed_mask
                                       : &mask_state->completed_mask_high;
        atomicOr(completed_word, task_mask);
        // The dependency masks may span two independently observed atomic
        // words. A single completion counter elects exactly one transition
        // owner without requiring a coherent 64-bit mask snapshot.
        const uint32_t completed_count =
            atomicAdd(&mask_state->completed_count, 1u) + 1u;
        if (completed_count == program.task_count) {
          const bool advances_step =
              (program.flags & kProgramAdvancesStep) != 0;
          const uint32_t next_step =
              state->step + static_cast<uint32_t>(advances_step);
          state->step = next_step;
          state->executed_tasks += program.task_count;

          const bool predicate =
              (program.flags & kProgramEvaluatesPredicate) != 0 &&
              state->predicate_value >= state->stop_value;
          if (predicate) {
            __threadfence();
            atomicExch(&state->status, static_cast<uint32_t>(Status::complete));
          } else if (next_step >= state->max_steps) {
            __threadfence();
            atomicExch(&state->status,
                       static_cast<uint32_t>(Status::max_steps));
          } else {
            const uint32_t next_generation = scratch->selected_generation + 1u;
            if (next_generation >= phase_state_count) {
              state->error_code = 2;
              __threadfence();
              atomicExch(&state->status,
                         static_cast<uint32_t>(Status::invalid_program));
            } else {
              uint32_t next_program = program.next_program;
              if ((program.flags & kProgramBranchesOnState) != 0 &&
                  state->branch_value != 0) {
                next_program = program.branch_program;
              }
              __threadfence();
              atomicExch(&state->phase,
                         make_phase(next_generation, next_program));
            }
          }
        }
      }
    }
    __syncthreads();

    if (detail::load_status(state) != static_cast<uint32_t>(Status::running)) {
      return;
    }
  }
}

} // namespace lrrt::stateful_graph::device

#endif // LRRT_STATEFUL_GRAPH_DEVICE_SCHEDULER_HPP_
