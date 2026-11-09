#ifndef LRRT_PERSISTENT_GRAPH_COMPILED_SCHEDULER_HPP_
#define LRRT_PERSISTENT_GRAPH_COMPILED_SCHEDULER_HPP_

#include "lrrt/persistent_graph/device_scheduler.hpp"

namespace lrrt::persistent_graph::device {

struct CompiledSchedulerScratch {
  uint32_t validation_status;
};

namespace compiled_detail {

template <typename Workload>
__device__ __forceinline__ bool
validate_plan(const State *state, const Task *tasks, const Program *programs,
              MaskState *phase_states, const Stage *stages,
              const CompiledProgram *compiled_programs, uint32_t program_count,
              uint32_t total_task_count, uint32_t phase_state_count,
              uint32_t stage_count, void *opaque_context) {
  if (!detail::validate_programs<Workload>(state, tasks, programs, phase_states,
                                           program_count, total_task_count,
                                           phase_state_count, opaque_context) ||
      stages == nullptr || compiled_programs == nullptr || stage_count == 0) {
    return false;
  }

  for (uint32_t program_index = 0; program_index < program_count;
       ++program_index) {
    const Program program = programs[program_index];
    const CompiledProgram compiled = compiled_programs[program_index];
    if (compiled.stage_count == 0 || compiled.stage_offset > stage_count ||
        compiled.stage_count > stage_count - compiled.stage_offset) {
      return false;
    }

    uint32_t expected_task_offset = program.task_offset;
    uint64_t completed_mask = 0;
    for (uint32_t stage_index = 0; stage_index < compiled.stage_count;
         ++stage_index) {
      const Stage stage = stages[compiled.stage_offset + stage_index];
      const uint32_t completed_count =
          expected_task_offset - program.task_offset;
      if (stage.task_count == 0 || stage.task_offset != expected_task_offset ||
          stage.task_count > program.task_count - completed_count) {
        return false;
      }
      for (uint32_t task_index = 0; task_index < stage.task_count;
           ++task_index) {
        const uint32_t local_task = completed_count + task_index;
        const uint64_t dependencies =
            detail::task_dependencies(tasks[stage.task_offset + task_index]);
        if ((dependencies & completed_mask) != dependencies) {
          return false;
        }
      }
      const uint64_t stage_mask = detail::all_tasks_mask(stage.task_count)
                                  << completed_count;
      completed_mask |= stage_mask;
      expected_task_offset += stage.task_count;
    }
    if (expected_task_offset != program.task_offset + program.task_count) {
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ void finish_program(State *state,
                                               const Program &program,
                                               uint32_t generation,
                                               uint32_t phase_state_count) {
  const bool advances_step = (program.flags & kProgramAdvancesStep) != 0;
  const uint32_t next_step = state->step + static_cast<uint32_t>(advances_step);
  state->step = next_step;
  state->executed_tasks += program.task_count;

  const bool predicate = (program.flags & kProgramEvaluatesPredicate) != 0 &&
                         state->predicate_value >= state->stop_value;
  if (predicate) {
    atomicExch(&state->status, static_cast<uint32_t>(Status::complete));
  } else if (next_step >= state->max_steps) {
    atomicExch(&state->status, static_cast<uint32_t>(Status::max_steps));
  } else {
    const uint32_t next_generation = generation + 1u;
    if (next_generation >= phase_state_count) {
      state->error_code = 2;
      atomicExch(&state->status,
                 static_cast<uint32_t>(Status::invalid_program));
    } else {
      uint32_t next_program = program.next_program;
      if ((program.flags & kProgramBranchesOnState) != 0 &&
          state->branch_value != 0) {
        next_program = program.branch_program;
      }
      atomicExch(&state->phase, make_phase(next_generation, next_program));
    }
  }
}

} // namespace compiled_detail

// Executes a host-compiled sequence of dependency-independent stages. This
// fast path requires every launched workgroup to be concurrently resident so
// that its stage barriers cannot be blocked by an unscheduled workgroup.
template <typename Workload>
__device__ __forceinline__ void run_compiled_resident(
    State *state, const Task *tasks, const Program *programs,
    MaskState *phase_states, uint32_t program_count, uint32_t total_task_count,
    uint32_t phase_state_count, void *opaque_context, const Stage *stages,
    const CompiledProgram *compiled_programs, uint32_t stage_count) {
  __shared__ CompiledSchedulerScratch scratch_storage;
  CompiledSchedulerScratch *scratch = &scratch_storage;
  const uint32_t worker = blockIdx.x;

  if (worker == 0) {
    if (threadIdx.x == 0) {
      scratch->validation_status =
          compiled_detail::validate_plan<Workload>(
              state, tasks, programs, phase_states, stages, compiled_programs,
              program_count, total_task_count, phase_state_count, stage_count,
              opaque_context)
              ? static_cast<uint32_t>(Status::running)
              : static_cast<uint32_t>(Status::invalid_program);
      if (scratch->validation_status ==
          static_cast<uint32_t>(Status::running)) {
        Workload::initialize_profile(opaque_context);
      }
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
    const uint32_t phase = detail::load_phase(state);
    const uint32_t generation = phase_generation(phase);
    const uint32_t program_index = phase_program(phase);
    const Program program = programs[program_index];
    const CompiledProgram compiled = compiled_programs[program_index];
    MaskState *barrier = &phase_states[generation];

    for (uint32_t stage_index = 0; stage_index < compiled.stage_count;
         ++stage_index) {
      const Stage stage = stages[compiled.stage_offset + stage_index];
      const uint32_t active_workers = state->worker_count;
      if (worker < active_workers) {
        Workload::execute_stage(stage, tasks, state, opaque_context, worker,
                                active_workers);
      }
      __syncthreads();

      if (threadIdx.x == 0) {
        __threadfence();
        // Every resident worker participates even when it had no task in this
        // stage. Otherwise an idle worker can lag behind, skip a completed
        // stage, and miss a later stage where it becomes active.
        const uint32_t arrived = atomicAdd(&barrier->claimed_mask, 1u) + 1u;
        if (arrived == state->worker_count) {
          atomicExch(&barrier->claimed_mask, 0u);
          if (stage_index + 1u == compiled.stage_count) {
            compiled_detail::finish_program(state, program, generation,
                                            phase_state_count);
          }
          Workload::complete_profile_stage(stage, tasks, opaque_context);
          __threadfence();
          atomicExch(&barrier->completed_mask, stage_index + 1u);
        }
        while (detail::load_mask(&barrier->completed_mask) < stage_index + 1u) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      __syncthreads();
    }

    if (detail::load_status(state) != static_cast<uint32_t>(Status::running)) {
      return;
    }
  }
}

} // namespace lrrt::persistent_graph::device

#endif // LRRT_PERSISTENT_GRAPH_COMPILED_SCHEDULER_HPP_
