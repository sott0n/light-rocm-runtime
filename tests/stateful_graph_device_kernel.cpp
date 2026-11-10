#include "lrrt/persistent_graph/compiled_scheduler.hpp"
#include "lrrt/persistent_graph/device_scheduler.hpp"
#include "tests/stateful_graph_device_context.hpp"

namespace sg = lrrt::persistent_graph;
namespace fixture = lrrt::persistent_graph::test;

namespace {

struct SyntheticWorkload {
  __device__ static __forceinline__ bool valid_context(const sg::State *,
                                                       void *opaque_context) {
    const auto *context = static_cast<fixture::Context *>(opaque_context);
    if (context != nullptr && context->validation_calls != nullptr) {
      atomicAdd(context->validation_calls, 1u);
    }
    return context != nullptr && context->magic == fixture::kContextMagic &&
           context->value_count == fixture::kValueCount &&
           context->values != nullptr && context->step_checksums != nullptr &&
           context->branch_steps != nullptr &&
           context->task_executions != nullptr &&
           context->validation_calls != nullptr;
  }

  __device__ static __forceinline__ bool valid_task(const sg::Task &task,
                                                    void *) {
    if (task.argument1 >= fixture::kTaskCount) {
      return false;
    }
    switch (static_cast<fixture::Opcode>(task.opcode)) {
    case fixture::Opcode::add_range:
      return task.begin < task.end && task.end <= fixture::kValueCount &&
             task.argument0 != 0;
    case fixture::Opcode::checksum:
      return task.begin == 0 && task.end == fixture::kValueCount;
    case fixture::Opcode::select_branch:
      return task.begin == 0 && task.end == 1 && task.argument0 != 0;
    case fixture::Opcode::advance_predicate:
      return task.begin == 0 && task.end == 1 && task.argument0 != 0;
    }
    return false;
  }

  __device__ static __forceinline__ void
  execute(const sg::Task &task, sg::State *state, void *opaque_context) {
    auto *context = static_cast<fixture::Context *>(opaque_context);
    if (threadIdx.x == 0) {
      ++context->task_executions[task.argument1];
    }

    switch (static_cast<fixture::Opcode>(task.opcode)) {
    case fixture::Opcode::add_range:
      for (uint32_t index = task.begin + threadIdx.x; index < task.end;
           index += sg::kWorkerThreads) {
        context->values[index] += task.argument0 + state->step;
      }
      return;
    case fixture::Opcode::checksum:
      if (threadIdx.x == 0) {
        uint32_t checksum = 0;
        for (uint32_t index = task.begin; index < task.end; ++index) {
          checksum += context->values[index];
        }
        if (state->step < fixture::kMaximumSteps) {
          context->step_checksums[state->step] = checksum;
        }
      }
      return;
    case fixture::Opcode::select_branch:
      if (threadIdx.x == 0) {
        state->branch_value =
            static_cast<uint32_t>((state->step + 1u) % task.argument0 == 0);
      }
      return;
    case fixture::Opcode::advance_predicate:
      if (threadIdx.x == 0) {
        if (state->predicate_value < fixture::kMaximumSteps) {
          context->branch_steps[state->predicate_value] = state->step;
        }
        state->predicate_value += task.argument0;
      }
      return;
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

  __device__ static __forceinline__ void initialize_profile(void *) {}

  __device__ static __forceinline__ void
  complete_profile_stage(const sg::Stage &, const sg::Task *, void *) {}
};

} // namespace

extern "C" __global__ void stateful_graph_dynamic_test_kernel(
    sg::State *state, const sg::Task *tasks, const sg::Program *programs,
    sg::MaskState *phase_states, uint32_t program_count,
    uint32_t total_task_count, uint32_t phase_state_count, void *opaque_context,
    const sg::Stage *stages, const sg::CompiledProgram *compiled_programs,
    uint32_t stage_count, uint32_t reserved,
    sg::ValidationState *validation_state) {
  (void)stages;
  (void)compiled_programs;
  (void)stage_count;
  (void)reserved;
  sg::device::run_resident<SyntheticWorkload>(
      state, tasks, programs, phase_states, program_count, total_task_count,
      phase_state_count, opaque_context, validation_state);
}

extern "C" __global__ void stateful_graph_compiled_test_kernel(
    sg::State *state, const sg::Task *tasks, const sg::Program *programs,
    sg::MaskState *phase_states, uint32_t program_count,
    uint32_t total_task_count, uint32_t phase_state_count, void *opaque_context,
    const sg::Stage *stages, const sg::CompiledProgram *compiled_programs,
    uint32_t stage_count, uint32_t reserved,
    sg::ValidationState *validation_state) {
  (void)reserved;
  sg::device::run_compiled_resident<SyntheticWorkload>(
      state, tasks, programs, phase_states, program_count, total_task_count,
      phase_state_count, opaque_context, stages, compiled_programs, stage_count,
      validation_state);
}
