#include "benchmarks/persistent_graph_context.hpp"
#include "lrrt/persistent_graph/compiled_scheduler.hpp"
#include "lrrt/persistent_graph/device_scheduler.hpp"

namespace sg = lrrt::persistent_graph;
namespace fixture = lrrt::benchmarks::persistent_graph;

namespace {

struct ControlWorkload {
  __device__ static __forceinline__ bool valid_context(const sg::State *,
                                                       void *opaque_context) {
    const auto *context = static_cast<fixture::Context *>(opaque_context);
    return context != nullptr && context->magic == fixture::kContextMagic &&
           context->task_count != 0 && context->task_counters != nullptr;
  }

  __device__ static __forceinline__ bool valid_task(const sg::Task &task,
                                                    void *opaque_context) {
    const auto *context = static_cast<fixture::Context *>(opaque_context);
    return task.opcode == fixture::kIncrementOpcode &&
           task.argument0 < context->task_count && task.argument1 == 1;
  }

  __device__ static __forceinline__ void
  execute(const sg::Task &task, sg::State *, void *opaque_context) {
    if (threadIdx.x == 0) {
      auto *context = static_cast<fixture::Context *>(opaque_context);
      context->task_counters[task.argument0] += task.argument1;
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

extern "C" __global__ void
persistent_graph_clock_conditioning_kernel(unsigned long long cycles,
                                           uint32_t *output) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t value = index + 1u;
  const unsigned long long start = clock64();
  do {
#pragma unroll
    for (uint32_t round = 0; round < 16; ++round) {
      value = value * 1664525u + 1013904223u;
    }
  } while (clock64() - start < cycles);
  output[index] = value;
}

extern "C" __global__ void persistent_graph_dynamic_benchmark_kernel(
    sg::State *state, const sg::Task *tasks, const sg::Program *programs,
    sg::MaskState *phase_states, uint32_t program_count,
    uint32_t total_task_count, uint32_t phase_state_count, void *opaque_context,
    const sg::Stage *stages, const sg::CompiledProgram *compiled_programs,
    uint32_t stage_count, uint32_t reserved) {
  (void)stages;
  (void)compiled_programs;
  (void)stage_count;
  (void)reserved;
  sg::device::run_resident<ControlWorkload>(
      state, tasks, programs, phase_states, program_count, total_task_count,
      phase_state_count, opaque_context);
}

extern "C" __global__ void persistent_graph_compiled_benchmark_kernel(
    sg::State *state, const sg::Task *tasks, const sg::Program *programs,
    sg::MaskState *phase_states, uint32_t program_count,
    uint32_t total_task_count, uint32_t phase_state_count, void *opaque_context,
    const sg::Stage *stages, const sg::CompiledProgram *compiled_programs,
    uint32_t stage_count, uint32_t reserved) {
  (void)reserved;
  sg::device::run_compiled_resident<ControlWorkload>(
      state, tasks, programs, phase_states, program_count, total_task_count,
      phase_state_count, opaque_context, stages, compiled_programs,
      stage_count);
}
