#ifndef LRRT_PERSISTENT_GRAPH_RESIDENT_EXECUTOR_HPP_
#define LRRT_PERSISTENT_GRAPH_RESIDENT_EXECUTOR_HPP_

#include "lrrt/lrrt.hpp"
#include "lrrt/persistent_graph/abi.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lrrt::persistent_graph {

struct ResidentConfig {
  uint32_t phase_state_count;
  uint32_t worker_count;
};

struct ResidentAdmission {
  uint32_t requested_worker_count;
  uint32_t safe_worker_limit;
  uint32_t worker_count;
};

inline uint32_t resident_worker_limit(const DeviceProperties &device,
                                      const KernelProperties &kernel,
                                      uint32_t block_threads = kWorkerThreads,
                                      uint32_t dynamic_group_segment_size = 0) {
  if (device.compute_unit_count == 0 || device.wavefront_size == 0 ||
      device.maximum_waves_per_compute_unit == 0 ||
      device.maximum_workgroup_size == 0 || block_threads == 0) {
    throw std::invalid_argument("incomplete resident resource properties");
  }
  if (block_threads > device.maximum_workgroup_size) {
    throw std::invalid_argument("resident workgroup exceeds device limit");
  }
  const uint32_t waves =
      (block_threads + device.wavefront_size - 1u) / device.wavefront_size;
  if (waves > device.maximum_waves_per_compute_unit) {
    throw std::invalid_argument("resident workgroup exceeds wave limit");
  }
  const uint64_t group_segment_size =
      static_cast<uint64_t>(kernel.group_segment_size) +
      dynamic_group_segment_size;
  if (group_segment_size > device.group_segment_size) {
    throw std::invalid_argument("resident workgroup exceeds LDS limit");
  }

  // Stateful graph kernels use grid-wide software barriers. Limiting the grid
  // to one workgroup per CU is conservative, but does not depend on fragile
  // architecture-specific SGPR/VGPR occupancy formulas.
  return std::min(device.compute_unit_count, kMaximumResidentWorkers);
}

inline ResidentAdmission admit_resident_workers(
    const DeviceProperties &device, const KernelProperties &kernel,
    uint32_t requested_worker_count, uint32_t block_threads = kWorkerThreads,
    uint32_t dynamic_group_segment_size = 0) {
  if (requested_worker_count == 0) {
    throw std::invalid_argument("resident worker request must be nonzero");
  }
  const uint32_t safe_limit = resident_worker_limit(
      device, kernel, block_threads, dynamic_group_segment_size);
  return {requested_worker_count, safe_limit,
          std::min(requested_worker_count, safe_limit)};
}

inline ResidentAdmission
admit_resident_workers(Device device, const Kernel &kernel,
                       uint32_t requested_worker_count,
                       uint32_t block_threads = kWorkerThreads,
                       uint32_t dynamic_group_segment_size = 0) {
  return admit_resident_workers(device.properties(), kernel.properties(),
                                requested_worker_count, block_threads,
                                dynamic_group_segment_size);
}

struct CompiledPlan {
  std::vector<Stage> stages;
  std::vector<CompiledProgram> programs;
};

namespace detail {

struct ValidatedGraph {
  size_t task_bytes;
  size_t program_bytes;
  size_t phase_state_bytes;
};

inline uint64_t task_dependencies(const Task &task) {
  return static_cast<uint64_t>(task.dependency_mask) |
         (static_cast<uint64_t>(task.dependency_mask_high) << 32u);
}

inline uint64_t valid_task_mask(uint32_t task_count) {
  return task_count == kMaximumTasks
             ? UINT64_MAX
             : (static_cast<uint64_t>(1) << task_count) - 1u;
}

inline ValidatedGraph
validate_resident_graph(const std::vector<Task> &tasks,
                        const std::vector<Program> &programs,
                        ResidentConfig config) {
  if (tasks.empty()) {
    throw std::invalid_argument("stateful graph must contain tasks");
  }
  if (tasks.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("invalid stateful graph task count");
  }
  if (programs.empty() || programs.size() > kMaximumPrograms) {
    throw std::invalid_argument("invalid stateful graph program count");
  }
  constexpr uint32_t kMaximumPhaseStates =
      phase_generation(std::numeric_limits<uint32_t>::max()) + 1u;
  if (config.phase_state_count == 0 ||
      config.phase_state_count > kMaximumPhaseStates) {
    throw std::invalid_argument("invalid stateful graph phase-state count");
  }
  if (config.worker_count == 0 ||
      config.worker_count > kMaximumResidentWorkers) {
    throw std::invalid_argument("invalid stateful graph resident worker count");
  }

  const uint32_t program_count = static_cast<uint32_t>(programs.size());
  for (const Program &program : programs) {
    if (program.task_count == 0 || program.task_count > kMaximumTasks ||
        program.task_offset > tasks.size() ||
        program.task_count > tasks.size() - program.task_offset) {
      throw std::invalid_argument("invalid stateful graph task range");
    }
    if (program.next_program >= program_count ||
        ((program.flags & kProgramBranchesOnState) != 0 &&
         program.branch_program >= program_count)) {
      throw std::invalid_argument("invalid stateful graph successor");
    }
    if ((program.flags & ~(kProgramAdvancesStep | kProgramEvaluatesPredicate |
                           kProgramBranchesOnState)) != 0) {
      throw std::invalid_argument("invalid stateful graph program flags");
    }

    const uint64_t valid_mask = valid_task_mask(program.task_count);
    for (uint32_t index = 0; index < program.task_count; ++index) {
      const uint64_t dependencies =
          task_dependencies(tasks[program.task_offset + index]);
      if ((dependencies & ~valid_mask) != 0 ||
          (dependencies & (static_cast<uint64_t>(1) << index)) != 0) {
        throw std::invalid_argument("invalid stateful graph dependency mask");
      }
    }
  }

  return {
      tasks.size() * sizeof(Task),
      programs.size() * sizeof(Program),
      static_cast<size_t>(config.phase_state_count) * sizeof(MaskState),
  };
}

inline void validate_compiled_plan(const std::vector<Task> &tasks,
                                   const std::vector<Program> &programs,
                                   const CompiledPlan &plan) {
  if (plan.stages.empty() || plan.programs.size() != programs.size()) {
    throw std::invalid_argument("invalid stateful graph compiled plan size");
  }
  for (size_t program_index = 0; program_index < programs.size();
       ++program_index) {
    const Program &program = programs[program_index];
    const CompiledProgram &compiled = plan.programs[program_index];
    if (compiled.stage_count == 0 ||
        compiled.stage_offset > plan.stages.size() ||
        compiled.stage_count > plan.stages.size() - compiled.stage_offset) {
      throw std::invalid_argument("invalid stateful graph compiled stages");
    }

    uint32_t expected_task_offset = program.task_offset;
    uint64_t completed_mask = 0;
    for (uint32_t stage_index = 0; stage_index < compiled.stage_count;
         ++stage_index) {
      const Stage &stage = plan.stages[compiled.stage_offset + stage_index];
      const uint32_t completed_count =
          expected_task_offset - program.task_offset;
      if (stage.task_count == 0 || stage.task_offset != expected_task_offset ||
          stage.task_count > program.task_count - completed_count) {
        throw std::invalid_argument("invalid stateful graph compiled stage");
      }
      for (uint32_t task_index = 0; task_index < stage.task_count;
           ++task_index) {
        const uint64_t dependencies =
            task_dependencies(tasks[stage.task_offset + task_index]);
        if ((dependencies & completed_mask) != dependencies) {
          throw std::invalid_argument(
              "compiled stage contains dependent tasks");
        }
      }
      completed_mask |= valid_task_mask(stage.task_count) << completed_count;
      expected_task_offset += stage.task_count;
    }
    if (expected_task_offset != program.task_offset + program.task_count) {
      throw std::invalid_argument("compiled plan does not cover program");
    }
  }
}

} // namespace detail

inline void validate_resident_graph(const std::vector<Task> &tasks,
                                    const std::vector<Program> &programs,
                                    ResidentConfig config) {
  (void)detail::validate_resident_graph(tasks, programs, config);
}

inline CompiledPlan compile_resident_plan(const std::vector<Task> &tasks,
                                          const std::vector<Program> &programs,
                                          ResidentConfig config) {
  validate_resident_graph(tasks, programs, config);
  CompiledPlan plan;
  plan.programs.reserve(programs.size());
  for (const Program &program : programs) {
    const uint32_t stage_offset = static_cast<uint32_t>(plan.stages.size());
    uint32_t next_task = 0;
    uint64_t completed_mask = 0;
    while (next_task < program.task_count) {
      const uint32_t stage_begin = next_task;
      while (next_task < program.task_count) {
        const Task &task = tasks[program.task_offset + next_task];
        const uint64_t dependencies = detail::task_dependencies(task);
        if ((dependencies & completed_mask) != dependencies) {
          break;
        }
        ++next_task;
      }
      if (next_task == stage_begin) {
        throw std::invalid_argument(
            "stateful graph tasks are not in topological stage order");
      }
      const uint32_t stage_task_count = next_task - stage_begin;
      plan.stages.push_back(
          {program.task_offset + stage_begin, stage_task_count});
      completed_mask |= detail::valid_task_mask(stage_task_count)
                        << stage_begin;
    }
    plan.programs.push_back(
        {stage_offset,
         static_cast<uint32_t>(plan.stages.size()) - stage_offset});
  }
  detail::validate_compiled_plan(tasks, programs, plan);
  return plan;
}

// Owns the model-independent control buffers for one resident graph. The
// caller supplies a workload-specialized kernel and binds its opaque device
// context; task semantics remain outside this runtime layer.
class ResidentExecutor {
public:
  ResidentExecutor(Device device, const Kernel &kernel,
                   const std::vector<Task> &tasks,
                   const std::vector<Program> &programs, ResidentConfig config)
      : ResidentExecutor(
            device, kernel, tasks, programs, config,
            detail::validate_resident_graph(tasks, programs, config)) {}

  ResidentExecutor(Device device, const Kernel &kernel,
                   const std::vector<Task> &tasks,
                   const std::vector<Program> &programs, ResidentConfig config,
                   const CompiledPlan &plan)
      : ResidentExecutor(device, kernel, tasks, programs, config) {
    upload_compiled_plan(tasks, programs, plan);
  }

  ResidentExecutor(const ResidentExecutor &) = delete;
  ResidentExecutor &operator=(const ResidentExecutor &) = delete;
  ResidentExecutor(ResidentExecutor &&) = default;
  ResidentExecutor &operator=(ResidentExecutor &&) = default;

  void upload_graph(const std::vector<Task> &tasks,
                    const std::vector<Program> &programs) {
    const detail::ValidatedGraph graph =
        detail::validate_resident_graph(tasks, programs, config_);
    if (graph.task_bytes > task_buffer_.size() ||
        graph.program_bytes > program_buffer_.size()) {
      throw std::invalid_argument(
          "stateful graph exceeds resident executor capacity");
    }
    copy_to_device(task_buffer_, tasks);
    copy_to_device(program_buffer_, programs);
    task_count_ = static_cast<uint32_t>(tasks.size());
    program_count_ = static_cast<uint32_t>(programs.size());
    stage_count_ = 0;
    initialized_ = false;
  }

  void upload_graph(const std::vector<Task> &tasks,
                    const std::vector<Program> &programs,
                    const CompiledPlan &plan) {
    upload_graph(tasks, programs);
    upload_compiled_plan(tasks, programs, plan);
  }

  void bind_context(void *device_context) { context_ = device_context; }

  void initialize(uint32_t program_index, uint32_t max_steps,
                  uint32_t stop_value) {
    if (program_index >= program_count_) {
      throw std::invalid_argument("invalid stateful graph initial program");
    }
    if (max_steps == 0) {
      throw std::invalid_argument("stateful graph max steps must be nonzero");
    }
    State state{};
    state.abi_version = kAbiVersion;
    state.status = static_cast<uint32_t>(Status::ready);
    state.phase = make_phase(0, program_index);
    state.max_steps = max_steps;
    state.stop_value = stop_value;
    state.worker_count = config_.worker_count;
    copy_to_device(state_buffer_, &state, sizeof(state));
    copy_to_device(phase_state_buffer_, empty_phase_states_);
    initialized_ = true;
  }

  void launch() const {
    require_initialized();
    lrrt::launch(kernel_, launch_config_, arguments());
  }

  void launch(const Queue &queue,
              const std::vector<const Event *> &dependencies = {}) const {
    require_initialized();
    lrrt::launch(queue, kernel_, launch_config_, arguments(), dependencies);
  }

  State read_state() const {
    require_initialized();
    State state{};
    copy_to_host(&state, state_buffer_, sizeof(state));
    return state;
  }

  State *device_state() const {
    return static_cast<State *>(state_buffer_.data());
  }
  const Task *device_tasks() const {
    return static_cast<const Task *>(task_buffer_.data());
  }
  uint32_t task_count() const { return task_count_; }
  uint32_t program_count() const { return program_count_; }
  uint32_t phase_state_count() const { return config_.phase_state_count; }
  uint32_t worker_count() const { return config_.worker_count; }
  bool has_compiled_plan() const { return stage_count_ != 0; }

private:
  ResidentExecutor(Device device, const Kernel &kernel,
                   const std::vector<Task> &tasks,
                   const std::vector<Program> &programs, ResidentConfig config,
                   detail::ValidatedGraph graph)
      : kernel_(kernel.get()), config_(config),
        launch_config_{{config.worker_count * kWorkerThreads, 1, 1},
                       {kWorkerThreads, 1, 1},
                       0},
        state_buffer_(device, sizeof(State)),
        task_buffer_(device, graph.task_bytes),
        program_buffer_(device, graph.program_bytes),
        phase_state_buffer_(device, graph.phase_state_bytes),
        empty_phase_states_(config.phase_state_count), context_(nullptr),
        task_count_(0), program_count_(0), stage_count_(0),
        initialized_(false) {
    upload_graph(tasks, programs);
  }

  ResidentArgs arguments() const {
    return {
        static_cast<State *>(state_buffer_.data()),
        static_cast<const Task *>(task_buffer_.data()),
        static_cast<const Program *>(program_buffer_.data()),
        static_cast<MaskState *>(phase_state_buffer_.data()),
        program_count_,
        task_count_,
        config_.phase_state_count,
        context_,
        stage_buffer_ ? static_cast<const Stage *>(stage_buffer_->data())
                      : nullptr,
        compiled_program_buffer_ ? static_cast<const CompiledProgram *>(
                                       compiled_program_buffer_->data())
                                 : nullptr,
        stage_count_,
        0,
    };
  }

  void upload_compiled_plan(const std::vector<Task> &tasks,
                            const std::vector<Program> &programs,
                            const CompiledPlan &plan) {
    detail::validate_compiled_plan(tasks, programs, plan);
    const size_t stage_bytes = plan.stages.size() * sizeof(Stage);
    const size_t compiled_program_bytes =
        plan.programs.size() * sizeof(CompiledProgram);
    if (!stage_buffer_) {
      stage_buffer_.emplace(state_buffer_.device(), stage_bytes);
      compiled_program_buffer_.emplace(state_buffer_.device(),
                                       compiled_program_bytes);
    } else if (stage_bytes > stage_buffer_->size() ||
               compiled_program_bytes > compiled_program_buffer_->size()) {
      throw std::invalid_argument(
          "compiled plan exceeds resident executor capacity");
    }
    copy_to_device(*stage_buffer_, plan.stages);
    copy_to_device(*compiled_program_buffer_, plan.programs);
    stage_count_ = static_cast<uint32_t>(plan.stages.size());
    initialized_ = false;
  }

  void require_initialized() const {
    if (!initialized_) {
      throw std::logic_error("stateful graph executor is not initialized");
    }
  }

  lr_kernel_t *kernel_;
  ResidentConfig config_;
  lr_launch_config_t launch_config_;
  DeviceBuffer state_buffer_;
  DeviceBuffer task_buffer_;
  DeviceBuffer program_buffer_;
  DeviceBuffer phase_state_buffer_;
  std::optional<DeviceBuffer> stage_buffer_;
  std::optional<DeviceBuffer> compiled_program_buffer_;
  std::vector<MaskState> empty_phase_states_;
  void *context_;
  uint32_t task_count_;
  uint32_t program_count_;
  uint32_t stage_count_;
  bool initialized_;
};

} // namespace lrrt::persistent_graph

#endif // LRRT_PERSISTENT_GRAPH_RESIDENT_EXECUTOR_HPP_
