#ifndef LRRT_STATEFUL_GRAPH_RESIDENT_EXECUTOR_HPP_
#define LRRT_STATEFUL_GRAPH_RESIDENT_EXECUTOR_HPP_

#include "executor/stateful_graph/stateful_graph_abi.hpp"
#include "lrrt/lrrt.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lrrt::stateful_graph {

struct ResidentConfig {
  uint32_t phase_state_count;
  uint32_t worker_count;
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

} // namespace detail

inline void validate_resident_graph(const std::vector<Task> &tasks,
                                    const std::vector<Program> &programs,
                                    ResidentConfig config) {
  (void)detail::validate_resident_graph(tasks, programs, config);
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
    initialized_ = false;
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
        task_count_(0), program_count_(0), initialized_(false) {
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
    };
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
  std::vector<MaskState> empty_phase_states_;
  void *context_;
  uint32_t task_count_;
  uint32_t program_count_;
  bool initialized_;
};

} // namespace lrrt::stateful_graph

#endif // LRRT_STATEFUL_GRAPH_RESIDENT_EXECUTOR_HPP_
