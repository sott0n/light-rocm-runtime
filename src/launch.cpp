#include "launch_profile.hpp"
#include "runtime_internal.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <vector>

using namespace lrrt_internal;

#ifndef LRRT_ENABLE_LAUNCH_PROFILING
#define LRRT_ENABLE_LAUNCH_PROFILING 0
#endif

#if LRRT_ENABLE_HSA
namespace {

#if LRRT_ENABLE_LAUNCH_PROFILING
using ProfileClock = std::chrono::steady_clock;

thread_local bool g_launch_profiling_enabled = false;
thread_local LaunchProfile g_thread_launch_profile;

class ScopedLaunchProfile {
public:
  ScopedLaunchProfile()
      : enabled_(g_launch_profiling_enabled), begin_(now_if_enabled()) {}

  ~ScopedLaunchProfile() {
    if (enabled_) {
      ++g_thread_launch_profile.launch_count;
      g_thread_launch_profile.total_ns += elapsed_ns(begin_);
    }
  }

private:
  ProfileClock::time_point now_if_enabled() const {
    return enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
  }

  static uint64_t elapsed_ns(ProfileClock::time_point begin) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            ProfileClock::now() - begin)
            .count());
  }

  bool enabled_;
  ProfileClock::time_point begin_;
};

class ScopedLaunchPhase {
public:
  explicit ScopedLaunchPhase(LaunchProfilePhase phase)
      : enabled_(g_launch_profiling_enabled), phase_(phase),
        begin_(enabled_ ? ProfileClock::now() : ProfileClock::time_point{}) {}

  ~ScopedLaunchPhase() {
    if (enabled_) {
      const auto duration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              ProfileClock::now() - begin_);
      g_thread_launch_profile.phase_ns[static_cast<size_t>(phase_)] +=
          static_cast<uint64_t>(duration.count());
    }
  }

private:
  bool enabled_;
  LaunchProfilePhase phase_;
  ProfileClock::time_point begin_;
};
#else
class ScopedLaunchProfile {};

class ScopedLaunchPhase {
public:
  explicit ScopedLaunchPhase(LaunchProfilePhase) {}
};
#endif

class LaunchSubmissionPins {
public:
  LaunchSubmissionPins(lr_module_t *module, QueueState *queue)
      : module_(module), queue_(queue) {
    ++module_->active_submissions;
    ++queue_->active_submissions;
  }

  LaunchSubmissionPins(const LaunchSubmissionPins &) = delete;
  LaunchSubmissionPins &operator=(const LaunchSubmissionPins &) = delete;

  ~LaunchSubmissionPins() {
    --queue_->active_submissions;
    --module_->active_submissions;
    g_launch_state_changed.notify_all();
  }

private:
  lr_module_t *module_;
  QueueState *queue_;
};

class EventDependencyPins {
public:
  EventDependencyPins(const std::vector<lr_event_t *> &dependencies,
                      bool enabled)
      : dependencies_(enabled ? &dependencies : nullptr) {
    if (dependencies_) {
      for (lr_event_t *event : *dependencies_) {
        ++event->active_synchronizers;
      }
    }
  }

  EventDependencyPins(const EventDependencyPins &) = delete;
  EventDependencyPins &operator=(const EventDependencyPins &) = delete;

  ~EventDependencyPins() {
    if (dependencies_) {
      for (lr_event_t *event : *dependencies_) {
        --event->active_synchronizers;
      }
      g_event_state_changed.notify_all();
    }
  }

private:
  const std::vector<lr_event_t *> *dependencies_;
};

class QueueLocalSubmissionScope {
public:
  QueueLocalSubmissionScope(std::unique_lock<std::mutex> *devices_lock,
                            std::unique_lock<std::mutex> *queue_lock,
                            bool enabled)
      : devices_lock_(devices_lock), queue_lock_(queue_lock),
        enabled_(enabled) {
    if (enabled_) {
      devices_lock_->unlock();
    }
  }

  QueueLocalSubmissionScope(const QueueLocalSubmissionScope &) = delete;
  QueueLocalSubmissionScope &
  operator=(const QueueLocalSubmissionScope &) = delete;

  ~QueueLocalSubmissionScope() {
    if (enabled_) {
      ScopedLaunchPhase phase(LaunchProfilePhase::LockRestoration);
      // Restore the global-before-queue lock order before lifetime pins are
      // released by their enclosing scope.
      queue_lock_->unlock();
      devices_lock_->lock();
    }
  }

private:
  std::unique_lock<std::mutex> *devices_lock_;
  std::unique_lock<std::mutex> *queue_lock_;
  bool enabled_;
};

} // namespace
#endif

namespace lrrt_internal {

void set_thread_launch_profiling(bool enabled) {
#if LRRT_ENABLE_HSA && LRRT_ENABLE_LAUNCH_PROFILING
  g_launch_profiling_enabled = enabled;
#else
  (void)enabled;
#endif
}

void reset_thread_launch_profile() {
#if LRRT_ENABLE_HSA && LRRT_ENABLE_LAUNCH_PROFILING
  g_thread_launch_profile = LaunchProfile{};
#endif
}

LaunchProfile thread_launch_profile() {
#if LRRT_ENABLE_HSA && LRRT_ENABLE_LAUNCH_PROFILING
  return g_thread_launch_profile;
#else
  return LaunchProfile{};
#endif
}

} // namespace lrrt_internal

extern "C" {

static lr_status_t
launch_impl(lr_kernel_t *kernel, const lr_launch_config_t *config,
            const void *args, size_t args_size, lr_queue_t *execution_queue,
            bool use_default_queue, lr_event_t *const *explicit_dependencies,
            size_t dependency_count, bool use_implicit_dependencies) {
#if LRRT_ENABLE_HSA
  ScopedLaunchProfile launch_profile;
#endif
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!kernel || !config || !args || args_size == 0) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (config->grid.x == 0 || config->grid.y == 0 || config->grid.z == 0 ||
      config->block.x == 0 || config->block.y == 0 || config->block.z == 0) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (config->grid.x < config->block.x || config->grid.y < config->block.y ||
      config->grid.z < config->block.z) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (config->block.x > UINT16_MAX || config->block.y > UINT16_MAX ||
      config->block.z > UINT16_MAX) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

#if LRRT_ENABLE_HSA
  std::unique_lock<std::mutex> lock(g_devices_mutex, std::defer_lock);
  {
    ScopedLaunchPhase phase(LaunchProfilePhase::GlobalLockWait);
    lock.lock();
  }
  if (!valid_kernel_locked(kernel)) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  lr_device_t device = kernel->module->device;
  if (device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  DeviceState &state = g_devices[device.index];
  if (use_default_queue) {
    execution_queue = state.default_queue;
  }
  if (!execution_queue || !valid_queue_locked(execution_queue) ||
      execution_queue->device.index != device.index ||
      !state.has_kernarg_region || args_size > kernel->kernarg_size) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  QueueState &queue = execution_queue->state;
  std::unique_lock<std::mutex> queue_lock(queue.mutex, std::defer_lock);
  {
    ScopedLaunchPhase phase(LaunchProfilePhase::QueueLockWait);
    queue_lock.lock();
  }
  LaunchSubmissionPins submission_pins(kernel->module, &queue);
  const bool use_queue_local_submission =
      !use_default_queue && !use_implicit_dependencies && dependency_count == 0;
  const bool use_explicit_queue_dependencies =
      !use_default_queue && !use_implicit_dependencies && dependency_count != 0;
  std::vector<lr_event_t *> event_dependencies;
  while (true) {
    event_dependencies.clear();
    if (!use_implicit_dependencies) {
      lr_status_t dependency_status;
      {
        ScopedLaunchPhase phase(LaunchProfilePhase::DependencyCollection);
        dependency_status = collect_event_dependencies_locked(
            device, explicit_dependencies, dependency_count, nullptr,
            &event_dependencies);
      }
      if (dependency_status != LR_SUCCESS) {
        return dependency_status;
      }
    }
    const std::vector<lr_event_t *> *capacity_dependencies =
        use_implicit_dependencies ? nullptr : &event_dependencies;
    const size_t required_packets =
        use_queue_local_submission
            ? 1
            : event_dependency_packet_count_locked(&state, &queue,
                                                   capacity_dependencies) +
                  1;
    bool lock_released = false;
    lr_status_t capacity_status;
    {
      ScopedLaunchPhase phase(LaunchProfilePhase::QueueCapacity);
      if (use_queue_local_submission) {
        reap_completed_dispatches_locally_locked(&queue);
        if (has_queue_capacity_locked(&queue, required_packets)) {
          break;
        }
      }
      capacity_status = ensure_queue_capacity_locked(
          &lock, &queue_lock, &queue, required_packets, &lock_released);
    }
    if (capacity_status != LR_SUCCESS) {
      return capacity_status;
    }
    if (!lock_released) {
      break;
    }
    if (!valid_kernel_locked(kernel) || !valid_queue_locked(execution_queue)) {
      return LR_ERROR_INVALID_ARGUMENT;
    }
  }
  const hsa_region_t kernarg_region = state.kernarg_region;
  const std::vector<lr_event_t *> *dependencies =
      use_implicit_dependencies ? nullptr : &event_dependencies;
  EventDependencyPins dependency_pins(event_dependencies,
                                      use_explicit_queue_dependencies);
  KernargBuffer kernarg{};
  hsa_signal_t signal{};
  auto acquire_submission_resources = [&]() -> lr_status_t {
    hsa_status_t status = acquire_kernarg_locked(
        &queue, kernarg_region, kernel->kernarg_size, &kernarg);
    if (status != HSA_STATUS_SUCCESS) {
      return to_lr_status(status);
    }
    status = acquire_signal_locked(&queue, &signal);
    if (status != HSA_STATUS_SUCCESS) {
      queue.kernarg_pool.push_back(kernarg);
      kernarg = KernargBuffer{};
      return to_lr_status(status);
    }
    return LR_SUCCESS;
  };

  std::optional<QueueLocalSubmissionScope> queue_local_scope;
  if (use_queue_local_submission || use_explicit_queue_dependencies) {
    queue_local_scope.emplace(&lock, &queue_lock, true);
  }
  {
    ScopedLaunchPhase phase(LaunchProfilePhase::ResourceAcquisition);
    lr_status_t resource_status = acquire_submission_resources();
    if (resource_status != LR_SUCCESS) {
      return resource_status;
    }
  }
  if (!use_queue_local_submission) {
    lr_status_t dependency_status;
    {
      ScopedLaunchPhase phase(LaunchProfilePhase::DependencyRegistration);
      dependency_status =
          use_explicit_queue_dependencies
              ? enqueue_explicit_event_dependencies_locally_locked(
                    &queue, signal, event_dependencies)
              : enqueue_event_dependencies_locked(&lock, &state, &queue_lock,
                                                  &queue, signal, dependencies);
    }
    if (dependency_status != LR_SUCCESS) {
      queue.signal_pool.push_back(signal);
      queue.kernarg_pool.push_back(kernarg);
      return dependency_status;
    }
  }
  {
    ScopedLaunchPhase phase(LaunchProfilePhase::PacketPublication);
    std::memset(kernarg.ptr, 0, kernel->kernarg_size);
    std::memcpy(kernarg.ptr, args, args_size);
    // Keep packets on the same lrrt queue completion-ordered. Several executor
    // pipelines pass one kernel's output directly to the next kernel.
    const bool wait_for_dependencies =
        !queue.pending_dispatches.empty() ||
        (use_implicit_dependencies ? !queue.pending_barriers.empty()
                                   : !event_dependencies.empty());

    const uint64_t index =
        hsa_queue_add_write_index_scacq_screl(queue.queue, 1);
    auto *packets =
        static_cast<hsa_kernel_dispatch_packet_t *>(queue.queue->base_address);
    hsa_kernel_dispatch_packet_t *packet =
        &packets[index & (queue.queue->size - 1)];
    std::memset(packet, 0, sizeof(*packet));
    packet->setup = packet_setup(dispatch_dimensions(config));
    packet->workgroup_size_x = static_cast<uint16_t>(config->block.x);
    packet->workgroup_size_y = static_cast<uint16_t>(config->block.y);
    packet->workgroup_size_z = static_cast<uint16_t>(config->block.z);
    packet->grid_size_x = config->grid.x;
    packet->grid_size_y = config->grid.y;
    packet->grid_size_z = config->grid.z;
    packet->private_segment_size = kernel->private_segment_size;
    packet->group_segment_size =
        kernel->group_segment_size + config->shared_memory_bytes;
    packet->kernel_object = kernel->object;
    packet->kernarg_address = kernarg.ptr;
    packet->completion_signal = signal;
    uint16_t header =
        wait_for_dependencies
            ? barrier_packet_header(HSA_PACKET_TYPE_KERNEL_DISPATCH)
            : packet_header(HSA_PACKET_TYPE_KERNEL_DISPATCH);
    publish_packet_header(&packet->header, header);

    hsa_signal_store_screlease(queue.queue->doorbell_signal, index);
    queue.pending_dispatches.push_back(PendingDispatch{signal, kernarg});
  }
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_launch(lr_kernel_t *kernel, const lr_launch_config_t *config,
                      const void *args, size_t args_size) {
  return launch_impl(kernel, config, args, args_size, nullptr, true, nullptr, 0,
                     true);
}

lr_status_t lr_launch_with_dependencies(lr_kernel_t *kernel,
                                        const lr_launch_config_t *config,
                                        const void *args, size_t args_size,
                                        lr_event_t *const *dependencies,
                                        size_t dependency_count) {
  return launch_impl(kernel, config, args, args_size, nullptr, true,
                     dependencies, dependency_count, false);
}

lr_status_t lr_launch_on_queue(lr_queue_t *queue, lr_kernel_t *kernel,
                               const lr_launch_config_t *config,
                               const void *args, size_t args_size) {
  return launch_impl(kernel, config, args, args_size, queue, false, nullptr, 0,
                     false);
}

lr_status_t lr_launch_on_queue_with_dependencies(
    lr_queue_t *queue, lr_kernel_t *kernel, const lr_launch_config_t *config,
    const void *args, size_t args_size, lr_event_t *const *dependencies,
    size_t dependency_count) {
  return launch_impl(kernel, config, args, args_size, queue, false,
                     dependencies, dependency_count, false);
}

} // extern "C"
