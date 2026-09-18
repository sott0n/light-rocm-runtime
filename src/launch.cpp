#include "launch_profile.hpp"
#include "runtime_internal.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
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

class InitialGlobalLockHoldProfile {
public:
  InitialGlobalLockHoldProfile()
      : enabled_(g_launch_profiling_enabled), finished_(false),
        begin_(enabled_ ? ProfileClock::now() : ProfileClock::time_point{}) {}

  ~InitialGlobalLockHoldProfile() { finish(); }

  void finish() {
    if (!enabled_ || finished_) {
      return;
    }
    finished_ = true;
    g_thread_launch_profile.initial_global_lock_hold_ns +=
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                ProfileClock::now() - begin_)
                .count());
  }

private:
  bool enabled_;
  bool finished_;
  ProfileClock::time_point begin_;
};
#else
class ScopedLaunchProfile {};

class ScopedLaunchPhase {
public:
  explicit ScopedLaunchPhase(LaunchProfilePhase) {}
};

class InitialGlobalLockHoldProfile {
public:
  void finish() {}
};
#endif

class LaunchSubmissionPins {
public:
  LaunchSubmissionPins(lr_module_t *module, QueueState *queue)
      : module_(module), queue_(queue) {
    module_->active_submissions.retain();
    queue_->active_submissions.retain();
  }

  LaunchSubmissionPins(const LaunchSubmissionPins &) = delete;
  LaunchSubmissionPins &operator=(const LaunchSubmissionPins &) = delete;

  ~LaunchSubmissionPins() {
    ScopedLaunchPhase phase(LaunchProfilePhase::SubmissionPinRelease);
    // Queue release is last so runtime shutdown cannot delete modules or
    // events while this submission is still releasing their pins.
    module_->active_submissions.release();
    queue_->active_submissions.release();
  }

private:
  lr_module_t *module_;
  QueueState *queue_;
};

class FallbackEventDependencyPins {
public:
  explicit FallbackEventDependencyPins(
      const std::vector<lr_event_t *> &dependencies)
      : dependencies_(dependencies), retained_(false) {}

  void retain() {
    if (retained_) {
      return;
    }
    for (lr_event_t *event : dependencies_) {
      event->active_launch_dependencies.retain();
    }
    retained_ = true;
  }

  FallbackEventDependencyPins(const FallbackEventDependencyPins &) = delete;
  FallbackEventDependencyPins &
  operator=(const FallbackEventDependencyPins &) = delete;

  ~FallbackEventDependencyPins() {
    if (retained_) {
      ScopedLaunchPhase phase(LaunchProfilePhase::EventPinRelease);
      for (lr_event_t *event : dependencies_) {
        event->active_launch_dependencies.release();
      }
    }
  }

private:
  const std::vector<lr_event_t *> &dependencies_;
  bool retained_;
};

} // namespace
#endif

#if LRRT_ENABLE_LIGHT_ROCR
namespace {

lr_status_t
light_rocr_kernarg_status(light_rocr::runtime::KernargBufferError error) {
  using light_rocr::runtime::KernargBufferError;
  switch (error) {
  case KernargBufferError::UnsupportedAlignment:
    return LR_ERROR_NOT_SUPPORTED;
  case KernargBufferError::AllocationFailed:
    return LR_ERROR_RUNTIME;
  case KernargBufferError::None:
    return LR_SUCCESS;
  default:
    return LR_ERROR_INVALID_ARGUMENT;
  }
}

lr_status_t
light_rocr_submit_status(light_rocr::transport::hsakmt::AqlSubmitError error) {
  using light_rocr::transport::hsakmt::AqlSubmitError;
  switch (error) {
  case AqlSubmitError::InsufficientScratch:
    return LR_ERROR_NOT_SUPPORTED;
  case AqlSubmitError::None:
    return LR_SUCCESS;
  default:
    return LR_ERROR_RUNTIME;
  }
}

uint16_t light_rocr_dispatch_dimensions(const lr_launch_config_t *config) {
  if (config->grid.z > 1 || config->block.z > 1) {
    return 3;
  }
  if (config->grid.y > 1 || config->block.y > 1) {
    return 2;
  }
  return 1;
}

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
  const bool use_queue_local_submission =
      !use_default_queue && !use_implicit_dependencies && dependency_count == 0;
  const bool use_explicit_queue_dependencies =
      !use_default_queue && !use_implicit_dependencies && dependency_count != 0;
  const bool use_queue_local_locking =
      use_queue_local_submission || use_explicit_queue_dependencies;
  RuntimeLock lock(g_devices_mutex, std::defer_lock);
  RuntimeReadLock read_lock(g_devices_mutex, std::defer_lock);
  {
    ScopedLaunchPhase phase(LaunchProfilePhase::GlobalLockWait);
    // Explicit-queue submissions only inspect the registries here. Queue and
    // module pins keep those resources alive after this lock ends. Event
    // dependencies stay protected by the queue lock on the common path.
    if (use_queue_local_locking) {
      read_lock.lock();
    } else {
      lock.lock();
    }
  }
  InitialGlobalLockHoldProfile global_lock_hold_profile;
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
  std::vector<lr_event_t *> event_dependencies;
  if (use_explicit_queue_dependencies) {
    ScopedLaunchPhase phase(LaunchProfilePhase::DependencyCollection);
    lr_status_t dependency_status = collect_event_dependencies_locked(
        device, explicit_dependencies, dependency_count, nullptr,
        &event_dependencies);
    if (dependency_status != LR_SUCCESS) {
      return dependency_status;
    }
  }
  LaunchSubmissionPins submission_pins(kernel->module, &queue);
  FallbackEventDependencyPins dependency_pins(event_dependencies);
  std::unique_lock<std::mutex> queue_lock(queue.mutex, std::defer_lock);
  if (use_explicit_queue_dependencies) {
    // Hand Event lifetime protection directly from the registry read lock to
    // an uncontended queue lock. If another producer owns the queue, retain
    // the Event pins before dropping the registry lock so shared-queue
    // contention does not extend the global read-side critical section.
    {
      ScopedLaunchPhase phase(LaunchProfilePhase::QueueLockWait);
      if (!queue_lock.try_lock()) {
        dependency_pins.retain();
        global_lock_hold_profile.finish();
        read_lock.unlock();
        queue_lock.lock();
      }
    }
    if (read_lock.owns_lock()) {
      global_lock_hold_profile.finish();
      read_lock.unlock();
    }
  } else if (use_queue_local_locking) {
    global_lock_hold_profile.finish();
    read_lock.unlock();
  }

  if (!queue_lock.owns_lock()) {
    ScopedLaunchPhase phase(LaunchProfilePhase::QueueLockWait);
    queue_lock.lock();
  }
  if (use_queue_local_locking) {
    const size_t required_packets =
        use_queue_local_submission ? 1
                                   : event_dependency_packet_count_locked(
                                         &state, &queue, &event_dependencies) +
                                         1;
    {
      ScopedLaunchPhase phase(LaunchProfilePhase::QueueCapacity);
      reap_completed_dispatches_locally_locked(&queue);
      if (!has_queue_capacity_locked(&queue, required_packets)) {
        // Backpressure requires dropping the queue before reacquiring the
        // registry exclusively. Retain Event pins only on this slow path.
        if (use_explicit_queue_dependencies) {
          dependency_pins.retain();
        }
        queue_lock.unlock();
        lock.lock();
        queue_lock.lock();
        if (!valid_kernel_locked(kernel) ||
            !valid_queue_locked(execution_queue)) {
          return LR_ERROR_INVALID_ARGUMENT;
        }
        lr_status_t capacity_status = ensure_queue_capacity_locked(
            &lock, &queue_lock, &queue, required_packets);
        if (capacity_status != LR_SUCCESS) {
          return capacity_status;
        }
        lock.unlock();
      }
    }
  } else {
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
          event_dependency_packet_count_locked(&state, &queue,
                                               capacity_dependencies) +
          1;
      bool lock_released = false;
      lr_status_t capacity_status;
      {
        ScopedLaunchPhase phase(LaunchProfilePhase::QueueCapacity);
        capacity_status = ensure_queue_capacity_locked(
            &lock, &queue_lock, &queue, required_packets, &lock_released);
      }
      if (capacity_status != LR_SUCCESS) {
        return capacity_status;
      }
      if (!lock_released) {
        break;
      }
      if (!valid_kernel_locked(kernel) ||
          !valid_queue_locked(execution_queue)) {
        return LR_ERROR_INVALID_ARGUMENT;
      }
    }
  }
  const hsa_region_t kernarg_region = state.kernarg_region;
  const std::vector<lr_event_t *> *dependencies =
      use_implicit_dependencies ? nullptr : &event_dependencies;
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
#elif LRRT_ENABLE_LIGHT_ROCR
  std::lock_guard<RuntimeMutex> lock(g_devices_mutex);
  if (!valid_kernel_locked(kernel)) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  lr_device_t device = kernel->module->device;
  if (device.index >= g_devices.size() || !g_kfd_session) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  DeviceState &state = g_devices[device.index];
  if (use_default_queue) {
    execution_queue = state.default_queue;
  }
  if (!execution_queue || !valid_light_rocr_queue_locked(execution_queue) ||
      execution_queue->device.index != device.index) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (dependency_count != 0) {
    if (!explicit_dependencies) {
      return LR_ERROR_INVALID_ARGUMENT;
    }
    return LR_ERROR_NOT_SUPPORTED;
  }
  (void)explicit_dependencies;
  (void)use_implicit_dependencies;

  const auto &kernels = kernel->module->executable_image.code_object().kernels;
  if (kernel->image_kernel_index >= kernels.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  const light_rocr::loader::KernelInfo &kernel_info =
      kernels[kernel->image_kernel_index];
  const lr_status_t scratch_status = ensure_light_rocr_queue_scratch_locked(
      &state, execution_queue, kernel_info.private_segment_size);
  if (scratch_status != LR_SUCCESS) {
    return scratch_status;
  }

  // Barrier-bit ordering is not enabled on this path yet. Retire an earlier
  // dispatch before publishing the next one so LRRT's same-queue completion
  // ordering remains correct.
  if (!execution_queue->pending_dispatches.empty()) {
    const lr_status_t synchronization_status =
        synchronize_light_rocr_queue_locked(execution_queue);
    if (synchronization_status != LR_SUCCESS) {
      return synchronization_status;
    }
  }

  auto kernarg = light_rocr::transport::hsakmt::create_kernarg_buffer(
      *g_kfd_session, state.node.node_id, kernel_info, args, args_size);
  if (!kernarg) {
    const lr_status_t status = light_rocr_kernarg_status(kernarg.status.error);
    if (kernarg.buffer.owns_allocation() && !kernarg.buffer.release()) {
      return LR_ERROR_RUNTIME;
    }
    return status;
  }

  auto signal = g_kfd_session->create_user_signal(state.node.node_id, 1);
  if (!signal) {
    if (!kernarg.buffer.release()) {
      return LR_ERROR_RUNTIME;
    }
    return LR_ERROR_RUNTIME;
  }

  const auto &runtime_image = kernel->module->executable_image.runtime_image();
  const auto &kernarg_info = kernarg.buffer.runtime_buffer();
  if (!runtime_image ||
      runtime_image.kernels().size() !=
          runtime_image.code_object().kernels.size() ||
      kernel->image_kernel_index >= runtime_image.kernels().size() ||
      !kernarg_info ||
      kernarg_info.kernarg_size() != kernel_info.kernarg_size ||
      kernarg_info.alignment() < kernel_info.kernarg_alignment ||
      (kernel_info.kernarg_size == 0 && kernarg_info.gpu_address() != 0) ||
      (kernel_info.kernarg_size != 0 &&
       (kernarg_info.gpu_address() == 0 ||
        kernarg_info.gpu_address() % kernel_info.kernarg_alignment != 0))) {
    const bool kernarg_released = static_cast<bool>(kernarg.buffer.release());
    const bool signal_released = static_cast<bool>(signal.signal.release());
    return kernarg_released && signal_released ? LR_ERROR_INVALID_ARGUMENT
                                               : LR_ERROR_RUNTIME;
  }
  if (kernel_info.uses_dynamic_stack) {
    const bool kernarg_released = static_cast<bool>(kernarg.buffer.release());
    const bool signal_released = static_cast<bool>(signal.signal.release());
    return kernarg_released && signal_released ? LR_ERROR_NOT_SUPPORTED
                                               : LR_ERROR_RUNTIME;
  }
  if (config->shared_memory_bytes >
      std::numeric_limits<uint32_t>::max() - kernel_info.group_segment_size) {
    const bool kernarg_released = static_cast<bool>(kernarg.buffer.release());
    const bool signal_released = static_cast<bool>(signal.signal.release());
    return kernarg_released && signal_released ? LR_ERROR_INVALID_ARGUMENT
                                               : LR_ERROR_RUNTIME;
  }

  light_rocr::runtime::AqlKernelDispatchPacket packet;
  packet.header = light_rocr::runtime::kAqlKernelDispatchHeader;
  packet.setup = static_cast<uint16_t>(
      light_rocr_dispatch_dimensions(config)
      << light_rocr::runtime::kAqlKernelDispatchDimensionsShift);
  packet.workgroup_size_x = static_cast<uint16_t>(config->block.x);
  packet.workgroup_size_y = static_cast<uint16_t>(config->block.y);
  packet.workgroup_size_z = static_cast<uint16_t>(config->block.z);
  packet.grid_size_x = config->grid.x;
  packet.grid_size_y = config->grid.y;
  packet.grid_size_z = config->grid.z;
  packet.private_segment_size = kernel_info.private_segment_size;
  packet.group_segment_size =
      kernel_info.group_segment_size + config->shared_memory_bytes;
  packet.kernel_object = runtime_image.kernels()[kernel->image_kernel_index]
                             .descriptor_gpu_address;
  packet.kernarg_address = kernarg_info.gpu_address();
  packet.completion_signal = signal.signal.gpu_handle();
  const auto packet_status =
      light_rocr::runtime::validate_kernel_dispatch_packet(packet);
  if (!packet_status) {
    const bool kernarg_released = static_cast<bool>(kernarg.buffer.release());
    const bool signal_released = static_cast<bool>(signal.signal.release());
    return kernarg_released && signal_released ? LR_ERROR_INVALID_ARGUMENT
                                               : LR_ERROR_RUNTIME;
  }

  std::unique_ptr<lr_queue_t::PendingDispatch> pending;
  try {
    pending = std::make_unique<lr_queue_t::PendingDispatch>(
        std::move(signal.signal), std::move(kernarg.buffer));
    execution_queue->pending_dispatches.reserve(
        execution_queue->pending_dispatches.size() + 1);
  } catch (const std::bad_alloc &) {
    if (pending) {
      (void)pending->kernarg.release();
      (void)pending->completion_signal.release();
    } else {
      (void)kernarg.buffer.release();
      (void)signal.signal.release();
    }
    return LR_ERROR_RUNTIME;
  }

  const auto submitted = execution_queue->queue.submit_kernel_dispatch(packet);
  if (!submitted) {
    const lr_status_t status = light_rocr_submit_status(submitted.error);
    const bool kernarg_released = static_cast<bool>(pending->kernarg.release());
    const bool signal_released =
        static_cast<bool>(pending->completion_signal.release());
    return kernarg_released && signal_released ? status : LR_ERROR_RUNTIME;
  }

  execution_queue->pending_dispatches.push_back(std::move(pending));
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
