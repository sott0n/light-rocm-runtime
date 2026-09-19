#include "aql_producer.hpp"
#include "launch_profile.hpp"
#if LRRT_ENABLE_LIGHT_ROCR
#include "light_rocr/runtime/aql.hpp"
#include "light_rocr_aql_queue.hpp"
#endif
#include "runtime_internal.hpp"

#include <cassert>
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

namespace {

AqlKernelDispatchParameters
aql_dispatch_parameters(const lr_launch_config_t *config,
                        uint32_t private_segment_size,
                        uint32_t group_segment_size, uint64_t kernel_object,
                        uint64_t kernarg_address, uint64_t completion_signal) {
  return {config->grid.x,
          config->grid.y,
          config->grid.z,
          static_cast<uint16_t>(config->block.x),
          static_cast<uint16_t>(config->block.y),
          static_cast<uint16_t>(config->block.z),
          private_segment_size,
          group_segment_size,
          kernel_object,
          kernarg_address,
          completion_signal};
}

lr_status_t aql_submit_status(AqlSubmitError error) {
  switch (error) {
  case AqlSubmitError::None:
    return LR_SUCCESS;
  case AqlSubmitError::InvalidPacket:
  case AqlSubmitError::InvalidQueue:
    return LR_ERROR_INVALID_ARGUMENT;
  case AqlSubmitError::QueueFull:
  case AqlSubmitError::ReserveFailed:
    return LR_ERROR_RUNTIME;
  }
  return LR_ERROR_RUNTIME;
}

} // namespace

#if LRRT_ENABLE_HSA
namespace {

static_assert(sizeof(AqlKernelDispatchPacket) ==
              sizeof(hsa_kernel_dispatch_packet_t));
static_assert(offsetof(AqlKernelDispatchPacket, completion_signal) ==
              offsetof(hsa_kernel_dispatch_packet_t, completion_signal));

uint64_t rocr_load_read_index(void *context) {
  return hsa_queue_load_read_index_scacquire(
      static_cast<hsa_queue_t *>(context));
}

uint64_t rocr_load_write_index(void *context) {
  return hsa_queue_load_write_index_relaxed(
      static_cast<hsa_queue_t *>(context));
}

bool rocr_reserve_packet(void *context, uint64_t *packet_id) {
  *packet_id = hsa_queue_add_write_index_scacq_screl(
      static_cast<hsa_queue_t *>(context), 1);
  return true;
}

void rocr_ring_doorbell(void *context, uint64_t packet_id) {
  auto *queue = static_cast<hsa_queue_t *>(context);
  hsa_signal_store_screlease(queue->doorbell_signal, packet_id);
}

AqlQueueProducerOps rocr_producer_ops(hsa_queue_t *queue) {
  return {queue,
          queue != nullptr ? queue->base_address : nullptr,
          queue != nullptr ? queue->size : 0,
          rocr_load_read_index,
          rocr_load_write_index,
          rocr_reserve_packet,
          rocr_ring_doorbell,
          nullptr};
}

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

static_assert(kAqlPacketHeaderBarrierShift ==
              light_rocr::runtime::kAqlPacketHeaderBarrierShift);
static_assert(kAqlKernelDispatchHeader ==
              light_rocr::runtime::kAqlKernelDispatchHeader);
static_assert(sizeof(AqlKernelDispatchPacket) ==
              sizeof(light_rocr::runtime::AqlKernelDispatchPacket));
static_assert(alignof(AqlKernelDispatchPacket) ==
              alignof(light_rocr::runtime::AqlKernelDispatchPacket));

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

bool valid_light_rocr_dispatch_packet(void *context,
                                      const AqlKernelDispatchPacket &packet) {
  using namespace light_rocr::runtime;
  auto *queue = static_cast<light_rocr::transport::hsakmt::AqlQueue *>(context);
  const uint64_t workgroup_size =
      static_cast<uint64_t>(packet.workgroup_size_x) *
      static_cast<uint64_t>(packet.workgroup_size_y) *
      static_cast<uint64_t>(packet.workgroup_size_z);
  if (packet.workgroup_size_x > kGfx1101WorkgroupMaximumDimension ||
      packet.workgroup_size_y > kGfx1101WorkgroupMaximumDimension ||
      packet.workgroup_size_z > kGfx1101WorkgroupMaximumDimension ||
      workgroup_size > kGfx1101WorkgroupMaximumSize ||
      packet.group_segment_size > kGfx1101GroupSegmentMaximumSize ||
      packet.private_segment_size > queue->scratch_private_segment_size() ||
      packet.kernel_object % kAmdKernelDescriptorAlignment != 0 ||
      (packet.kernarg_address != 0 &&
       packet.kernarg_address % kAmdKernargMinimumAlignment != 0) ||
      (packet.completion_signal != 0 &&
       packet.completion_signal % kAmdSignalAlignment != 0)) {
    return false;
  }
  return true;
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
    const AqlSubmitResult submitted = submit_aql_kernel_dispatch(
        rocr_producer_ops(queue.queue),
        aql_dispatch_parameters(
            config, kernel->private_segment_size,
            kernel->group_segment_size + config->shared_memory_bytes,
            kernel->object,
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kernarg.ptr)),
            signal.handle),
        {queue.pending_dispatches.size(), use_implicit_dependencies
                                              ? !queue.pending_barriers.empty()
                                              : !event_dependencies.empty()});
    if (!submitted) {
      queue.signal_pool.push_back(signal);
      queue.kernarg_pool.push_back(kernarg);
      return aql_submit_status(submitted.error);
    }
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

  const lr_status_t capacity_status =
      ensure_light_rocr_queue_capacity_locked(execution_queue, 1);
  if (capacity_status != LR_SUCCESS) {
    return capacity_status;
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

  const AqlSubmitResult submitted = submit_aql_kernel_dispatch(
      light_rocr_producer_ops(&execution_queue->queue,
                              valid_light_rocr_dispatch_packet),
      aql_dispatch_parameters(
          config, kernel_info.private_segment_size,
          kernel_info.group_segment_size + config->shared_memory_bytes,
          runtime_image.kernels()[kernel->image_kernel_index]
              .descriptor_gpu_address,
          pending->kernarg.gpu_address(),
          pending->completion_signal.gpu_handle()),
      {execution_queue->pending_dispatches.size(), false});
  if (!submitted) {
    const bool kernarg_released = static_cast<bool>(pending->kernarg.release());
    const bool signal_released =
        static_cast<bool>(pending->completion_signal.release());
    return kernarg_released && signal_released
               ? aql_submit_status(submitted.error)
               : LR_ERROR_RUNTIME;
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
