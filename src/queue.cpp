#include "runtime_internal.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lrrt_internal {

#if LRRT_ENABLE_HSA
std::unordered_set<lr_queue_t *> g_queues;

struct SynchronizationDependency {
  hsa_signal_t signal;
  lr_event_t *event;
};

struct QueueDestruction {
  hsa_signal_t completion_signal;
  hsa_signal_value_t wait_value;
  size_t destruction_pin_count;
};

bool valid_queue_locked(lr_queue_t *queue) {
  return g_queues.find(queue) != g_queues.end() && !queue->state.destroying;
}

hsa_status_t create_queue(lr_device_t device_handle, DeviceState *device,
                          bool is_default, lr_queue_t **queue) {
  uint32_t max_queue_size = 0;
  hsa_status_t status = hsa_agent_get_info(
      device->agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &max_queue_size);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (max_queue_size == 0) {
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  uint32_t queue_size = 1024;
  while (queue_size > max_queue_size) {
    queue_size >>= 1;
  }
  if (queue_size == 0) {
    queue_size = 1;
  }

  auto *created_queue = new lr_queue_t{};
  created_queue->device = device_handle;
  created_queue->is_default = is_default;
  created_queue->state.active_synchronizers = 0;
  created_queue->state.destroying = false;
  status = hsa_queue_create(device->agent, queue_size, HSA_QUEUE_TYPE_MULTI,
                            nullptr, nullptr, UINT32_MAX, UINT32_MAX,
                            &created_queue->state.queue);
  if (status != HSA_STATUS_SUCCESS) {
    delete created_queue;
    return status;
  }

  status =
      hsa_amd_profiling_set_profiler_enabled(created_queue->state.queue, 1);
  if (status != HSA_STATUS_SUCCESS) {
    hsa_queue_destroy(created_queue->state.queue);
    delete created_queue;
    return status;
  }
  device->queues.push_back(created_queue);
  g_queues.insert(created_queue);
  *queue = created_queue;
  return HSA_STATUS_SUCCESS;
}

uint16_t packet_header(hsa_packet_type_t type) {
  return static_cast<uint16_t>(
      (type << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
}

uint16_t barrier_packet_header(hsa_packet_type_t type) {
  return static_cast<uint16_t>(packet_header(type) |
                               (1 << HSA_PACKET_HEADER_BARRIER));
}

void publish_packet_header(uint16_t *header, uint16_t value) {
  __atomic_store_n(header, value, __ATOMIC_RELEASE);
}

uint16_t packet_setup(uint16_t dimensions) {
  return static_cast<uint16_t>(dimensions
                               << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);
}

uint16_t dispatch_dimensions(const lr_launch_config_t *config) {
  if (config->grid.z > 1 || config->block.z > 1) {
    return 3;
  }
  if (config->grid.y > 1 || config->block.y > 1) {
    return 2;
  }
  return 1;
}

lr_status_t drain_queue_work_locked(QueueState *queue) {
  lr_status_t result = LR_SUCCESS;
  for (const PendingBarrier &barrier : queue->pending_barriers) {
    hsa_signal_value_t value = hsa_signal_wait_scacquire(
        barrier.retirement_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
        HSA_WAIT_STATE_BLOCKED);
    if (value != 0 && result == LR_SUCCESS) {
      result = LR_ERROR_RUNTIME;
    }
    for (lr_event_t *event : barrier.dependencies) {
      release_event_dependency(event, queue);
    }
  }
  queue->pending_barriers.clear();

  for (const PendingDispatch &dispatch : queue->pending_dispatches) {
    hsa_signal_value_t value = hsa_signal_wait_scacquire(
        dispatch.completion_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
        HSA_WAIT_STATE_BLOCKED);
    if (value != 0 && result == LR_SUCCESS) {
      result = LR_ERROR_RUNTIME;
    }
    const bool progress_waiter = std::any_of(
        queue->progress_wait_signals.begin(),
        queue->progress_wait_signals.end(), [&dispatch](hsa_signal_t signal) {
          return signal.handle == dispatch.completion_signal.handle;
        });
    if (progress_waiter) {
      queue->deferred_dispatches.push_back(dispatch);
    } else {
      hsa_signal_store_relaxed(dispatch.completion_signal, 1);
      queue->signal_pool.push_back(dispatch.completion_signal);
      queue->kernarg_pool.push_back(dispatch.kernarg);
    }
  }
  queue->pending_dispatches.clear();
  return result;
}

lr_status_t drain_queue_locked(QueueState *queue) {
  lr_status_t result = LR_SUCCESS;
  std::vector<lr_event_t *> pending_events = queue->pending_events;
  for (lr_event_t *event : pending_events) {
    lr_status_t status = event_wait_locked(event, queue);
    if (status != LR_SUCCESS && result == LR_SUCCESS) {
      result = status;
    }
  }
  lr_status_t status = drain_queue_work_locked(queue);
  return result == LR_SUCCESS ? status : result;
}

lr_status_t drain_device_locked(DeviceState *device) {
  lr_status_t result = LR_SUCCESS;
  std::vector<lr_event_t *> pending_events = device->pending_events;
  for (lr_event_t *event : pending_events) {
    lr_status_t status = event_wait_locked(event);
    if (status != LR_SUCCESS && result == LR_SUCCESS) {
      result = status;
    }
  }
  for (lr_queue_t *queue : device->queues) {
    std::lock_guard<std::mutex> queue_lock(queue->state.mutex);
    lr_status_t status = drain_queue_work_locked(&queue->state);
    if (status != LR_SUCCESS && result == LR_SUCCESS) {
      result = status;
    }
  }
  return result;
}

void release_queue_pools_locked(QueueState *queue, lr_status_t *result) {
  for (hsa_signal_t signal : queue->signal_pool) {
    hsa_status_t status = hsa_signal_destroy(signal);
    if (status != HSA_STATUS_SUCCESS && *result == LR_SUCCESS) {
      *result = to_lr_status(status);
    }
  }
  queue->signal_pool.clear();

  for (KernargBuffer kernarg : queue->kernarg_pool) {
    hsa_status_t status = hsa_memory_free(kernarg.ptr);
    if (status != HSA_STATUS_SUCCESS && *result == LR_SUCCESS) {
      *result = to_lr_status(status);
    }
  }
  queue->kernarg_pool.clear();
}

hsa_status_t acquire_signal_locked(QueueState *queue, hsa_signal_t *signal) {
  if (!queue->signal_pool.empty()) {
    *signal = queue->signal_pool.back();
    queue->signal_pool.pop_back();
    hsa_signal_store_relaxed(*signal, 1);
    return HSA_STATUS_SUCCESS;
  }
  return hsa_signal_create(1, 0, nullptr, signal);
}

hsa_status_t acquire_kernarg_locked(QueueState *queue, hsa_region_t region,
                                    size_t size, KernargBuffer *kernarg) {
  for (size_t i = 0; i < queue->kernarg_pool.size(); ++i) {
    if (queue->kernarg_pool[i].size >= size) {
      *kernarg = queue->kernarg_pool[i];
      queue->kernarg_pool[i] = queue->kernarg_pool.back();
      queue->kernarg_pool.pop_back();
      return HSA_STATUS_SUCCESS;
    }
  }

  kernarg->ptr = nullptr;
  kernarg->size = size;
  return hsa_memory_allocate(region, size, &kernarg->ptr);
}

void reap_completed_barriers_locked(QueueState *queue) {
  size_t index = 0;
  while (index < queue->pending_barriers.size()) {
    PendingBarrier &barrier = queue->pending_barriers[index];
    if (hsa_signal_load_scacquire(barrier.retirement_signal) != 0) {
      ++index;
      continue;
    }

    for (lr_event_t *event : barrier.dependencies) {
      release_event_dependency(event, queue);
    }
    if (index + 1 != queue->pending_barriers.size()) {
      barrier = std::move(queue->pending_barriers.back());
    }
    queue->pending_barriers.pop_back();
  }
}

void reap_completed_dispatches_locked(QueueState *queue) {
  // Dispatch packets are submitted completion-ordered on each lrrt queue, so
  // retaining the unfinished suffix also keeps the oldest waiter at the front.
  size_t completed_count = 0;
  while (completed_count < queue->pending_dispatches.size()) {
    PendingDispatch &dispatch = queue->pending_dispatches[completed_count];
    if (hsa_signal_load_scacquire(dispatch.completion_signal) != 0) {
      break;
    }
    const bool progress_waiter = std::any_of(
        queue->progress_wait_signals.begin(),
        queue->progress_wait_signals.end(), [&dispatch](hsa_signal_t signal) {
          return signal.handle == dispatch.completion_signal.handle;
        });
    const bool synchronization_consumer = std::any_of(
        queue->pending_synchronizations.begin(),
        queue->pending_synchronizations.end(),
        [&dispatch](const PendingSynchronization &synchronization) {
          return hsa_signal_load_scacquire(synchronization.completion_signal) !=
                     0 &&
                 std::any_of(synchronization.dispatch_dependencies.begin(),
                             synchronization.dispatch_dependencies.end(),
                             [&dispatch](hsa_signal_t dependency) {
                               return dependency.handle ==
                                      dispatch.completion_signal.handle;
                             });
        });
    if (synchronization_consumer) {
      break;
    }

    if (progress_waiter) {
      queue->deferred_dispatches.push_back(dispatch);
    } else {
      hsa_signal_store_relaxed(dispatch.completion_signal, 1);
      queue->signal_pool.push_back(dispatch.completion_signal);
      queue->kernarg_pool.push_back(dispatch.kernarg);
    }
    ++completed_count;
  }
  if (completed_count != 0) {
    queue->pending_dispatches.erase(queue->pending_dispatches.begin(),
                                    queue->pending_dispatches.begin() +
                                        completed_count);
  }
}

void reap_completed_dispatches_locally_locked(QueueState *queue) {
  // Barrier and event retirement also updates globally protected event state.
  // A dispatch-only queue can recycle its completed resources independently.
  if (queue->pending_barriers.empty() && queue->pending_events.empty()) {
    reap_completed_dispatches_locked(queue);
  }
}

lr_status_t reap_completed_queue_events_locked(QueueState *queue) {
  lr_status_t result = LR_SUCCESS;
  size_t index = 0;
  while (index < queue->pending_events.size()) {
    lr_event_t *event = queue->pending_events[index];
    if (hsa_signal_load_scacquire(event->signal) != 0) {
      ++index;
      continue;
    }

    lr_status_t status = event_wait_locked(event, queue);
    if (status != LR_SUCCESS && result == LR_SUCCESS) {
      result = status;
    }
  }
  return result;
}

lr_status_t reap_completed_queue_work_locked(QueueState *queue) {
  // Barriers retain dispatch completion signals, so retire them before those
  // signals can be reset and returned to the queue pool.
  reap_completed_barriers_locked(queue);
  reap_completed_dispatches_locked(queue);
  return reap_completed_queue_events_locked(queue);
}

size_t pending_queue_packet_count(const QueueState *queue) {
  return queue->pending_dispatches.size() + queue->pending_barriers.size() +
         queue->pending_events.size();
}

bool has_queue_capacity_locked(const QueueState *queue,
                               size_t required_packets) {
  return required_packets != 0 && required_packets <= queue->queue->size &&
         pending_queue_packet_count(queue) + required_packets <=
             queue->queue->size;
}

lr_status_t
wait_for_queue_progress_locked(RuntimeLock *devices_lock,
                               std::unique_lock<std::mutex> *queue_lock,
                               QueueState *queue, bool allow_destroying) {
  hsa_signal_t signal{};
  lr_event_t *event = nullptr;
  if (!queue->pending_dispatches.empty()) {
    signal = queue->pending_dispatches.front().completion_signal;
  } else if (!queue->pending_events.empty()) {
    event = queue->pending_events.front();
    signal = event->signal;
    ++event->active_synchronizers;
  } else {
    return LR_ERROR_RUNTIME;
  }

  queue->progress_wait_signals.push_back(signal);
  ++queue->active_synchronizers;
  queue_lock->unlock();
  devices_lock->unlock();
  hsa_signal_value_t value = hsa_signal_wait_scacquire(
      signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  devices_lock->lock();
  queue_lock->lock();

  auto waiter = std::find_if(queue->progress_wait_signals.begin(),
                             queue->progress_wait_signals.end(),
                             [signal](hsa_signal_t pending) {
                               return pending.handle == signal.handle;
                             });
  if (waiter != queue->progress_wait_signals.end()) {
    queue->progress_wait_signals.erase(waiter);
  }
  const bool signal_still_waited = std::any_of(
      queue->progress_wait_signals.begin(), queue->progress_wait_signals.end(),
      [signal](hsa_signal_t pending) {
        return pending.handle == signal.handle;
      });
  if (!signal_still_waited) {
    auto deferred = std::find_if(
        queue->deferred_dispatches.begin(), queue->deferred_dispatches.end(),
        [signal](const PendingDispatch &item) {
          return item.completion_signal.handle == signal.handle;
        });
    if (deferred != queue->deferred_dispatches.end()) {
      hsa_signal_store_relaxed(deferred->completion_signal, 1);
      queue->signal_pool.push_back(deferred->completion_signal);
      queue->kernarg_pool.push_back(deferred->kernarg);
      queue->deferred_dispatches.erase(deferred);
    }
  }
  if (event) {
    --event->active_synchronizers;
    g_event_state_changed.notify_all();
  }
  --queue->active_synchronizers;
  g_queue_state_changed.notify_all();

  if (value != 0) {
    return LR_ERROR_RUNTIME;
  }
  if (queue->destroying && !allow_destroying) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  return reap_completed_queue_work_locked(queue);
}

lr_status_t
ensure_queue_capacity_locked(RuntimeLock *devices_lock,
                             std::unique_lock<std::mutex> *queue_lock,
                             QueueState *queue, size_t required_packets,
                             bool *lock_released, bool allow_destroying) {
  if (lock_released) {
    *lock_released = false;
  }
  if (required_packets == 0 || required_packets > queue->queue->size) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  while (true) {
    lr_status_t reap_status = reap_completed_queue_work_locked(queue);
    if (reap_status != LR_SUCCESS) {
      return reap_status;
    }
    if (has_queue_capacity_locked(queue, required_packets)) {
      return LR_SUCCESS;
    }

    if (lock_released) {
      *lock_released = true;
    }
    lr_status_t wait_status = wait_for_queue_progress_locked(
        devices_lock, queue_lock, queue, allow_destroying);
    if (wait_status != LR_SUCCESS) {
      return wait_status;
    }
  }
}

lr_status_t enqueue_event_dependencies_locked(
    RuntimeLock *devices_lock, DeviceState *device,
    std::unique_lock<std::mutex> *queue_lock, QueueState *queue,
    hsa_signal_t retirement_signal,
    const std::vector<lr_event_t *> *explicit_dependencies) {
  while (true) {
    lr_status_t reap_status = reap_completed_queue_work_locked(queue);
    if (reap_status != LR_SUCCESS) {
      return reap_status;
    }

    std::vector<lr_event_t *> dependencies;
    if (explicit_dependencies) {
      for (lr_event_t *event : *explicit_dependencies) {
        if (event->pending && !event_has_queue_dependency(event, queue) &&
            hsa_signal_load_scacquire(event->signal) != 0) {
          dependencies.push_back(event);
        }
      }
    } else {
      for (lr_event_t *event : device->pending_events) {
        if (event->pending && event->kind == lr_event_t::Kind::AsyncCopy &&
            !event_has_queue_dependency(event, queue) &&
            hsa_signal_load_scacquire(event->signal) != 0) {
          dependencies.push_back(event);
        }
      }
    }

    const size_t barrier_count = (dependencies.size() + 4) / 5;
    for (lr_event_t *event : dependencies) {
      ++event->active_synchronizers;
    }
    lr_status_t capacity_status = ensure_queue_capacity_locked(
        devices_lock, queue_lock, queue, barrier_count + 1);
    if (capacity_status != LR_SUCCESS) {
      for (lr_event_t *event : dependencies) {
        --event->active_synchronizers;
      }
      g_event_state_changed.notify_all();
      return capacity_status;
    }

    for (size_t offset = 0; offset < dependencies.size(); offset += 5) {
      const uint64_t index =
          hsa_queue_add_write_index_scacq_screl(queue->queue, 1);
      auto *packets =
          static_cast<hsa_barrier_and_packet_t *>(queue->queue->base_address);
      hsa_barrier_and_packet_t *packet =
          &packets[index & (queue->queue->size - 1)];
      std::memset(packet, 0, sizeof(*packet));

      std::vector<lr_event_t *> packet_dependencies;
      const size_t end = std::min(offset + 5, dependencies.size());
      packet_dependencies.reserve(end - offset);
      for (size_t i = offset; i < end; ++i) {
        packet->dep_signal[i - offset] = dependencies[i]->signal;
        retain_event_dependency(dependencies[i], queue);
        packet_dependencies.push_back(dependencies[i]);
      }
      publish_packet_header(&packet->header,
                            barrier_packet_header(HSA_PACKET_TYPE_BARRIER_AND));
      queue->pending_barriers.push_back(
          PendingBarrier{retirement_signal, std::move(packet_dependencies)});
      hsa_signal_store_screlease(queue->queue->doorbell_signal, index);
    }
    for (lr_event_t *event : dependencies) {
      --event->active_synchronizers;
    }
    if (!dependencies.empty()) {
      g_event_state_changed.notify_all();
    }
    return LR_SUCCESS;
  }
}

lr_status_t enqueue_explicit_event_dependencies_locally_locked(
    QueueState *queue, hsa_signal_t retirement_signal,
    const std::vector<lr_event_t *> &explicit_dependencies) {
  // The caller holds the queue lock and pins every event against re-record and
  // destruction. Event completion may race with this path, but a completed
  // signal can simply be omitted (or safely consumed as an already-satisfied
  // barrier dependency).
  std::vector<lr_event_t *> dependencies;
  dependencies.reserve(explicit_dependencies.size());
  for (lr_event_t *event : explicit_dependencies) {
    if (!event_has_queue_dependency(event, queue) &&
        hsa_signal_load_scacquire(event->signal) != 0) {
      dependencies.push_back(event);
    }
  }

  const size_t barrier_count = (dependencies.size() + 4) / 5;
  if (!has_queue_capacity_locked(queue, barrier_count + 1)) {
    // Capacity was reserved while the registry lock was held. Since the queue
    // remains locked, the required packet count cannot increase here.
    return LR_ERROR_RUNTIME;
  }

  for (size_t offset = 0; offset < dependencies.size(); offset += 5) {
    const uint64_t index =
        hsa_queue_add_write_index_scacq_screl(queue->queue, 1);
    auto *packets =
        static_cast<hsa_barrier_and_packet_t *>(queue->queue->base_address);
    hsa_barrier_and_packet_t *packet =
        &packets[index & (queue->queue->size - 1)];
    std::memset(packet, 0, sizeof(*packet));

    std::vector<lr_event_t *> packet_dependencies;
    const size_t end = std::min(offset + 5, dependencies.size());
    packet_dependencies.reserve(end - offset);
    for (size_t i = offset; i < end; ++i) {
      packet->dep_signal[i - offset] = dependencies[i]->signal;
      retain_event_dependency(dependencies[i], queue);
      packet_dependencies.push_back(dependencies[i]);
    }
    publish_packet_header(&packet->header,
                          barrier_packet_header(HSA_PACKET_TYPE_BARRIER_AND));
    queue->pending_barriers.push_back(
        PendingBarrier{retirement_signal, std::move(packet_dependencies)});
    hsa_signal_store_screlease(queue->queue->doorbell_signal, index);
  }
  return LR_SUCCESS;
}

size_t event_dependency_packet_count_locked(
    DeviceState *device, QueueState *queue,
    const std::vector<lr_event_t *> *explicit_dependencies) {
  size_t dependency_count = 0;
  const std::vector<lr_event_t *> &events =
      explicit_dependencies ? *explicit_dependencies : device->pending_events;
  for (lr_event_t *event : events) {
    const bool eligible_kind =
        explicit_dependencies || event->kind == lr_event_t::Kind::AsyncCopy;
    if (eligible_kind && event->pending &&
        !event_has_queue_dependency(event, queue) &&
        hsa_signal_load_scacquire(event->signal) != 0) {
      ++dependency_count;
    }
  }
  return (dependency_count + 4) / 5;
}

void release_device_queue_pools_locked(DeviceState *device,
                                       lr_status_t *result) {
  for (lr_queue_t *queue : device->queues) {
    std::lock_guard<std::mutex> queue_lock(queue->state.mutex);
    release_queue_pools_locked(&queue->state, result);
  }
}

void destroy_all_queues_locked() {
  for (DeviceState &device : g_devices) {
    for (lr_queue_t *queue : device.queues) {
      hsa_queue_destroy(queue->state.queue);
      g_queues.erase(queue);
      delete queue;
    }
    device.queues.clear();
    device.default_queue = nullptr;
  }
  g_queues.clear();
}

void wait_for_queue_synchronizers_locked(RuntimeLock *devices_lock) {
  g_queue_state_changed.wait(*devices_lock, [] {
    for (const DeviceState &device : g_devices) {
      for (const lr_queue_t *queue : device.queues) {
        if (queue->state.active_synchronizers != 0) {
          return false;
        }
      }
    }
    return true;
  });
}

void wait_for_queue_submissions_locked(RuntimeLock *devices_lock) {
  std::vector<QueueState *> queues;
  for (DeviceState &device : g_devices) {
    for (lr_queue_t *queue : device.queues) {
      queues.push_back(&queue->state);
    }
  }
  devices_lock->unlock();
  for (QueueState *queue : queues) {
    queue->active_submissions.wait_until_empty();
  }
  devices_lock->lock();
}

lr_status_t enqueue_queue_synchronization_locked(
    QueueState *queue, hsa_signal_t *completion_signal,
    RuntimeLock *devices_lock, std::unique_lock<std::mutex> *queue_lock) {
  *completion_signal = hsa_signal_t{};
  if (queue->destroying) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  std::vector<SynchronizationDependency> dependencies;
  std::vector<hsa_signal_t> dispatch_dependencies;
  size_t packet_count = 0;
  while (true) {
    lr_status_t reap_status = reap_completed_queue_work_locked(queue);
    if (reap_status != LR_SUCCESS) {
      return reap_status;
    }

    dependencies.clear();
    dependencies.reserve(queue->pending_dispatches.size() +
                         queue->pending_events.size());
    dispatch_dependencies.clear();
    dispatch_dependencies.reserve(queue->pending_dispatches.size());
    for (const PendingDispatch &dispatch : queue->pending_dispatches) {
      dependencies.push_back(
          SynchronizationDependency{dispatch.completion_signal, nullptr});
      dispatch_dependencies.push_back(dispatch.completion_signal);
    }
    for (lr_event_t *event : queue->pending_events) {
      if (event->pending && hsa_signal_load_scacquire(event->signal) != 0) {
        dependencies.push_back(SynchronizationDependency{event->signal, event});
      }
    }
    if (dependencies.empty()) {
      return LR_SUCCESS;
    }

    packet_count = (dependencies.size() + 4) / 5;
    if (pending_queue_packet_count(queue) + packet_count <=
        queue->queue->size) {
      break;
    }

    lr_status_t wait_status =
        wait_for_queue_progress_locked(devices_lock, queue_lock, queue, false);
    if (wait_status != LR_SUCCESS) {
      return wait_status;
    }
  }

  hsa_status_t status =
      hsa_signal_create(static_cast<hsa_signal_value_t>(packet_count), 0,
                        nullptr, completion_signal);
  if (status != HSA_STATUS_SUCCESS) {
    return to_lr_status(status);
  }

  uint64_t last_index = 0;
  for (size_t offset = 0; offset < dependencies.size(); offset += 5) {
    last_index = hsa_queue_add_write_index_scacq_screl(queue->queue, 1);
    auto *packets =
        static_cast<hsa_barrier_and_packet_t *>(queue->queue->base_address);
    hsa_barrier_and_packet_t *packet =
        &packets[last_index & (queue->queue->size - 1)];
    std::memset(packet, 0, sizeof(*packet));
    const size_t end = std::min(offset + 5, dependencies.size());
    for (size_t i = offset; i < end; ++i) {
      packet->dep_signal[i - offset] = dependencies[i].signal;
    }
    packet->completion_signal = *completion_signal;
    publish_packet_header(&packet->header,
                          barrier_packet_header(HSA_PACKET_TYPE_BARRIER_AND));

    std::vector<lr_event_t *> event_dependencies;
    for (size_t i = offset; i < end; ++i) {
      if (dependencies[i].event) {
        retain_event_dependency(dependencies[i].event, queue);
        event_dependencies.push_back(dependencies[i].event);
      }
    }
    queue->pending_barriers.push_back(
        PendingBarrier{*completion_signal, std::move(event_dependencies)});
  }

  queue->pending_synchronizations.push_back(PendingSynchronization{
      *completion_signal, std::move(dispatch_dependencies)});
  ++queue->active_synchronizers;
  hsa_signal_store_screlease(queue->queue->doorbell_signal, last_index);
  return LR_SUCCESS;
}

lr_status_t enqueue_queue_tail_marker_locked(
    QueueState *queue, hsa_signal_t *completion_signal,
    RuntimeLock *devices_lock, std::unique_lock<std::mutex> *queue_lock,
    bool allow_destroying) {
  *completion_signal = hsa_signal_t{};
  lr_status_t capacity_status = ensure_queue_capacity_locked(
      devices_lock, queue_lock, queue, 1, nullptr, allow_destroying);
  if (capacity_status != LR_SUCCESS) {
    return capacity_status;
  }

  hsa_status_t status = hsa_signal_create(1, 0, nullptr, completion_signal);
  if (status != HSA_STATUS_SUCCESS) {
    return to_lr_status(status);
  }

  const uint64_t index = hsa_queue_add_write_index_scacq_screl(queue->queue, 1);
  auto *packets =
      static_cast<hsa_barrier_and_packet_t *>(queue->queue->base_address);
  hsa_barrier_and_packet_t *packet = &packets[index & (queue->queue->size - 1)];
  std::memset(packet, 0, sizeof(*packet));
  packet->completion_signal = *completion_signal;
  // The barrier bit makes this marker complete after every packet already
  // published to the queue, without borrowing any earlier packet's signal.
  publish_packet_header(&packet->header,
                        barrier_packet_header(HSA_PACKET_TYPE_BARRIER_AND));

  queue->pending_barriers.push_back(PendingBarrier{*completion_signal, {}});
  queue->pending_synchronizations.push_back(
      PendingSynchronization{*completion_signal, {}});
  ++queue->active_synchronizers;
  hsa_signal_store_screlease(queue->queue->doorbell_signal, index);
  return LR_SUCCESS;
}

lr_status_t finish_queue_synchronization_locked(QueueState *queue,
                                                hsa_signal_t completion_signal,
                                                hsa_signal_value_t wait_value) {
  auto pending = std::find_if(
      queue->pending_synchronizations.begin(),
      queue->pending_synchronizations.end(),
      [completion_signal](const PendingSynchronization &synchronization) {
        return synchronization.completion_signal.handle ==
               completion_signal.handle;
      });
  if (pending != queue->pending_synchronizations.end()) {
    queue->pending_synchronizations.erase(pending);
  }
  --queue->active_synchronizers;
  g_queue_state_changed.notify_all();

  lr_status_t result = wait_value == 0 ? LR_SUCCESS : LR_ERROR_RUNTIME;
  lr_status_t reap_status = reap_completed_queue_work_locked(queue);
  if (result == LR_SUCCESS) {
    result = reap_status;
  }
  hsa_status_t status = hsa_signal_destroy(completion_signal);
  if (result == LR_SUCCESS && status != HSA_STATUS_SUCCESS) {
    result = to_lr_status(status);
  }
  return result;
}

lr_status_t prepare_queue_destruction_locked(
    QueueState *queue, QueueDestruction *destruction, RuntimeLock *devices_lock,
    std::unique_lock<std::mutex> *queue_lock) {
  destruction->completion_signal = hsa_signal_t{};
  destruction->wait_value = 0;
  destruction->destruction_pin_count = 1;
  queue->destroying = true;
  if (!queue->active_submissions.empty()) {
    queue_lock->unlock();
    devices_lock->unlock();
    queue->active_submissions.wait_until_empty();
    devices_lock->lock();
    queue_lock->lock();
  }
  ++queue->active_synchronizers;
  while (queue->active_synchronizers != 1) {
    queue_lock->unlock();
    g_queue_state_changed.wait(
        *devices_lock, [queue] { return queue->active_synchronizers == 1; });
    queue_lock->lock();
  }

  lr_status_t status = enqueue_queue_tail_marker_locked(
      queue, &destruction->completion_signal, devices_lock, queue_lock, true);
  if (status != LR_SUCCESS) {
    --queue->active_synchronizers;
    g_queue_state_changed.notify_all();
    queue->destroying = false;
    return status;
  }
  ++destruction->destruction_pin_count;
  return LR_SUCCESS;
}

void wait_for_queue_destruction(QueueDestruction *destruction) {
  if (destruction->completion_signal.handle == 0) {
    return;
  }
  destruction->wait_value = hsa_signal_wait_scacquire(
      destruction->completion_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
      HSA_WAIT_STATE_BLOCKED);
}

lr_status_t finish_queue_destruction_locked(
    QueueState *queue, const QueueDestruction &destruction,
    RuntimeLock *devices_lock, std::unique_lock<std::mutex> *queue_lock) {
  while (queue->active_synchronizers != destruction.destruction_pin_count) {
    queue_lock->unlock();
    g_queue_state_changed.wait(*devices_lock, [queue, &destruction] {
      return queue->active_synchronizers == destruction.destruction_pin_count;
    });
    queue_lock->lock();
  }
  lr_status_t result = finish_queue_synchronization_locked(
      queue, destruction.completion_signal, destruction.wait_value);
  --queue->active_synchronizers;
  g_queue_state_changed.notify_all();
  if (result == LR_SUCCESS) {
    result = drain_queue_locked(queue);
  }
  return result;
}
#endif

} // namespace lrrt_internal

using namespace lrrt_internal;

extern "C" {

lr_status_t lr_queue_create(lr_device_t device, lr_queue_t **queue) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!queue) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  *queue = nullptr;
#if LRRT_ENABLE_HSA
  std::lock_guard<RuntimeMutex> lock(g_devices_mutex);
  if (device.index >= g_devices.size() ||
      !g_devices[device.index].default_queue) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  hsa_status_t status =
      create_queue(device, &g_devices[device.index], false, queue);
  return to_lr_status(status);
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_queue_destroy(lr_queue_t *queue) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!queue) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
#if LRRT_ENABLE_HSA
  RuntimeLock lock(g_devices_mutex);
  auto queue_entry = g_queues.find(queue);
  if (queue_entry == g_queues.end() || queue->is_default ||
      queue->device.index >= g_devices.size() || queue->state.destroying) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  QueueDestruction destruction{};
  std::unique_lock<std::mutex> queue_lock(queue->state.mutex);
  lr_status_t result = prepare_queue_destruction_locked(
      &queue->state, &destruction, &lock, &queue_lock);
  if (result != LR_SUCCESS) {
    return result;
  }

  queue_lock.unlock();
  lock.unlock();
  wait_for_queue_destruction(&destruction);
  lock.lock();
  queue_lock.lock();

  DeviceState &device = g_devices[queue->device.index];
  result = finish_queue_destruction_locked(&queue->state, destruction, &lock,
                                           &queue_lock);
  release_queue_pools_locked(&queue->state, &result);
  if (result != LR_SUCCESS) {
    queue->state.destroying = false;
    return result;
  }
  hsa_status_t status = hsa_queue_destroy(queue->state.queue);
  if (status != HSA_STATUS_SUCCESS) {
    queue->state.destroying = false;
    return to_lr_status(status);
  }
  auto device_queue =
      std::find(device.queues.begin(), device.queues.end(), queue);
  if (device_queue != device.queues.end()) {
    device.queues.erase(device_queue);
  }
  g_queues.erase(queue_entry);
  queue_lock.unlock();
  delete queue;
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_queue_synchronize(lr_queue_t *queue) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!queue) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
#if LRRT_ENABLE_HSA
  RuntimeLock lock(g_devices_mutex);
  if (!valid_queue_locked(queue) || queue->device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  QueueState &state = queue->state;
  std::unique_lock<std::mutex> queue_lock(state.mutex);
  hsa_signal_t completion_signal{};
  lr_status_t status = enqueue_queue_synchronization_locked(
      &state, &completion_signal, &lock, &queue_lock);
  if (status != LR_SUCCESS || completion_signal.handle == 0) {
    return status;
  }

  queue_lock.unlock();
  lock.unlock();
  hsa_signal_value_t value =
      hsa_signal_wait_scacquire(completion_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  lock.lock();
  queue_lock.lock();

  return finish_queue_synchronization_locked(&state, completion_signal, value);
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

} // extern "C"
