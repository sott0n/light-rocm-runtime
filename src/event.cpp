#include "runtime_internal.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace lrrt_internal {

#if LRRT_ENABLE_HSA
std::unordered_set<lr_event_t *> g_events;

bool valid_event_locked(lr_event_t *event) {
  return g_events.find(event) != g_events.end();
}

bool event_has_queue_dependency(lr_event_t *event, QueueState *queue) {
  std::lock_guard<std::mutex> lock(event->dependency_mutex);
  return event->dependency_queues.count(queue) != 0;
}

void retain_event_dependency(lr_event_t *event, QueueState *queue) {
  std::lock_guard<std::mutex> lock(event->dependency_mutex);
  ++event->dependency_count;
  if (queue) {
    event->dependency_queues.insert(queue);
  }
}

void release_event_dependency(lr_event_t *event, QueueState *queue) {
  std::lock_guard<std::mutex> lock(event->dependency_mutex);
  --event->dependency_count;
  if (queue) {
    event->dependency_queues.erase(queue);
  }
}

size_t event_dependency_count(lr_event_t *event) {
  std::lock_guard<std::mutex> lock(event->dependency_mutex);
  return event->dependency_count;
}

std::vector<QueueState *> event_dependency_queues(lr_event_t *event) {
  std::lock_guard<std::mutex> lock(event->dependency_mutex);
  return {event->dependency_queues.begin(), event->dependency_queues.end()};
}

void clear_event_dependency_queues(lr_event_t *event) {
  std::lock_guard<std::mutex> lock(event->dependency_mutex);
  event->dependency_queues.clear();
}

lr_status_t finish_event_wait_locked(lr_event_t *event,
                                     hsa_signal_value_t value,
                                     QueueState *locked_queue) {
  if (!event->pending) {
    return LR_SUCCESS;
  }

  DeviceState &device = g_devices[event->device.index];
  auto pending = std::find(device.pending_events.begin(),
                           device.pending_events.end(), event);
  if (pending != device.pending_events.end()) {
    *pending = device.pending_events.back();
    device.pending_events.pop_back();
  }
  if (event->recorded_queue) {
    std::unique_lock<std::mutex> recorded_queue_lock;
    if (event->recorded_queue != locked_queue) {
      recorded_queue_lock =
          std::unique_lock<std::mutex>(event->recorded_queue->mutex);
    }
    if (value == 0) {
      // A marker signal may also retire copy-dependency barriers submitted
      // immediately before that marker. Release those references before the
      // marker signal can be reset for reuse.
      reap_completed_barriers_locked(event->recorded_queue);
    }
    auto queue_pending =
        std::find(event->recorded_queue->pending_events.begin(),
                  event->recorded_queue->pending_events.end(), event);
    if (queue_pending != event->recorded_queue->pending_events.end()) {
      *queue_pending = event->recorded_queue->pending_events.back();
      event->recorded_queue->pending_events.pop_back();
    }
    event->recorded_queue = nullptr;
  }

  event->pending = false;
  lr_status_t result = value == 0 ? LR_SUCCESS : LR_ERROR_RUNTIME;
  if (result == LR_SUCCESS && event->kind == lr_event_t::Kind::Marker) {
    hsa_amd_profiling_dispatch_time_t time{};
    hsa_status_t status =
        hsa_amd_profiling_get_dispatch_time(device.agent, event->signal, &time);
    if (status != HSA_STATUS_SUCCESS) {
      result = to_lr_status(status);
    } else {
      event->start_tick = time.start;
      event->completion_tick = time.end;
    }
  } else if (result == LR_SUCCESS &&
             event->kind == lr_event_t::Kind::AsyncCopy) {
    hsa_amd_profiling_async_copy_time_t time{};
    hsa_status_t status =
        hsa_amd_profiling_get_async_copy_time(event->signal, &time);
    if (status != HSA_STATUS_SUCCESS) {
      result = to_lr_status(status);
    } else {
      event->start_tick = time.start;
      event->completion_tick = time.end;
    }
  } else if (result == LR_SUCCESS) {
    result = LR_ERROR_INVALID_ARGUMENT;
  }

  if (event->locked_host_ptr) {
    hsa_status_t status = hsa_amd_memory_unlock(event->locked_host_ptr);
    event->locked_host_ptr = nullptr;
    if (status != HSA_STATUS_SUCCESS && result == LR_SUCCESS) {
      result = to_lr_status(status);
    }
  }

  for (lr_event_t *dependency : event->dependencies) {
    release_event_dependency(dependency);
  }
  event->dependencies.clear();
  event->kind = lr_event_t::Kind::None;
  event->completed = result == LR_SUCCESS;
  return result;
}

lr_status_t event_wait_locked(lr_event_t *event, QueueState *locked_queue) {
  if (!event->pending) {
    return LR_SUCCESS;
  }

  hsa_signal_value_t value =
      hsa_signal_wait_scacquire(event->signal, HSA_SIGNAL_CONDITION_LT, 1,
                                UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  return finish_event_wait_locked(event, value, locked_queue);
}

void wait_for_event_synchronizers_locked(
    std::unique_lock<std::mutex> *devices_lock, lr_event_t *event) {
  g_event_state_changed.wait(*devices_lock, [event] {
    return !valid_event_locked(event) || event->active_synchronizers == 0;
  });
}

void wait_for_all_event_synchronizers_locked(
    std::unique_lock<std::mutex> *devices_lock) {
  g_event_state_changed.wait(*devices_lock, [] {
    return std::all_of(g_events.begin(), g_events.end(), [](lr_event_t *event) {
      return event->active_synchronizers == 0;
    });
  });
}

void reap_completed_event_dependencies_locked(lr_event_t *event) {
  std::vector<QueueState *> queues = event_dependency_queues(event);
  for (QueueState *queue : queues) {
    std::lock_guard<std::mutex> queue_lock(queue->mutex);
    reap_completed_barriers_locked(queue);
  }
}

lr_status_t
wait_for_event_consumers_locked(std::unique_lock<std::mutex> *devices_lock,
                                DeviceState *device, lr_event_t *event) {
  ++event->active_synchronizers;
  lr_status_t result = LR_SUCCESS;
  while (event_dependency_count(event) != 0) {
    reap_completed_event_dependencies_locked(event);
    if (event_dependency_count(event) == 0) {
      break;
    }

    auto dependent = std::find_if(
        device->pending_events.begin(), device->pending_events.end(),
        [event](lr_event_t *pending) {
          return pending->pending &&
                 std::find(pending->dependencies.begin(),
                           pending->dependencies.end(),
                           event) != pending->dependencies.end();
        });
    if (dependent != device->pending_events.end()) {
      lr_event_t *dependent_event = *dependent;
      hsa_signal_t signal = dependent_event->signal;
      QueueState *recorded_queue = dependent_event->recorded_queue;
      ++dependent_event->active_synchronizers;
      if (recorded_queue) {
        ++recorded_queue->active_synchronizers;
      }

      devices_lock->unlock();
      hsa_signal_value_t value =
          hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                    UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
      devices_lock->lock();

      lr_status_t status = finish_event_wait_locked(dependent_event, value);
      --dependent_event->active_synchronizers;
      g_event_state_changed.notify_all();
      if (recorded_queue) {
        --recorded_queue->active_synchronizers;
        g_queue_state_changed.notify_all();
      }
      if (status != LR_SUCCESS) {
        result = status;
        break;
      }
      continue;
    }

    std::vector<QueueState *> dependency_queues =
        event_dependency_queues(event);
    if (!dependency_queues.empty()) {
      QueueState *queue = dependency_queues.front();
      ++queue->active_synchronizers;
      std::unique_lock<std::mutex> queue_lock(queue->mutex);
      hsa_signal_t completion_signal{};
      lr_status_t status = enqueue_queue_tail_marker_locked(
          queue, &completion_signal, devices_lock, &queue_lock);
      if (status != LR_SUCCESS) {
        --queue->active_synchronizers;
        g_queue_state_changed.notify_all();
        result = status;
        break;
      }

      queue_lock.unlock();
      devices_lock->unlock();
      hsa_signal_value_t value =
          hsa_signal_wait_scacquire(completion_signal, HSA_SIGNAL_CONDITION_LT,
                                    1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
      devices_lock->lock();
      queue_lock.lock();

      status =
          finish_queue_synchronization_locked(queue, completion_signal, value);
      --queue->active_synchronizers;
      g_queue_state_changed.notify_all();
      if (status != LR_SUCCESS) {
        result = status;
        break;
      }
      continue;
    }

    result = LR_ERROR_RUNTIME;
    break;
  }
  --event->active_synchronizers;
  g_event_state_changed.notify_all();
  return result;
}

lr_status_t collect_event_dependencies_locked(
    lr_device_t device, lr_event_t *const *dependencies,
    size_t dependency_count, const lr_event_t *completion_event,
    std::vector<lr_event_t *> *pending_dependencies) {
  if (dependency_count != 0 && !dependencies) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (dependency_count > UINT32_MAX) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  std::unordered_set<lr_event_t *> unique_dependencies;
  pending_dependencies->clear();
  pending_dependencies->reserve(dependency_count);
  for (size_t i = 0; i < dependency_count; ++i) {
    lr_event_t *dependency = dependencies[i];
    if (!dependency || dependency == completion_event ||
        g_events.find(dependency) == g_events.end() ||
        dependency->device.index != device.index ||
        (!dependency->pending && !dependency->completed) ||
        !unique_dependencies.insert(dependency).second) {
      return LR_ERROR_INVALID_ARGUMENT;
    }
    if (dependency->pending &&
        hsa_signal_load_scacquire(dependency->signal) != 0) {
      pending_dependencies->push_back(dependency);
    }
  }
  return LR_SUCCESS;
}

void release_events_locked() {
  for (lr_event_t *event : g_events) {
    hsa_signal_destroy(event->signal);
    delete event;
  }
  g_events.clear();
}
#endif

} // namespace lrrt_internal

using namespace lrrt_internal;

extern "C" {

lr_status_t lr_event_create(lr_device_t device, lr_event_t **event) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!valid_device(device) || !event) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  *event = nullptr;
#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (device.index >= g_devices.size() ||
      !g_devices[device.index].default_queue) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  auto *created_event = new lr_event_t{};
  created_event->device = device;
  hsa_status_t status =
      hsa_signal_create(1, 0, nullptr, &created_event->signal);
  if (status != HSA_STATUS_SUCCESS) {
    delete created_event;
    return to_lr_status(status);
  }

  created_event->kind = lr_event_t::Kind::None;
  created_event->pending = false;
  created_event->completed = false;
  created_event->active_synchronizers = 0;
  created_event->destroying = false;
  created_event->dependency_count = 0;
  created_event->start_tick = 0;
  created_event->completion_tick = 0;
  created_event->recorded_queue = nullptr;
  created_event->locked_host_ptr = nullptr;
  g_events.insert(created_event);
  *event = created_event;
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_event_destroy(lr_event_t *event) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!event) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

#if LRRT_ENABLE_HSA
  std::unique_lock<std::mutex> lock(g_devices_mutex);
  auto event_entry = g_events.find(event);
  if (event_entry == g_events.end() ||
      event->device.index >= g_devices.size() || event->destroying) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  event->destroying = true;
  wait_for_event_synchronizers_locked(&lock, event);

  lr_status_t wait_status = event_wait_locked(event);
  if (wait_status != LR_SUCCESS) {
    event->destroying = false;
    return wait_status;
  }
  lr_status_t consumer_status = wait_for_event_consumers_locked(
      &lock, &g_devices[event->device.index], event);
  if (consumer_status != LR_SUCCESS) {
    event->destroying = false;
    return consumer_status;
  }

  hsa_status_t status = hsa_signal_destroy(event->signal);
  if (status != HSA_STATUS_SUCCESS) {
    event->destroying = false;
    return to_lr_status(status);
  }

  g_events.erase(event_entry);
  delete event;
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

static lr_status_t event_record_impl(lr_event_t *event, lr_queue_t *queue,
                                     bool use_default_queue) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!event) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

#if LRRT_ENABLE_HSA
  std::unique_lock<std::mutex> lock(g_devices_mutex);
  if (g_events.find(event) == g_events.end() ||
      event->device.index >= g_devices.size() || event->destroying) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  wait_for_event_synchronizers_locked(&lock, event);
  if (!valid_event_locked(event) || event->destroying) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  lr_status_t wait_status = event_wait_locked(event);
  if (wait_status != LR_SUCCESS) {
    return wait_status;
  }
  DeviceState &device = g_devices[event->device.index];
  lr_status_t consumer_status =
      wait_for_event_consumers_locked(&lock, &device, event);
  if (consumer_status != LR_SUCCESS) {
    return consumer_status;
  }
  if (!valid_event_locked(event) || event->destroying) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  if (use_default_queue) {
    queue = device.default_queue;
  }
  if (!queue || !valid_queue_locked(queue) ||
      queue->device.index != event->device.index) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  QueueState &state = queue->state;
  std::unique_lock<std::mutex> queue_lock(state.mutex);
  ++event->active_synchronizers;
  lr_status_t capacity_status =
      use_default_queue
          ? enqueue_event_dependencies_locked(&lock, &device, &queue_lock,
                                              &state, event->signal, nullptr)
          : ensure_queue_capacity_locked(&lock, &queue_lock, &state, 1);
  --event->active_synchronizers;
  g_event_state_changed.notify_all();
  if (capacity_status != LR_SUCCESS) {
    return capacity_status;
  }

  hsa_signal_store_relaxed(event->signal, 1);
  event->kind = lr_event_t::Kind::Marker;
  event->completed = false;
  clear_event_dependency_queues(event);
  event->start_tick = 0;
  event->completion_tick = 0;
  event->recorded_queue = &state;

  const uint64_t index = hsa_queue_add_write_index_scacq_screl(state.queue, 1);
  auto *packets =
      static_cast<hsa_barrier_and_packet_t *>(state.queue->base_address);
  hsa_barrier_and_packet_t *packet = &packets[index & (state.queue->size - 1)];
  std::memset(packet, 0, sizeof(*packet));
  packet->completion_signal = event->signal;
  publish_packet_header(&packet->header,
                        barrier_packet_header(HSA_PACKET_TYPE_BARRIER_AND));

  event->pending = true;
  state.pending_events.push_back(event);
  device.pending_events.push_back(event);
  hsa_signal_store_screlease(state.queue->doorbell_signal, index);
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_event_record(lr_event_t *event) {
  return event_record_impl(event, nullptr, true);
}

lr_status_t lr_event_record_on_queue(lr_event_t *event, lr_queue_t *queue) {
  return event_record_impl(event, queue, false);
}

lr_status_t lr_event_synchronize(lr_event_t *event) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!event) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

#if LRRT_ENABLE_HSA
  std::unique_lock<std::mutex> lock(g_devices_mutex);
  if (!valid_event_locked(event) || event->device.index >= g_devices.size() ||
      event->destroying) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (!event->pending) {
    return LR_SUCCESS;
  }

  hsa_signal_t signal = event->signal;
  QueueState *recorded_queue = event->recorded_queue;
  ++event->active_synchronizers;
  if (recorded_queue) {
    ++recorded_queue->active_synchronizers;
  }

  lock.unlock();
  hsa_signal_value_t value = hsa_signal_wait_scacquire(
      signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  lock.lock();

  lr_status_t result = finish_event_wait_locked(event, value);
  --event->active_synchronizers;
  g_event_state_changed.notify_all();
  if (recorded_queue) {
    --recorded_queue->active_synchronizers;
    g_queue_state_changed.notify_all();
  }
  return result;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_event_elapsed_time_ns(const lr_event_t *start,
                                     const lr_event_t *end,
                                     uint64_t *elapsed_ns) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!start || !end || !elapsed_ns) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  *elapsed_ns = 0;
#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (g_events.find(const_cast<lr_event_t *>(start)) == g_events.end() ||
      g_events.find(const_cast<lr_event_t *>(end)) == g_events.end() ||
      start->device.index != end->device.index || start->pending ||
      end->pending || !start->completed || !end->completed) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  uint64_t frequency = 0;
  hsa_status_t status =
      hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency);
  if (status != HSA_STATUS_SUCCESS) {
    return to_lr_status(status);
  }
  if (frequency == 0 || end->completion_tick < start->completion_tick) {
    return LR_ERROR_RUNTIME;
  }

  uint64_t ticks = end->completion_tick - start->completion_tick;
  *elapsed_ns = static_cast<uint64_t>(
      (static_cast<unsigned __int128>(ticks) * 1000000000ULL) / frequency);
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_event_duration_ns(const lr_event_t *event,
                                 uint64_t *duration_ns) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!event || !duration_ns) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  *duration_ns = 0;
#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (g_events.find(const_cast<lr_event_t *>(event)) == g_events.end() ||
      event->device.index >= g_devices.size() || event->pending ||
      !event->completed || event->completion_tick < event->start_tick) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  uint64_t frequency = 0;
  hsa_status_t status =
      hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency);
  if (status != HSA_STATUS_SUCCESS) {
    return to_lr_status(status);
  }
  if (frequency == 0) {
    return LR_ERROR_RUNTIME;
  }

  const uint64_t ticks = event->completion_tick - event->start_tick;
  *duration_ns = static_cast<uint64_t>(
      (static_cast<unsigned __int128>(ticks) * 1000000000ULL) / frequency);
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

} // extern "C"
