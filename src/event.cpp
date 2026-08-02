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

lr_status_t event_wait_locked(lr_event_t *event) {
  if (!event->pending) {
    return LR_SUCCESS;
  }

  DeviceState &device = g_devices[event->device.index];
  hsa_signal_value_t value =
      hsa_signal_wait_scacquire(event->signal, HSA_SIGNAL_CONDITION_LT, 1,
                                UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  auto pending = std::find(device.pending_events.begin(),
                           device.pending_events.end(), event);
  if (pending != device.pending_events.end()) {
    *pending = device.pending_events.back();
    device.pending_events.pop_back();
  }
  if (event->recorded_queue) {
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
    --dependency->dependency_count;
  }
  event->dependencies.clear();
  event->kind = lr_event_t::Kind::None;
  event->completed = result == LR_SUCCESS;
  return result;
}

void reap_completed_event_dependencies_locked(lr_event_t *event) {
  std::vector<QueueState *> queues(event->dependency_queues.begin(),
                                   event->dependency_queues.end());
  for (QueueState *queue : queues) {
    reap_completed_barriers_locked(queue);
  }
}

lr_status_t wait_for_event_consumers_locked(DeviceState *device,
                                            lr_event_t *event) {
  while (event->dependency_count != 0) {
    reap_completed_event_dependencies_locked(event);
    if (event->dependency_count == 0) {
      return LR_SUCCESS;
    }

    bool waited = false;
    std::vector<lr_event_t *> pending_events = device->pending_events;
    for (lr_event_t *dependent : pending_events) {
      if (!dependent->pending ||
          std::find(dependent->dependencies.begin(),
                    dependent->dependencies.end(),
                    event) == dependent->dependencies.end()) {
        continue;
      }
      lr_status_t status = event_wait_locked(dependent);
      if (status != LR_SUCCESS) {
        return status;
      }
      waited = true;
    }

    std::vector<QueueState *> queues(event->dependency_queues.begin(),
                                     event->dependency_queues.end());
    for (QueueState *queue : queues) {
      auto barrier = std::find_if(
          queue->pending_barriers.begin(), queue->pending_barriers.end(),
          [event](const PendingBarrier &pending) {
            return std::find(pending.dependencies.begin(),
                             pending.dependencies.end(),
                             event) != pending.dependencies.end();
          });
      if (barrier == queue->pending_barriers.end()) {
        continue;
      }
      hsa_signal_value_t value = hsa_signal_wait_scacquire(
          barrier->retirement_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
          HSA_WAIT_STATE_BLOCKED);
      if (value != 0) {
        return LR_ERROR_RUNTIME;
      }
      reap_completed_barriers_locked(queue);
      waited = true;
    }

    if (!waited && event->dependency_count != 0) {
      return LR_ERROR_RUNTIME;
    }
  }
  return LR_SUCCESS;
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
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  auto event_entry = g_events.find(event);
  if (event_entry == g_events.end() ||
      event->device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  lr_status_t wait_status = event_wait_locked(event);
  if (wait_status != LR_SUCCESS) {
    return wait_status;
  }
  lr_status_t consumer_status =
      wait_for_event_consumers_locked(&g_devices[event->device.index], event);
  if (consumer_status != LR_SUCCESS) {
    return consumer_status;
  }

  hsa_status_t status = hsa_signal_destroy(event->signal);
  if (status != HSA_STATUS_SUCCESS) {
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
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (g_events.find(event) == g_events.end() ||
      event->device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  DeviceState &device = g_devices[event->device.index];
  if (use_default_queue) {
    queue = device.default_queue;
  }
  if (!queue || !valid_queue_locked(queue) ||
      queue->device.index != event->device.index) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  QueueState &state = queue->state;

  lr_status_t wait_status = event_wait_locked(event);
  if (wait_status != LR_SUCCESS) {
    return wait_status;
  }
  lr_status_t consumer_status = wait_for_event_consumers_locked(&device, event);
  if (consumer_status != LR_SUCCESS) {
    return consumer_status;
  }
  lr_status_t capacity_status =
      use_default_queue ? enqueue_event_dependencies_locked(
                              &device, &state, event->signal, nullptr)
                        : ensure_queue_capacity_locked(&state, 1);
  if (capacity_status != LR_SUCCESS) {
    return capacity_status;
  }

  hsa_signal_store_relaxed(event->signal, 1);
  event->kind = lr_event_t::Kind::Marker;
  event->completed = false;
  event->dependency_queues.clear();
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
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (g_events.find(event) == g_events.end() ||
      event->device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  return event_wait_locked(event);
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
