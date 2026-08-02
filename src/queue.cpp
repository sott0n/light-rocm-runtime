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

bool valid_queue_locked(lr_queue_t *queue) {
  return g_queues.find(queue) != g_queues.end();
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
      --event->dependency_count;
      event->dependency_queues.erase(queue);
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
    hsa_signal_store_relaxed(dispatch.completion_signal, 1);
    queue->signal_pool.push_back(dispatch.completion_signal);
    queue->kernarg_pool.push_back(dispatch.kernarg);
  }
  queue->pending_dispatches.clear();
  return result;
}

lr_status_t drain_queue_locked(QueueState *queue) {
  lr_status_t result = LR_SUCCESS;
  std::vector<lr_event_t *> pending_events = queue->pending_events;
  for (lr_event_t *event : pending_events) {
    lr_status_t status = event_wait_locked(event);
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
      --event->dependency_count;
      event->dependency_queues.erase(queue);
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

    hsa_signal_store_relaxed(dispatch.completion_signal, 1);
    queue->signal_pool.push_back(dispatch.completion_signal);
    queue->kernarg_pool.push_back(dispatch.kernarg);
    ++completed_count;
  }
  if (completed_count != 0) {
    queue->pending_dispatches.erase(queue->pending_dispatches.begin(),
                                    queue->pending_dispatches.begin() +
                                        completed_count);
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

    lr_status_t status = event_wait_locked(event);
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

lr_status_t wait_for_queue_progress_locked(QueueState *queue) {
  if (!queue->pending_dispatches.empty()) {
    hsa_signal_value_t value = hsa_signal_wait_scacquire(
        queue->pending_dispatches.front().completion_signal,
        HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    if (value != 0) {
      return LR_ERROR_RUNTIME;
    }
  } else if (!queue->pending_events.empty()) {
    lr_status_t status = event_wait_locked(queue->pending_events.front());
    if (status != LR_SUCCESS) {
      return status;
    }
  } else if (!queue->pending_barriers.empty()) {
    hsa_signal_value_t value = hsa_signal_wait_scacquire(
        queue->pending_barriers.front().retirement_signal,
        HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    if (value != 0) {
      return LR_ERROR_RUNTIME;
    }
  } else {
    return LR_ERROR_RUNTIME;
  }
  return reap_completed_queue_work_locked(queue);
}

lr_status_t ensure_queue_capacity_locked(QueueState *queue,
                                         size_t required_packets) {
  if (required_packets == 0 || required_packets > queue->queue->size) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  while (true) {
    lr_status_t reap_status = reap_completed_queue_work_locked(queue);
    if (reap_status != LR_SUCCESS) {
      return reap_status;
    }
    if (pending_queue_packet_count(queue) + required_packets <=
        queue->queue->size) {
      return LR_SUCCESS;
    }

    lr_status_t wait_status = wait_for_queue_progress_locked(queue);
    if (wait_status != LR_SUCCESS) {
      return wait_status;
    }
  }
}

lr_status_t enqueue_event_dependencies_locked(
    DeviceState *device, QueueState *queue, hsa_signal_t retirement_signal,
    const std::vector<lr_event_t *> *explicit_dependencies) {
  while (true) {
    lr_status_t reap_status = reap_completed_queue_work_locked(queue);
    if (reap_status != LR_SUCCESS) {
      return reap_status;
    }

    std::vector<lr_event_t *> dependencies;
    if (explicit_dependencies) {
      for (lr_event_t *event : *explicit_dependencies) {
        if (event->pending && event->dependency_queues.count(queue) == 0 &&
            hsa_signal_load_scacquire(event->signal) != 0) {
          dependencies.push_back(event);
        }
      }
    } else {
      for (lr_event_t *event : device->pending_events) {
        if (event->pending && event->kind == lr_event_t::Kind::AsyncCopy &&
            event->dependency_queues.count(queue) == 0 &&
            hsa_signal_load_scacquire(event->signal) != 0) {
          dependencies.push_back(event);
        }
      }
    }

    const size_t barrier_count = (dependencies.size() + 4) / 5;
    lr_status_t capacity_status =
        ensure_queue_capacity_locked(queue, barrier_count + 1);
    if (capacity_status != LR_SUCCESS) {
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
        ++dependencies[i]->dependency_count;
        dependencies[i]->dependency_queues.insert(queue);
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
}

void release_device_queue_pools_locked(DeviceState *device,
                                       lr_status_t *result) {
  for (lr_queue_t *queue : device->queues) {
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
  std::lock_guard<std::mutex> lock(g_devices_mutex);
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
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  auto queue_entry = g_queues.find(queue);
  if (queue_entry == g_queues.end() || queue->is_default ||
      queue->device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  DeviceState &device = g_devices[queue->device.index];
  lr_status_t result = drain_queue_locked(&queue->state);
  release_queue_pools_locked(&queue->state, &result);
  if (result != LR_SUCCESS) {
    return result;
  }
  hsa_status_t status = hsa_queue_destroy(queue->state.queue);
  if (status != HSA_STATUS_SUCCESS) {
    return to_lr_status(status);
  }
  auto device_queue =
      std::find(device.queues.begin(), device.queues.end(), queue);
  if (device_queue != device.queues.end()) {
    device.queues.erase(device_queue);
  }
  g_queues.erase(queue_entry);
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
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (g_queues.find(queue) == g_queues.end() ||
      queue->device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  return drain_queue_locked(&queue->state);
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

} // extern "C"
