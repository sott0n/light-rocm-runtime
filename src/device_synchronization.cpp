#include "runtime_internal.hpp"

#include <cstdint>
#include <mutex>
#include <vector>

namespace lrrt_internal {

#if LRRT_ENABLE_HSA
namespace {

struct DeviceQueueSynchronization {
  QueueState *queue;
  hsa_signal_t signal;
  hsa_signal_value_t wait_value;
};

struct DeviceEventSynchronization {
  lr_event_t *event;
  hsa_signal_t signal;
  QueueState *recorded_queue;
  hsa_signal_value_t wait_value;
};

struct DeviceSynchronization {
  std::vector<DeviceQueueSynchronization> queues;
  std::vector<DeviceEventSynchronization> events;
  lr_status_t preparation_status;
};

DeviceSynchronization prepare_device_synchronization_locked(
    DeviceState *device, std::unique_lock<std::mutex> *devices_lock) {
  DeviceSynchronization synchronization{};
  synchronization.preparation_status = LR_SUCCESS;
  synchronization.events.reserve(device->pending_events.size());
  for (lr_event_t *event : device->pending_events) {
    if (!event->pending) {
      continue;
    }
    QueueState *recorded_queue = event->recorded_queue;
    ++event->active_synchronizers;
    if (recorded_queue) {
      ++recorded_queue->active_synchronizers;
    }
    synchronization.events.push_back(
        DeviceEventSynchronization{event, event->signal, recorded_queue, 1});
  }

  synchronization.queues.reserve(device->queues.size());
  std::vector<QueueState *> queues;
  queues.reserve(device->queues.size());
  for (lr_queue_t *queue : device->queues) {
    ++queue->state.active_synchronizers;
    queues.push_back(&queue->state);
  }
  for (QueueState *queue : queues) {
    hsa_signal_t signal{};
    lr_status_t status =
        enqueue_queue_synchronization_locked(queue, &signal, devices_lock);
    if (status != LR_SUCCESS) {
      synchronization.preparation_status = status;
      break;
    }
    if (signal.handle != 0) {
      synchronization.queues.push_back(
          DeviceQueueSynchronization{queue, signal, 1});
    }
  }
  for (QueueState *queue : queues) {
    --queue->active_synchronizers;
  }
  g_queue_state_changed.notify_all();
  return synchronization;
}

void wait_for_device_synchronization(DeviceSynchronization *synchronization) {
  for (DeviceEventSynchronization &event : synchronization->events) {
    event.wait_value =
        hsa_signal_wait_scacquire(event.signal, HSA_SIGNAL_CONDITION_LT, 1,
                                  UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  }
  for (DeviceQueueSynchronization &queue : synchronization->queues) {
    queue.wait_value =
        hsa_signal_wait_scacquire(queue.signal, HSA_SIGNAL_CONDITION_LT, 1,
                                  UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  }
}

lr_status_t
finish_device_synchronization_locked(DeviceSynchronization *synchronization) {
  lr_status_t result = synchronization->preparation_status;
  for (DeviceEventSynchronization &event : synchronization->events) {
    lr_status_t status =
        finish_event_wait_locked(event.event, event.wait_value);
    if (result == LR_SUCCESS && status != LR_SUCCESS) {
      result = status;
    }
    --event.event->active_synchronizers;
    g_event_state_changed.notify_all();
    if (event.recorded_queue) {
      --event.recorded_queue->active_synchronizers;
      g_queue_state_changed.notify_all();
    }
  }
  for (DeviceQueueSynchronization &queue : synchronization->queues) {
    lr_status_t status = finish_queue_synchronization_locked(
        queue.queue, queue.signal, queue.wait_value);
    if (result == LR_SUCCESS && status != LR_SUCCESS) {
      result = status;
    }
  }
  return result;
}

} // namespace

lr_status_t synchronize_device(DeviceState *device,
                               std::unique_lock<std::mutex> *devices_lock) {
  DeviceSynchronization synchronization =
      prepare_device_synchronization_locked(device, devices_lock);

  devices_lock->unlock();
  wait_for_device_synchronization(&synchronization);
  devices_lock->lock();

  return finish_device_synchronization_locked(&synchronization);
}
#endif

} // namespace lrrt_internal
