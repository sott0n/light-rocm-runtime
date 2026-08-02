#include "lrrt/lrrt.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if LRRT_ENABLE_HSA
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif

#if LRRT_ENABLE_HSA
struct KernargBuffer {
  void *ptr;
  size_t size;
};

struct PendingDispatch {
  hsa_signal_t completion_signal;
  KernargBuffer kernarg;
};

struct PendingBarrier {
  // Completion of the following dispatch proves the barrier was consumed.
  hsa_signal_t retirement_signal;
  std::vector<lr_event_t *> dependencies;
};

struct QueueState {
  hsa_queue_t *queue;
  std::vector<PendingDispatch> pending_dispatches;
  std::vector<PendingBarrier> pending_barriers;
  std::vector<lr_event_t *> pending_events;
  std::vector<hsa_signal_t> signal_pool;
  std::vector<KernargBuffer> kernarg_pool;
};
#endif

struct lr_event_t {
  lr_device_t device;
#if LRRT_ENABLE_HSA
  enum class Kind {
    None,
    Marker,
    AsyncCopy,
  };

  hsa_signal_t signal;
  Kind kind;
  bool pending;
  bool completed;
  std::unordered_set<QueueState *> dependency_queues;
  size_t dependency_count;
  uint64_t start_tick;
  uint64_t completion_tick;
  std::vector<lr_event_t *> dependencies;
  QueueState *recorded_queue;
  void *locked_host_ptr;
#endif
};

struct lr_queue_t {
  lr_device_t device;
  bool is_default;
#if LRRT_ENABLE_HSA
  QueueState state;
#endif
};

struct lr_module_t {
  lr_device_t device;
  std::vector<lr_kernel_t *> kernels;
#if LRRT_ENABLE_HSA
  hsa_code_object_reader_t reader;
  hsa_executable_t executable;
  hsa_loaded_code_object_t loaded_code_object;
#endif
};

struct lr_kernel_t {
  lr_module_t *module;
#if LRRT_ENABLE_HSA
  uint64_t object;
  uint32_t kernarg_size;
  uint32_t group_segment_size;
  uint32_t private_segment_size;
#endif
};

namespace {

std::atomic<bool> g_initialized{false};

#if LRRT_ENABLE_HSA
struct DeviceState {
  hsa_agent_t agent;
  std::string name;
  hsa_region_t global_region;
  hsa_region_t kernarg_region;
  bool has_global_region;
  bool has_kernarg_region;
  lr_queue_t *default_queue;
  std::vector<lr_queue_t *> queues;
  std::vector<lr_event_t *> pending_events;
  lr_memory_stats_t memory_stats;
};

struct SymbolSearch {
  const char *name;
  std::string descriptor_name;
  hsa_executable_symbol_t symbol;
  bool found;
};

struct AllocationInfo {
  uint32_t device_index;
  size_t size;
};

struct HostAllocationInfo {
  uint32_t device_index;
  size_t size;
  void *agent_ptr;
};

std::mutex g_devices_mutex;
std::vector<DeviceState> g_devices;
hsa_agent_t g_host_agent{};
bool g_has_host_agent = false;
std::unordered_map<void *, AllocationInfo> g_allocations;
std::unordered_map<void *, HostAllocationInfo> g_host_allocations;
std::unordered_set<lr_event_t *> g_events;
std::unordered_set<lr_queue_t *> g_queues;
std::unordered_set<lr_module_t *> g_modules;
std::unordered_set<lr_kernel_t *> g_kernels;

lr_status_t to_lr_status(hsa_status_t status) {
  return status == HSA_STATUS_SUCCESS ? LR_SUCCESS : LR_ERROR_RUNTIME;
}

hsa_status_t collect_agents(hsa_agent_t agent, void *data) {
  auto *devices = static_cast<std::vector<DeviceState> *>(data);
  hsa_device_type_t device_type = HSA_DEVICE_TYPE_CPU;
  hsa_status_t status =
      hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  if (device_type == HSA_DEVICE_TYPE_CPU) {
    if (!g_has_host_agent) {
      g_host_agent = agent;
      g_has_host_agent = true;
    }
  } else if (device_type == HSA_DEVICE_TYPE_GPU) {
    char name[64] = {};
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
    devices->push_back(DeviceState{
        agent, std::string(name), {}, {}, false, false, nullptr, {}, {}, {}});
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t select_device_regions(hsa_region_t region, void *data) {
  auto *device = static_cast<DeviceState *>(data);
  hsa_region_segment_t segment = HSA_REGION_SEGMENT_READONLY;
  hsa_status_t status =
      hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (segment != HSA_REGION_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  bool alloc_allowed = false;
  status = hsa_region_get_info(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
                               &alloc_allowed);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (!alloc_allowed) {
    return HSA_STATUS_SUCCESS;
  }

  uint32_t flags = 0;
  status = hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  if (!device->has_global_region &&
      (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) != 0) {
    device->global_region = region;
    device->has_global_region = true;
  }
  if (!device->has_kernarg_region &&
      (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) != 0) {
    device->kernarg_region = region;
    device->has_kernarg_region = true;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t populate_device_regions() {
  for (DeviceState &device : g_devices) {
    hsa_status_t status =
        hsa_agent_iterate_regions(device.agent, select_device_regions, &device);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }
  return HSA_STATUS_SUCCESS;
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

hsa_status_t find_kernel_symbol(hsa_executable_t, hsa_agent_t,
                                hsa_executable_symbol_t symbol, void *data) {
  auto *search = static_cast<SymbolSearch *>(data);
  hsa_symbol_kind_t kind = HSA_SYMBOL_KIND_VARIABLE;
  hsa_status_t status = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind);
  if (status != HSA_STATUS_SUCCESS || kind != HSA_SYMBOL_KIND_KERNEL) {
    return status;
  }

  uint32_t name_length = 0;
  status = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_length);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  std::string name(name_length, '\0');
  status = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data());
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  if (name == search->name || name == search->descriptor_name) {
    search->symbol = symbol;
    search->found = true;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t destroy_module_resources(lr_module_t *module) {
  for (lr_kernel_t *kernel : module->kernels) {
    g_kernels.erase(kernel);
    delete kernel;
  }
  module->kernels.clear();

  hsa_status_t executable_status = hsa_executable_destroy(module->executable);
  hsa_status_t reader_status = hsa_code_object_reader_destroy(module->reader);
  delete module;
  if (executable_status != HSA_STATUS_SUCCESS) {
    return executable_status;
  }
  return reader_status;
}

void reap_completed_barriers_locked(QueueState *queue);

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

bool valid_allocation(void *ptr, lr_device_t device, size_t size) {
  if (!ptr || size == 0) {
    return false;
  }
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  for (const auto &entry : g_allocations) {
    const AllocationInfo &info = entry.second;
    if (info.device_index != device.index) {
      continue;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(entry.first);
    if (address < base) {
      continue;
    }
    const uintptr_t offset = address - base;
    if (offset <= info.size && size <= info.size - offset) {
      return true;
    }
  }
  return false;
}

enum class HostPointerLookup {
  Unregistered,
  Valid,
  Invalid,
};

HostPointerLookup translate_host_pointer(const void *ptr, lr_device_t device,
                                         size_t size, const void **agent_ptr) {
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  for (const auto &entry : g_host_allocations) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(entry.first);
    const HostAllocationInfo &info = entry.second;
    if (address < base) {
      continue;
    }
    const uintptr_t offset = address - base;
    if (offset >= info.size) {
      continue;
    }
    if (info.device_index != device.index || size > info.size - offset) {
      return HostPointerLookup::Invalid;
    }
    const uintptr_t agent_base = reinterpret_cast<uintptr_t>(info.agent_ptr);
    *agent_ptr = reinterpret_cast<const void *>(agent_base + offset);
    return HostPointerLookup::Valid;
  }
  return HostPointerLookup::Unregistered;
}

void record_allocation(DeviceState *device, size_t size) {
  lr_memory_stats_t &stats = device->memory_stats;
  stats.live_bytes += size;
  stats.total_allocated_bytes += size;
  ++stats.allocation_count;
  if (stats.live_bytes > stats.peak_live_bytes) {
    stats.peak_live_bytes = stats.live_bytes;
  }
}

void record_host_allocation(DeviceState *device, size_t size) {
  lr_memory_stats_t &stats = device->memory_stats;
  stats.pinned_host_live_bytes += size;
  stats.pinned_host_total_allocated_bytes += size;
  ++stats.pinned_host_allocation_count;
  if (stats.pinned_host_live_bytes > stats.pinned_host_peak_live_bytes) {
    stats.pinned_host_peak_live_bytes = stats.pinned_host_live_bytes;
  }
}

void record_host_free(DeviceState *device, size_t size) {
  lr_memory_stats_t &stats = device->memory_stats;
  stats.pinned_host_live_bytes = size > stats.pinned_host_live_bytes
                                     ? 0
                                     : stats.pinned_host_live_bytes - size;
  stats.pinned_host_total_freed_bytes += size;
  ++stats.pinned_host_free_count;
}

void record_free(DeviceState *device, size_t size) {
  lr_memory_stats_t &stats = device->memory_stats;
  stats.live_bytes = size > stats.live_bytes ? 0 : stats.live_bytes - size;
  stats.total_freed_bytes += size;
  ++stats.free_count;
}

void record_memcpy(DeviceState *device, lr_memcpy_kind_t kind, size_t size) {
  lr_memory_stats_t &stats = device->memory_stats;
  if (kind == LR_MEMCPY_HOST_TO_DEVICE) {
    stats.h2d_copy_bytes += size;
  } else if (kind == LR_MEMCPY_DEVICE_TO_HOST) {
    stats.d2h_copy_bytes += size;
  } else if (kind == LR_MEMCPY_DEVICE_TO_DEVICE) {
    stats.d2d_copy_bytes += size;
  }
  ++stats.memcpy_count;
}
#endif

bool valid_device(lr_device_t device) {
#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  return device.index < g_devices.size();
#else
  return device.index == 0;
#endif
}

} // namespace

extern "C" {

const char *lr_status_string(lr_status_t status) {
  switch (status) {
  case LR_SUCCESS:
    return "success";
  case LR_ERROR_INVALID_ARGUMENT:
    return "invalid argument";
  case LR_ERROR_NOT_INITIALIZED:
    return "runtime is not initialized";
  case LR_ERROR_ALREADY_INITIALIZED:
    return "runtime is already initialized";
  case LR_ERROR_NOT_SUPPORTED:
    return "operation is not supported by this build";
  case LR_ERROR_RUNTIME:
    return "runtime error";
  }
  return "unknown status";
}

lr_status_t lr_init(void) {
  bool expected = false;
  if (!g_initialized.compare_exchange_strong(expected, true)) {
    return LR_ERROR_ALREADY_INITIALIZED;
  }

#if LRRT_ENABLE_HSA
  hsa_status_t status = hsa_init();
  if (status != HSA_STATUS_SUCCESS) {
    g_initialized.store(false);
    return to_lr_status(status);
  }

  {
    std::lock_guard<std::mutex> lock(g_devices_mutex);
    g_devices.clear();
    g_host_agent = {};
    g_has_host_agent = false;
    status = hsa_iterate_agents(collect_agents, &g_devices);
    if (status == HSA_STATUS_SUCCESS) {
      status = populate_device_regions();
    }
  }
  if (status != HSA_STATUS_SUCCESS) {
    hsa_shut_down();
    g_initialized.store(false);
    return to_lr_status(status);
  }
#endif

  return LR_SUCCESS;
}

lr_status_t lr_shutdown(void) {
  bool expected = true;
  if (!g_initialized.compare_exchange_strong(expected, false)) {
    return LR_ERROR_NOT_INITIALIZED;
  }

#if LRRT_ENABLE_HSA
  {
    std::lock_guard<std::mutex> lock(g_devices_mutex);
    lr_status_t drain_status = LR_SUCCESS;
    for (DeviceState &device : g_devices) {
      lr_status_t status = drain_device_locked(&device);
      if (status != LR_SUCCESS && drain_status == LR_SUCCESS) {
        drain_status = status;
      }
      for (lr_queue_t *queue : device.queues) {
        release_queue_pools_locked(&queue->state, &drain_status);
      }
    }

    for (lr_module_t *module : g_modules) {
      destroy_module_resources(module);
    }
    g_modules.clear();
    g_kernels.clear();

    for (lr_event_t *event : g_events) {
      hsa_signal_destroy(event->signal);
      delete event;
    }
    g_events.clear();

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
    for (const auto &allocation : g_allocations) {
      hsa_memory_free(allocation.first);
    }
    g_allocations.clear();
    for (const auto &allocation : g_host_allocations) {
      hsa_status_t status = hsa_amd_memory_unlock(allocation.first);
      if (status == HSA_STATUS_SUCCESS) {
        std::free(allocation.first);
      } else if (drain_status == LR_SUCCESS) {
        drain_status = to_lr_status(status);
      }
    }
    g_host_allocations.clear();
    g_devices.clear();
    g_host_agent = {};
    g_has_host_agent = false;

    if (drain_status != LR_SUCCESS) {
      hsa_shut_down();
      return drain_status;
    }
  }

  hsa_status_t status = hsa_shut_down();
  if (status != HSA_STATUS_SUCCESS) {
    g_initialized.store(true);
    return to_lr_status(status);
  }
#endif

  return LR_SUCCESS;
}

lr_status_t lr_device_count(uint32_t *count) {
  if (!count) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }

  *count = 0;
#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  *count = static_cast<uint32_t>(g_devices.size());
  return LR_SUCCESS;
#else
  return LR_SUCCESS;
#endif
}

lr_status_t lr_device_open(uint32_t index, lr_device_t *device) {
  if (!device) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }

#if LRRT_ENABLE_HSA
  {
    std::lock_guard<std::mutex> lock(g_devices_mutex);
    if (index >= g_devices.size()) {
      return LR_ERROR_INVALID_ARGUMENT;
    }
    if (!g_devices[index].default_queue) {
      lr_device_t opened_device = {index};
      hsa_status_t status = create_queue(opened_device, &g_devices[index], true,
                                         &g_devices[index].default_queue);
      if (status != HSA_STATUS_SUCCESS) {
        return to_lr_status(status);
      }
    }
  }
#else
  if (index != 0) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  return LR_ERROR_NOT_SUPPORTED;
#endif

  device->index = index;
  return LR_SUCCESS;
}

lr_status_t lr_device_name(lr_device_t device, char *name, size_t name_size) {
  if (!name || name_size == 0) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }

#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  const std::string &device_name = g_devices[device.index].name;
  if (device_name.size() + 1 > name_size) {
    name[0] = '\0';
    return LR_ERROR_INVALID_ARGUMENT;
  }
  std::strncpy(name, device_name.c_str(), name_size);
  name[name_size - 1] = '\0';
  return LR_SUCCESS;
#else
  (void)device;
  name[0] = '\0';
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_get_memory_stats(lr_device_t device, lr_memory_stats_t *stats) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!valid_device(device) || !stats) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  *stats = g_devices[device.index].memory_stats;
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_reset_memory_stats(lr_device_t device) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!valid_device(device)) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  lr_memory_stats_t reset{};
  reset.live_bytes = g_devices[device.index].memory_stats.live_bytes;
  reset.peak_live_bytes = reset.live_bytes;
  reset.pinned_host_live_bytes =
      g_devices[device.index].memory_stats.pinned_host_live_bytes;
  reset.pinned_host_peak_live_bytes = reset.pinned_host_live_bytes;
  g_devices[device.index].memory_stats = reset;
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

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
  if (!queue || g_queues.find(queue) == g_queues.end() ||
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

lr_status_t lr_malloc(lr_device_t device, size_t size, void **ptr) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!valid_device(device) || size == 0 || !ptr) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  *ptr = nullptr;
#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  DeviceState &state = g_devices[device.index];
  if (!state.has_global_region) {
    return LR_ERROR_NOT_SUPPORTED;
  }

  hsa_status_t status = hsa_memory_allocate(state.global_region, size, ptr);
  if (status != HSA_STATUS_SUCCESS) {
    *ptr = nullptr;
    return to_lr_status(status);
  }

  status = hsa_memory_assign_agent(*ptr, state.agent, HSA_ACCESS_PERMISSION_RW);
  if (status != HSA_STATUS_SUCCESS) {
    hsa_memory_free(*ptr);
    *ptr = nullptr;
    return to_lr_status(status);
  }

  g_allocations[*ptr] = AllocationInfo{device.index, size};
  record_allocation(&state, size);
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_free(lr_device_t device, void *ptr) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!valid_device(device) || !ptr) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  auto allocation = g_allocations.find(ptr);
  if (allocation == g_allocations.end() ||
      allocation->second.device_index != device.index) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  lr_status_t drain_status = drain_device_locked(&g_devices[device.index]);
  if (drain_status != LR_SUCCESS) {
    return drain_status;
  }

  const size_t size = allocation->second.size;
  hsa_status_t status = hsa_memory_free(ptr);
  if (status != HSA_STATUS_SUCCESS) {
    return to_lr_status(status);
  }
  record_free(&g_devices[device.index], size);
  g_allocations.erase(allocation);
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_host_malloc(lr_device_t device, size_t size, void **ptr) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!valid_device(device) || size == 0 || !ptr) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  *ptr = nullptr;
#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (!g_has_host_agent) {
    return LR_ERROR_NOT_SUPPORTED;
  }

  void *host_ptr = std::malloc(size);
  if (!host_ptr) {
    return LR_ERROR_RUNTIME;
  }

  DeviceState &state = g_devices[device.index];
  void *agent_ptr = nullptr;
  hsa_status_t status =
      hsa_amd_memory_lock(host_ptr, size, &state.agent, 1, &agent_ptr);
  if (status != HSA_STATUS_SUCCESS) {
    std::free(host_ptr);
    return to_lr_status(status);
  }

  g_host_allocations[host_ptr] =
      HostAllocationInfo{device.index, size, agent_ptr};
  record_host_allocation(&state, size);
  *ptr = host_ptr;
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_host_free(lr_device_t device, void *ptr) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!valid_device(device) || !ptr) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  auto allocation = g_host_allocations.find(ptr);
  if (allocation == g_host_allocations.end() ||
      allocation->second.device_index != device.index) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  DeviceState &state = g_devices[device.index];
  lr_status_t drain_status = drain_device_locked(&state);
  if (drain_status != LR_SUCCESS) {
    return drain_status;
  }

  hsa_status_t status = hsa_amd_memory_unlock(ptr);
  if (status != HSA_STATUS_SUCCESS) {
    return to_lr_status(status);
  }

  const size_t size = allocation->second.size;
  std::free(ptr);
  g_host_allocations.erase(allocation);
  record_host_free(&state, size);
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_memcpy(lr_device_t device, void *dst, const void *src,
                      size_t size, lr_memcpy_kind_t kind) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!valid_device(device) || !dst || !src || size == 0) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (kind != LR_MEMCPY_HOST_TO_DEVICE && kind != LR_MEMCPY_DEVICE_TO_HOST &&
      kind != LR_MEMCPY_DEVICE_TO_DEVICE) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  if (kind == LR_MEMCPY_HOST_TO_DEVICE) {
    if (!valid_allocation(dst, device, size)) {
      return LR_ERROR_INVALID_ARGUMENT;
    }
  } else if (kind == LR_MEMCPY_DEVICE_TO_HOST) {
    if (!valid_allocation(const_cast<void *>(src), device, size)) {
      return LR_ERROR_INVALID_ARGUMENT;
    }
  } else {
    if (!valid_allocation(dst, device, size) ||
        !valid_allocation(const_cast<void *>(src), device, size)) {
      return LR_ERROR_INVALID_ARGUMENT;
    }
  }

  lr_status_t drain_status = drain_device_locked(&g_devices[device.index]);
  if (drain_status != LR_SUCCESS) {
    return drain_status;
  }

  hsa_status_t status = hsa_memory_copy(dst, src, size);
  if (status == HSA_STATUS_SUCCESS) {
    record_memcpy(&g_devices[device.index], kind, size);
  }
  return to_lr_status(status);
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

static lr_status_t memcpy_async_impl(lr_device_t device, void *dst,
                                     const void *src, size_t size,
                                     lr_memcpy_kind_t kind, lr_event_t *event,
                                     lr_event_t *const *explicit_dependencies,
                                     size_t dependency_count,
                                     bool use_implicit_dependencies) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!valid_device(device) || !dst || !src || size == 0 || !event) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (kind != LR_MEMCPY_HOST_TO_DEVICE && kind != LR_MEMCPY_DEVICE_TO_HOST &&
      kind != LR_MEMCPY_DEVICE_TO_DEVICE) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (device.index >= g_devices.size() ||
      g_events.find(event) == g_events.end() ||
      event->device.index != device.index) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  if (kind == LR_MEMCPY_HOST_TO_DEVICE) {
    if (!valid_allocation(dst, device, size)) {
      return LR_ERROR_INVALID_ARGUMENT;
    }
  } else if (kind == LR_MEMCPY_DEVICE_TO_HOST) {
    if (!valid_allocation(const_cast<void *>(src), device, size)) {
      return LR_ERROR_INVALID_ARGUMENT;
    }
  } else {
    if (!valid_allocation(dst, device, size) ||
        !valid_allocation(const_cast<void *>(src), device, size)) {
      return LR_ERROR_INVALID_ARGUMENT;
    }
  }

  DeviceState &state = g_devices[device.index];
  hsa_agent_t null_agent{};
  hsa_agent_t dst_agent = null_agent;
  hsa_agent_t src_agent = null_agent;
  void *copy_dst = dst;
  const void *copy_src = src;
  void *pageable_host_ptr = nullptr;
  if (kind == LR_MEMCPY_HOST_TO_DEVICE) {
    dst_agent = state.agent;
    const void *mapped_src = nullptr;
    HostPointerLookup lookup =
        translate_host_pointer(src, device, size, &mapped_src);
    if (lookup == HostPointerLookup::Invalid) {
      return LR_ERROR_INVALID_ARGUMENT;
    }
    if (lookup == HostPointerLookup::Valid) {
      copy_src = mapped_src;
      src_agent = g_host_agent;
    } else {
      pageable_host_ptr = const_cast<void *>(src);
    }
  } else if (kind == LR_MEMCPY_DEVICE_TO_HOST) {
    src_agent = state.agent;
    const void *mapped_dst = nullptr;
    HostPointerLookup lookup =
        translate_host_pointer(dst, device, size, &mapped_dst);
    if (lookup == HostPointerLookup::Invalid) {
      return LR_ERROR_INVALID_ARGUMENT;
    }
    if (lookup == HostPointerLookup::Valid) {
      copy_dst = const_cast<void *>(mapped_dst);
      dst_agent = g_host_agent;
    } else {
      pageable_host_ptr = dst;
    }
  } else {
    dst_agent = state.agent;
    src_agent = state.agent;
  }

  lr_status_t wait_status = event_wait_locked(event);
  if (wait_status != LR_SUCCESS) {
    return wait_status;
  }

  lr_status_t consumer_status = wait_for_event_consumers_locked(&state, event);
  if (consumer_status != LR_SUCCESS) {
    return consumer_status;
  }

  std::vector<lr_event_t *> event_dependencies;
  if (!use_implicit_dependencies) {
    lr_status_t dependency_status = collect_event_dependencies_locked(
        device, explicit_dependencies, dependency_count, event,
        &event_dependencies);
    if (dependency_status != LR_SUCCESS) {
      return dependency_status;
    }
  }

  hsa_status_t status = hsa_amd_profiling_async_copy_enable(true);
  if (status != HSA_STATUS_SUCCESS) {
    return to_lr_status(status);
  }

  hsa_signal_store_relaxed(event->signal, 1);
  event->kind = lr_event_t::Kind::AsyncCopy;
  event->completed = false;
  event->dependency_queues.clear();
  event->start_tick = 0;
  event->completion_tick = 0;
  event->recorded_queue = nullptr;
  event->locked_host_ptr = nullptr;

  if (pageable_host_ptr) {
    void *mapped_host_ptr = nullptr;
    status = hsa_amd_memory_lock(pageable_host_ptr, size, &state.agent, 1,
                                 &mapped_host_ptr);
    if (status != HSA_STATUS_SUCCESS) {
      event->kind = lr_event_t::Kind::None;
      return to_lr_status(status);
    }
    if (kind == LR_MEMCPY_HOST_TO_DEVICE) {
      copy_src = mapped_host_ptr;
      src_agent = g_host_agent;
    } else {
      copy_dst = mapped_host_ptr;
      dst_agent = g_host_agent;
    }
    event->locked_host_ptr = pageable_host_ptr;
  }

  std::vector<hsa_signal_t> dependencies;
  if (use_implicit_dependencies) {
    QueueState &default_queue = state.default_queue->state;
    lr_status_t reap_status = reap_completed_queue_work_locked(&default_queue);
    if (reap_status != LR_SUCCESS) {
      if (event->locked_host_ptr) {
        hsa_amd_memory_unlock(event->locked_host_ptr);
        event->locked_host_ptr = nullptr;
      }
      event->kind = lr_event_t::Kind::None;
      return reap_status;
    }
    dependencies.reserve(default_queue.pending_dispatches.size());
    for (const PendingDispatch &dispatch : default_queue.pending_dispatches) {
      dependencies.push_back(dispatch.completion_signal);
    }
  } else {
    dependencies.reserve(event_dependencies.size());
    for (lr_event_t *dependency : event_dependencies) {
      dependencies.push_back(dependency->signal);
    }
  }
  status =
      hsa_amd_memory_async_copy(copy_dst, dst_agent, copy_src, src_agent, size,
                                static_cast<uint32_t>(dependencies.size()),
                                dependencies.data(), event->signal);
  if (status != HSA_STATUS_SUCCESS) {
    if (event->locked_host_ptr) {
      hsa_amd_memory_unlock(event->locked_host_ptr);
      event->locked_host_ptr = nullptr;
    }
    event->kind = lr_event_t::Kind::None;
    return to_lr_status(status);
  }

  record_memcpy(&state, kind, size);
  event->dependencies = std::move(event_dependencies);
  for (lr_event_t *dependency : event->dependencies) {
    ++dependency->dependency_count;
  }
  event->pending = true;
  state.pending_events.push_back(event);
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_memcpy_async(lr_device_t device, void *dst, const void *src,
                            size_t size, lr_memcpy_kind_t kind,
                            lr_event_t *event) {
  return memcpy_async_impl(device, dst, src, size, kind, event, nullptr, 0,
                           true);
}

lr_status_t lr_memcpy_async_with_dependencies(lr_device_t device, void *dst,
                                              const void *src, size_t size,
                                              lr_memcpy_kind_t kind,
                                              lr_event_t *event,
                                              lr_event_t *const *dependencies,
                                              size_t dependency_count) {
  return memcpy_async_impl(device, dst, src, size, kind, event, dependencies,
                           dependency_count, false);
}

lr_status_t lr_module_load_hsaco(lr_device_t device, const void *image,
                                 size_t image_size, lr_module_t **module) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!valid_device(device) || !image || image_size == 0 || !module) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  *module = nullptr;
#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  DeviceState &state = g_devices[device.index];
  hsa_profile_t profile = HSA_PROFILE_FULL;
  hsa_status_t status =
      hsa_agent_get_info(state.agent, HSA_AGENT_INFO_PROFILE, &profile);
  if (status != HSA_STATUS_SUCCESS) {
    return to_lr_status(status);
  }

  auto *loaded_module = new lr_module_t{};
  loaded_module->device = device;

  status = hsa_code_object_reader_create_from_memory(image, image_size,
                                                     &loaded_module->reader);
  if (status != HSA_STATUS_SUCCESS) {
    delete loaded_module;
    return to_lr_status(status);
  }

  status =
      hsa_executable_create_alt(profile, HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR,
                                nullptr, &loaded_module->executable);
  if (status != HSA_STATUS_SUCCESS) {
    hsa_code_object_reader_destroy(loaded_module->reader);
    delete loaded_module;
    return to_lr_status(status);
  }

  status = hsa_executable_load_agent_code_object(
      loaded_module->executable, state.agent, loaded_module->reader, nullptr,
      &loaded_module->loaded_code_object);
  if (status != HSA_STATUS_SUCCESS) {
    hsa_executable_destroy(loaded_module->executable);
    hsa_code_object_reader_destroy(loaded_module->reader);
    delete loaded_module;
    return to_lr_status(status);
  }

  status = hsa_executable_freeze(loaded_module->executable, nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    hsa_executable_destroy(loaded_module->executable);
    hsa_code_object_reader_destroy(loaded_module->reader);
    delete loaded_module;
    return to_lr_status(status);
  }

  *module = loaded_module;
  g_modules.insert(loaded_module);
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_module_destroy(lr_module_t *module) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!module) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  auto module_entry = g_modules.find(module);
  if (module_entry == g_modules.end()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  if (module->device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  lr_status_t drain_status =
      drain_device_locked(&g_devices[module->device.index]);
  if (drain_status != LR_SUCCESS) {
    return drain_status;
  }

  g_modules.erase(module_entry);
  return to_lr_status(destroy_module_resources(module));
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_kernel_get(lr_module_t *module, const char *name,
                          lr_kernel_t **kernel) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!module || !name || !kernel) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  *kernel = nullptr;
#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (g_modules.find(module) == g_modules.end()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (module->device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  SymbolSearch search{name, std::string(name) + ".kd", {}, false};
  hsa_status_t status = hsa_executable_iterate_agent_symbols(
      module->executable, g_devices[module->device.index].agent,
      find_kernel_symbol, &search);
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    return to_lr_status(status);
  }
  if (!search.found) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  auto *loaded_kernel = new lr_kernel_t{};
  loaded_kernel->module = module;

  status = hsa_executable_symbol_get_info(
      search.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
      &loaded_kernel->object);
  if (status != HSA_STATUS_SUCCESS) {
    delete loaded_kernel;
    return to_lr_status(status);
  }
  status = hsa_executable_symbol_get_info(
      search.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
      &loaded_kernel->kernarg_size);
  if (status != HSA_STATUS_SUCCESS) {
    delete loaded_kernel;
    return to_lr_status(status);
  }
  status = hsa_executable_symbol_get_info(
      search.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
      &loaded_kernel->group_segment_size);
  if (status != HSA_STATUS_SUCCESS) {
    delete loaded_kernel;
    return to_lr_status(status);
  }
  status = hsa_executable_symbol_get_info(
      search.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
      &loaded_kernel->private_segment_size);
  if (status != HSA_STATUS_SUCCESS) {
    delete loaded_kernel;
    return to_lr_status(status);
  }

  module->kernels.push_back(loaded_kernel);
  g_kernels.insert(loaded_kernel);
  *kernel = loaded_kernel;
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

static lr_status_t
launch_impl(lr_kernel_t *kernel, const lr_launch_config_t *config,
            const void *args, size_t args_size, lr_queue_t *execution_queue,
            bool use_default_queue, lr_event_t *const *explicit_dependencies,
            size_t dependency_count, bool use_implicit_dependencies) {
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
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (g_kernels.find(kernel) == g_kernels.end() ||
      g_modules.find(kernel->module) == g_modules.end()) {
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
  if (!execution_queue || g_queues.find(execution_queue) == g_queues.end() ||
      execution_queue->device.index != device.index ||
      !state.has_kernarg_region || args_size > kernel->kernarg_size) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  QueueState &queue = execution_queue->state;
  std::vector<lr_event_t *> event_dependencies;
  if (!use_implicit_dependencies) {
    lr_status_t dependency_status = collect_event_dependencies_locked(
        device, explicit_dependencies, dependency_count, nullptr,
        &event_dependencies);
    if (dependency_status != LR_SUCCESS) {
      return dependency_status;
    }
  }
  KernargBuffer kernarg{};
  hsa_status_t status = acquire_kernarg_locked(&queue, state.kernarg_region,
                                               kernel->kernarg_size, &kernarg);
  if (status != HSA_STATUS_SUCCESS) {
    return to_lr_status(status);
  }
  std::memset(kernarg.ptr, 0, kernel->kernarg_size);
  std::memcpy(kernarg.ptr, args, args_size);

  hsa_signal_t signal{};
  status = acquire_signal_locked(&queue, &signal);
  if (status != HSA_STATUS_SUCCESS) {
    queue.kernarg_pool.push_back(kernarg);
    return to_lr_status(status);
  }
  const std::vector<lr_event_t *> *dependencies =
      use_implicit_dependencies ? nullptr : &event_dependencies;
  lr_status_t dependency_status =
      enqueue_event_dependencies_locked(&state, &queue, signal, dependencies);
  if (dependency_status != LR_SUCCESS) {
    queue.signal_pool.push_back(signal);
    queue.kernarg_pool.push_back(kernarg);
    return dependency_status;
  }
  // Keep packets on the same lrrt queue completion-ordered. Several executor
  // pipelines pass one kernel's output directly to the next kernel.
  const bool wait_for_dependencies =
      !queue.pending_dispatches.empty() ||
      (use_implicit_dependencies ? !queue.pending_barriers.empty()
                                 : !event_dependencies.empty());

  const uint64_t index = hsa_queue_add_write_index_scacq_screl(queue.queue, 1);
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
  uint16_t header = wait_for_dependencies
                        ? barrier_packet_header(HSA_PACKET_TYPE_KERNEL_DISPATCH)
                        : packet_header(HSA_PACKET_TYPE_KERNEL_DISPATCH);
  publish_packet_header(&packet->header, header);

  hsa_signal_store_screlease(queue.queue->doorbell_signal, index);
  queue.pending_dispatches.push_back(PendingDispatch{signal, kernarg});
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

lr_status_t lr_synchronize(lr_device_t device) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (device.index >= g_devices.size() ||
      !g_devices[device.index].default_queue) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  return drain_device_locked(&g_devices[device.index]);
#else
  if (!valid_device(device)) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

} // extern "C"
