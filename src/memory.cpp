#include "runtime_internal.hpp"

#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lrrt_internal {

#if LRRT_ENABLE_HSA
namespace {

struct AllocationInfo {
  uint32_t device_index;
  size_t size;
};

struct HostAllocationInfo {
  uint32_t device_index;
  size_t size;
  void *agent_ptr;
};

std::unordered_map<void *, AllocationInfo> g_allocations;
std::unordered_map<void *, HostAllocationInfo> g_host_allocations;

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

} // namespace

void release_memory_allocations_locked(lr_status_t *result) {
  for (const auto &allocation : g_allocations) {
    hsa_memory_free(allocation.first);
  }
  g_allocations.clear();
  for (const auto &allocation : g_host_allocations) {
    hsa_status_t status = hsa_amd_memory_unlock(allocation.first);
    if (status == HSA_STATUS_SUCCESS) {
      std::free(allocation.first);
    } else if (*result == LR_SUCCESS) {
      *result = to_lr_status(status);
    }
  }
  g_host_allocations.clear();
}
#endif

} // namespace lrrt_internal

using namespace lrrt_internal;

extern "C" {

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
  std::unique_lock<std::mutex> lock(g_devices_mutex);
  if (device.index >= g_devices.size() || !valid_event_locked(event) ||
      event->device.index != device.index || event->destroying) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  wait_for_event_synchronizers_locked(&lock, event);
  if (!valid_event_locked(event) || event->destroying) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  DeviceState &state = g_devices[device.index];
  lr_status_t wait_status = event_wait_locked(event);
  if (wait_status != LR_SUCCESS) {
    return wait_status;
  }

  lr_status_t consumer_status =
      wait_for_event_consumers_locked(&lock, &state, event);
  if (consumer_status != LR_SUCCESS) {
    return consumer_status;
  }
  if (!valid_event_locked(event) || event->destroying) {
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

} // extern "C"
