#include "runtime_internal.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace lrrt_internal {

std::atomic<bool> g_initialized{false};

#if LRRT_ENABLE_HSA
RuntimeMutex g_devices_mutex;
std::condition_variable_any g_queue_state_changed;
std::condition_variable_any g_event_state_changed;
std::vector<DeviceState> g_devices;
hsa_agent_t g_host_agent{};
bool g_has_host_agent = false;

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

#endif

bool valid_device(lr_device_t device) {
#if LRRT_ENABLE_HSA
  std::lock_guard<RuntimeMutex> lock(g_devices_mutex);
  return device.index < g_devices.size();
#else
  return device.index == 0;
#endif
}

} // namespace lrrt_internal

using namespace lrrt_internal;

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
    std::lock_guard<RuntimeMutex> lock(g_devices_mutex);
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
    RuntimeLock lock(g_devices_mutex);
    wait_for_queue_submissions_locked(&lock);
    wait_for_queue_synchronizers_locked(&lock);
    wait_for_all_event_synchronizers_locked(&lock);
    wait_for_memory_operations_locked(&lock);
    lr_status_t drain_status = LR_SUCCESS;
    for (DeviceState &device : g_devices) {
      lr_status_t status = drain_device_locked(&device);
      if (status != LR_SUCCESS && drain_status == LR_SUCCESS) {
        drain_status = status;
      }
      release_device_queue_pools_locked(&device, &drain_status);
    }

    release_modules_locked();
    release_events_locked();
    destroy_all_queues_locked();
    release_memory_allocations_locked(&drain_status);
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
  std::lock_guard<RuntimeMutex> lock(g_devices_mutex);
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
    std::lock_guard<RuntimeMutex> lock(g_devices_mutex);
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
  std::lock_guard<RuntimeMutex> lock(g_devices_mutex);
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

lr_status_t lr_synchronize(lr_device_t device) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
#if LRRT_ENABLE_HSA
  RuntimeLock lock(g_devices_mutex);
  if (device.index >= g_devices.size() ||
      !g_devices[device.index].default_queue) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  return synchronize_device(&g_devices[device.index], &lock);
#else
  if (!valid_device(device)) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

} // extern "C"
