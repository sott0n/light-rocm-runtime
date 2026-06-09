#include "lrrt/lrrt.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#if LRRT_ENABLE_HSA
#include <hsa/hsa.h>
#endif

namespace {

std::atomic<bool> g_initialized{false};

#if LRRT_ENABLE_HSA
struct DeviceState {
  hsa_agent_t agent;
  hsa_region_t global_region;
  bool has_global_region;
  hsa_queue_t *queue;
};

std::mutex g_devices_mutex;
std::vector<DeviceState> g_devices;
std::unordered_map<void *, uint32_t> g_allocations;

lr_status_t to_lr_status(hsa_status_t status) {
  return status == HSA_STATUS_SUCCESS ? LR_SUCCESS : LR_ERROR_RUNTIME;
}

hsa_status_t collect_gpu_agents(hsa_agent_t agent, void *data) {
  auto *devices = static_cast<std::vector<DeviceState> *>(data);
  hsa_device_type_t device_type = HSA_DEVICE_TYPE_CPU;
  hsa_status_t status =
      hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  if (device_type == HSA_DEVICE_TYPE_GPU) {
    devices->push_back(DeviceState{agent, {}, false, nullptr});
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t select_global_region(hsa_region_t region, void *data) {
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
  if ((flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) == 0) {
    return HSA_STATUS_SUCCESS;
  }

  device->global_region = region;
  device->has_global_region = true;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t populate_device_regions() {
  for (DeviceState &device : g_devices) {
    hsa_status_t status =
        hsa_agent_iterate_regions(device.agent, select_global_region, &device);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t create_default_queue(DeviceState *device) {
  uint32_t max_queue_size = 0;
  hsa_status_t status =
      hsa_agent_get_info(device->agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                         &max_queue_size);
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

  return hsa_queue_create(device->agent, queue_size, HSA_QUEUE_TYPE_MULTI,
                          nullptr, nullptr, UINT32_MAX, UINT32_MAX,
                          &device->queue);
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

}  // namespace

struct lr_module_t {
  lr_device_t device;
};

struct lr_kernel_t {
  lr_module_t *module;
};

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
    status = hsa_iterate_agents(collect_gpu_agents, &g_devices);
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
    for (DeviceState &device : g_devices) {
      if (device.queue) {
        hsa_queue_destroy(device.queue);
        device.queue = nullptr;
      }
    }
    for (const auto &allocation : g_allocations) {
      hsa_memory_free(allocation.first);
    }
    g_allocations.clear();
    g_devices.clear();
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
    if (!g_devices[index].queue) {
      hsa_status_t status = create_default_queue(&g_devices[index]);
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

  status = hsa_memory_assign_agent(*ptr, state.agent,
                                   HSA_ACCESS_PERMISSION_RW);
  if (status != HSA_STATUS_SUCCESS) {
    hsa_memory_free(*ptr);
    *ptr = nullptr;
    return to_lr_status(status);
  }

  g_allocations[*ptr] = device.index;
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
      allocation->second != device.index) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  hsa_status_t status = hsa_memory_free(ptr);
  if (status != HSA_STATUS_SUCCESS) {
    return to_lr_status(status);
  }
  g_allocations.erase(allocation);
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
    auto allocation = g_allocations.find(dst);
    if (allocation == g_allocations.end() ||
        allocation->second != device.index) {
      return LR_ERROR_INVALID_ARGUMENT;
    }
  } else if (kind == LR_MEMCPY_DEVICE_TO_HOST) {
    auto allocation = g_allocations.find(const_cast<void *>(src));
    if (allocation == g_allocations.end() ||
        allocation->second != device.index) {
      return LR_ERROR_INVALID_ARGUMENT;
    }
  } else {
    auto dst_allocation = g_allocations.find(dst);
    auto src_allocation = g_allocations.find(const_cast<void *>(src));
    if (dst_allocation == g_allocations.end() ||
        src_allocation == g_allocations.end() ||
        dst_allocation->second != device.index ||
        src_allocation->second != device.index) {
      return LR_ERROR_INVALID_ARGUMENT;
    }
  }

  return to_lr_status(hsa_memory_copy(dst, src, size));
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
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
  return LR_ERROR_NOT_SUPPORTED;
}

lr_status_t lr_module_destroy(lr_module_t *module) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!module) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  return LR_ERROR_NOT_SUPPORTED;
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
  return LR_ERROR_NOT_SUPPORTED;
}

lr_status_t lr_launch(lr_kernel_t *kernel, const lr_launch_config_t *config,
                      const void *args, size_t args_size) {
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

  return LR_ERROR_NOT_SUPPORTED;
}

lr_status_t lr_synchronize(lr_device_t device) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (device.index >= g_devices.size() || !g_devices[device.index].queue) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  return LR_SUCCESS;
#else
  if (!valid_device(device)) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

}  // extern "C"
