#include "lrrt/lrrt.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if LRRT_ENABLE_HSA
#include <hsa/hsa.h>
#endif

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
struct KernargBuffer {
  void *ptr;
  size_t size;
};

struct PendingDispatch {
  hsa_signal_t completion_signal;
  KernargBuffer kernarg;
};

struct DeviceState {
  hsa_agent_t agent;
  hsa_region_t global_region;
  hsa_region_t kernarg_region;
  bool has_global_region;
  bool has_kernarg_region;
  hsa_queue_t *queue;
  std::vector<PendingDispatch> pending_dispatches;
  std::vector<hsa_signal_t> signal_pool;
  std::vector<KernargBuffer> kernarg_pool;
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

std::mutex g_devices_mutex;
std::vector<DeviceState> g_devices;
std::unordered_map<void *, AllocationInfo> g_allocations;
std::unordered_set<lr_module_t *> g_modules;
std::unordered_set<lr_kernel_t *> g_kernels;

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
    devices->push_back(DeviceState{agent, {}, {}, false, false, nullptr});
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

uint16_t packet_header(hsa_packet_type_t type) {
  return static_cast<uint16_t>(
      (type << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
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

lr_status_t drain_device_locked(DeviceState *device) {
  lr_status_t result = LR_SUCCESS;
  for (const PendingDispatch &dispatch : device->pending_dispatches) {
    hsa_signal_value_t value = hsa_signal_wait_scacquire(
        dispatch.completion_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
        HSA_WAIT_STATE_BLOCKED);
    if (value != 0 && result == LR_SUCCESS) {
      result = LR_ERROR_RUNTIME;
    }
    hsa_signal_store_relaxed(dispatch.completion_signal, 1);
    device->signal_pool.push_back(dispatch.completion_signal);
    device->kernarg_pool.push_back(dispatch.kernarg);
  }
  device->pending_dispatches.clear();
  return result;
}

void release_device_pools_locked(DeviceState *device, lr_status_t *result) {
  for (hsa_signal_t signal : device->signal_pool) {
    hsa_status_t status = hsa_signal_destroy(signal);
    if (status != HSA_STATUS_SUCCESS && *result == LR_SUCCESS) {
      *result = to_lr_status(status);
    }
  }
  device->signal_pool.clear();

  for (KernargBuffer kernarg : device->kernarg_pool) {
    hsa_status_t status = hsa_memory_free(kernarg.ptr);
    if (status != HSA_STATUS_SUCCESS && *result == LR_SUCCESS) {
      *result = to_lr_status(status);
    }
  }
  device->kernarg_pool.clear();
}

hsa_status_t acquire_signal_locked(DeviceState *device, hsa_signal_t *signal) {
  if (!device->signal_pool.empty()) {
    *signal = device->signal_pool.back();
    device->signal_pool.pop_back();
    hsa_signal_store_relaxed(*signal, 1);
    return HSA_STATUS_SUCCESS;
  }
  return hsa_signal_create(1, 0, nullptr, signal);
}

hsa_status_t acquire_kernarg_locked(DeviceState *device, size_t size,
                                    KernargBuffer *kernarg) {
  for (size_t i = 0; i < device->kernarg_pool.size(); ++i) {
    if (device->kernarg_pool[i].size >= size) {
      *kernarg = device->kernarg_pool[i];
      device->kernarg_pool[i] = device->kernarg_pool.back();
      device->kernarg_pool.pop_back();
      return HSA_STATUS_SUCCESS;
    }
  }

  kernarg->ptr = nullptr;
  kernarg->size = size;
  return hsa_memory_allocate(device->kernarg_region, size, &kernarg->ptr);
}

bool valid_allocation(void *ptr, lr_device_t device, size_t size) {
  auto allocation = g_allocations.find(ptr);
  if (allocation == g_allocations.end()) {
    return false;
  }
  const AllocationInfo &info = allocation->second;
  return info.device_index == device.index && size <= info.size;
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
    lr_status_t drain_status = LR_SUCCESS;
    for (DeviceState &device : g_devices) {
      lr_status_t status = drain_device_locked(&device);
      if (status != LR_SUCCESS && drain_status == LR_SUCCESS) {
        drain_status = status;
      }
      release_device_pools_locked(&device, &drain_status);
    }

    for (lr_module_t *module : g_modules) {
      destroy_module_resources(module);
    }
    g_modules.clear();
    g_kernels.clear();

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

  g_allocations[*ptr] = AllocationInfo{device.index, size};
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

  status = hsa_code_object_reader_create_from_memory(
      image, image_size, &loaded_module->reader);
  if (status != HSA_STATUS_SUCCESS) {
    delete loaded_module;
    return to_lr_status(status);
  }

  status = hsa_executable_create_alt(
      profile, HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR, nullptr,
      &loaded_module->executable);
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
  if (!state.queue || !state.has_kernarg_region ||
      args_size > kernel->kernarg_size) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (state.pending_dispatches.size() >= state.queue->size) {
    lr_status_t drain_status = drain_device_locked(&state);
    if (drain_status != LR_SUCCESS) {
      return drain_status;
    }
  }

  KernargBuffer kernarg{};
  hsa_status_t status =
      acquire_kernarg_locked(&state, kernel->kernarg_size, &kernarg);
  if (status != HSA_STATUS_SUCCESS) {
    return to_lr_status(status);
  }
  std::memset(kernarg.ptr, 0, kernel->kernarg_size);
  std::memcpy(kernarg.ptr, args, args_size);

  hsa_signal_t signal{};
  status = acquire_signal_locked(&state, &signal);
  if (status != HSA_STATUS_SUCCESS) {
    state.kernarg_pool.push_back(kernarg);
    return to_lr_status(status);
  }

  const uint64_t index = hsa_queue_add_write_index_scacq_screl(state.queue, 1);
  auto *packets =
      static_cast<hsa_kernel_dispatch_packet_t *>(state.queue->base_address);
  hsa_kernel_dispatch_packet_t *packet =
      &packets[index & (state.queue->size - 1)];
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
  packet->header = packet_header(HSA_PACKET_TYPE_KERNEL_DISPATCH);

  hsa_signal_store_screlease(state.queue->doorbell_signal, index);
  state.pending_dispatches.push_back(PendingDispatch{signal, kernarg});
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
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
  return drain_device_locked(&g_devices[device.index]);
#else
  if (!valid_device(device)) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

}  // extern "C"
