#include "runtime_internal.hpp"

#include <string>
#include <unordered_set>

namespace lrrt_internal {

#if LRRT_ENABLE_HSA
namespace {

struct SymbolSearch {
  const char *name;
  std::string descriptor_name;
  hsa_executable_symbol_t symbol;
  bool found;
};

std::unordered_set<lr_module_t *> g_modules;
std::unordered_set<lr_kernel_t *> g_kernels;

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

} // namespace

bool valid_kernel_locked(lr_kernel_t *kernel) {
  return g_kernels.find(kernel) != g_kernels.end() &&
         g_modules.find(kernel->module) != g_modules.end();
}

void release_modules_locked() {
  for (lr_module_t *module : g_modules) {
    destroy_module_resources(module);
  }
  g_modules.clear();
  g_kernels.clear();
}
#endif

} // namespace lrrt_internal

using namespace lrrt_internal;

extern "C" {

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

} // extern "C"
