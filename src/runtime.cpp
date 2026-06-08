#include "lrrt/lrrt.h"

#include <atomic>

namespace {

std::atomic<bool> g_initialized{false};

bool valid_device(lr_device_t device) { return device.index == 0; }

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
  return LR_SUCCESS;
}

lr_status_t lr_shutdown(void) {
  bool expected = true;
  if (!g_initialized.compare_exchange_strong(expected, false)) {
    return LR_ERROR_NOT_INITIALIZED;
  }
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
  return LR_SUCCESS;
}

lr_status_t lr_device_open(uint32_t index, lr_device_t *device) {
  if (!device) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (index != 0) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  device->index = index;
  return LR_ERROR_NOT_SUPPORTED;
}

lr_status_t lr_malloc(lr_device_t device, size_t size, void **ptr) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!valid_device(device) || size == 0 || !ptr) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  *ptr = nullptr;
  return LR_ERROR_NOT_SUPPORTED;
}

lr_status_t lr_free(lr_device_t device, void *ptr) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!valid_device(device) || !ptr) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  return LR_ERROR_NOT_SUPPORTED;
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

  return LR_ERROR_NOT_SUPPORTED;
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
  if (!valid_device(device)) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  return LR_ERROR_NOT_SUPPORTED;
}

}  // extern "C"
