#pragma once

#include "executor/iree/hal_driver.h"

#include <cstring>
#include <fstream>
#include <stdio.h>
#include <vector>

#include "iree/io/file_handle.h"
#include "lrrt/lrrt.h"

namespace lrrt::iree::test {

inline bool string_view_equal(iree_string_view_t value, const char *expected) {
  return value.size == std::strlen(expected) &&
         std::memcmp(value.data, expected, value.size) == 0;
}

inline bool read_file(const char *path, std::vector<uint8_t> *out_data) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }
  file.seekg(0, std::ios::end);
  const std::streamoff size = file.tellg();
  if (size <= 0) {
    return false;
  }
  file.seekg(0, std::ios::beg);
  out_data->resize(static_cast<size_t>(size));
  return static_cast<bool>(
      file.read(reinterpret_cast<char *>(out_data->data()), size));
}

inline bool expect_ok(iree_status_t status, const char *label) {
  if (iree_status_is_ok(status)) {
    return true;
  }
  fprintf(stderr, "%s failed\n", label);
  iree_status_ignore(status);
  return false;
}

inline bool expect_unimplemented(iree_status_t status, const char *label) {
  if (iree_status_is_unimplemented(status)) {
    iree_status_free(status);
    return true;
  }
  fprintf(stderr, "%s did not return IREE_STATUS_UNIMPLEMENTED\n", label);
  iree_status_ignore(status);
  return false;
}

inline bool expect_unavailable(iree_status_t status, const char *label) {
  if (iree_status_is_unavailable(status)) {
    iree_status_free(status);
    return true;
  }
  fprintf(stderr, "%s did not return IREE_STATUS_UNAVAILABLE\n", label);
  iree_status_ignore(status);
  return false;
}

template <typename T, void (*ReleaseFn)(T *)> class ScopedIreePtr {
public:
  ScopedIreePtr() = default;
  ScopedIreePtr(const ScopedIreePtr &) = delete;
  ScopedIreePtr &operator=(const ScopedIreePtr &) = delete;
  ~ScopedIreePtr() { reset(); }

  T *get() const { return ptr_; }
  T **out() {
    reset();
    return &ptr_;
  }
  T *release() {
    T *value = ptr_;
    ptr_ = nullptr;
    return value;
  }
  void reset(T *value = nullptr) {
    if (ptr_) {
      ReleaseFn(ptr_);
    }
    ptr_ = value;
  }

private:
  T *ptr_ = nullptr;
};

using DriverPtr = ScopedIreePtr<iree_hal_driver_t, iree_hal_driver_release>;
using DevicePtr = ScopedIreePtr<iree_hal_device_t, iree_hal_device_release>;
using BufferPtr = ScopedIreePtr<iree_hal_buffer_t, iree_hal_buffer_release>;
using CommandBufferPtr =
    ScopedIreePtr<iree_hal_command_buffer_t, iree_hal_command_buffer_release>;
using ExecutableCachePtr = ScopedIreePtr<iree_hal_executable_cache_t,
                                         iree_hal_executable_cache_release>;
using ExecutablePtr =
    ScopedIreePtr<iree_hal_executable_t, iree_hal_executable_release>;
using FilePtr = ScopedIreePtr<iree_hal_file_t, iree_hal_file_release>;
using SemaphorePtr =
    ScopedIreePtr<iree_hal_semaphore_t, iree_hal_semaphore_release>;

class RegistryPtr {
public:
  RegistryPtr() = default;
  RegistryPtr(const RegistryPtr &) = delete;
  RegistryPtr &operator=(const RegistryPtr &) = delete;
  ~RegistryPtr() { reset(); }

  iree_hal_driver_registry_t *get() const { return ptr_; }
  iree_hal_driver_registry_t **out() {
    reset();
    return &ptr_;
  }
  void reset(iree_hal_driver_registry_t *value = nullptr) {
    if (ptr_) {
      iree_hal_driver_registry_free(ptr_);
    }
    ptr_ = value;
  }

private:
  iree_hal_driver_registry_t *ptr_ = nullptr;
};

class FileHandlePtr {
public:
  FileHandlePtr() = default;
  FileHandlePtr(const FileHandlePtr &) = delete;
  FileHandlePtr &operator=(const FileHandlePtr &) = delete;
  ~FileHandlePtr() { reset(); }

  iree_io_file_handle_t *get() const { return ptr_; }
  iree_io_file_handle_t **out() {
    reset();
    return &ptr_;
  }
  void reset(iree_io_file_handle_t *value = nullptr) {
    if (ptr_) {
      iree_io_file_handle_release(ptr_);
    }
    ptr_ = value;
  }

private:
  iree_io_file_handle_t *ptr_ = nullptr;
};

struct HalFixture {
  RegistryPtr registry;
  DriverPtr driver;
  DevicePtr device;
  iree_hal_allocator_t *allocator = nullptr;

  bool setup() {
    if (!expect_ok(iree_hal_driver_registry_allocate(iree_allocator_system(),
                                                     registry.out()),
                   "iree_hal_driver_registry_allocate")) {
      return false;
    }
    if (!expect_ok(lrrt_iree_hal_register_all_available_drivers(registry.get()),
                   "lrrt_iree_hal_register_all_available_drivers")) {
      return false;
    }
    if (!expect_ok(iree_hal_driver_registry_try_create(
                       registry.get(), IREE_SV("lrrt"), iree_allocator_system(),
                       driver.out()),
                   "iree_hal_driver_registry_try_create")) {
      return false;
    }
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    if (!expect_ok(iree_hal_driver_create_default_device(
                       driver.get(), &create_params, iree_allocator_system(),
                       device.out()),
                   "iree_hal_driver_create_default_device")) {
      return false;
    }
    allocator = iree_hal_device_allocator(device.get());
    return allocator != nullptr;
  }
};

inline iree_hal_buffer_params_t device_buffer_params() {
  iree_hal_buffer_params_t params = {};
  params.usage =
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  return params;
}

inline iree_hal_buffer_params_t host_buffer_params() {
  iree_hal_buffer_params_t params = {};
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
                 IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                 IREE_HAL_BUFFER_USAGE_MAPPING;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  return params;
}

inline int run_test(const char *name, bool (*test_fn)()) {
  if (!test_fn()) {
    fprintf(stderr, "%s failed\n", name);
    return 1;
  }
  return 0;
}

} // namespace lrrt::iree::test
