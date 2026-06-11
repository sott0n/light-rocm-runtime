#ifndef LRRT_LRRT_HPP_
#define LRRT_LRRT_HPP_

#include "lrrt/lrrt.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace lrrt {

class Error : public std::runtime_error {
public:
  Error(lr_status_t status, const char *operation)
      : std::runtime_error(std::string(operation) +
                           " failed: " + lr_status_string(status)),
        status_(status) {}

  lr_status_t status() const { return status_; }

private:
  lr_status_t status_;
};

inline void check(lr_status_t status, const char *operation) {
  if (status != LR_SUCCESS) {
    throw Error(status, operation);
  }
}

class Runtime {
public:
  Runtime() { check(lr_init(), "lr_init"); }
  ~Runtime() { lr_shutdown(); }

  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;
};

class DeviceBuffer {
public:
  DeviceBuffer(lr_device_t device, size_t size)
      : device_(device), ptr_(nullptr), size_(size) {
    check(lr_malloc(device_, size_, &ptr_), "lr_malloc");
  }
  ~DeviceBuffer() { reset(); }

  DeviceBuffer(DeviceBuffer &&other) noexcept
      : device_(other.device_), ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      reset();
      device_ = other.device_;
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void *data() const { return ptr_; }
  size_t size() const { return size_; }

private:
  void reset() noexcept {
    if (ptr_) {
      lr_free(device_, ptr_);
      ptr_ = nullptr;
      size_ = 0;
    }
  }

  lr_device_t device_;
  void *ptr_;
  size_t size_;
};

class Module {
public:
  Module(lr_device_t device, const void *image, size_t image_size)
      : module_(nullptr) {
    check(lr_module_load_hsaco(device, image, image_size, &module_),
          "lr_module_load_hsaco");
  }

  Module(lr_device_t device, const std::vector<unsigned char> &image)
      : Module(device, image.data(), image.size()) {}

  ~Module() { reset(); }

  Module(Module &&other) noexcept : module_(other.module_) {
    other.module_ = nullptr;
  }

  Module(const Module &) = delete;
  Module &operator=(const Module &) = delete;

  Module &operator=(Module &&other) noexcept {
    if (this != &other) {
      reset();
      module_ = other.module_;
      other.module_ = nullptr;
    }
    return *this;
  }

  lr_kernel_t *kernel(const char *name) const {
    lr_kernel_t *kernel = nullptr;
    check(lr_kernel_get(module_, name, &kernel), "lr_kernel_get");
    return kernel;
  }

  lr_module_t *get() const { return module_; }

private:
  void reset() noexcept {
    if (module_) {
      lr_module_destroy(module_);
      module_ = nullptr;
    }
  }

  lr_module_t *module_;
};

} // namespace lrrt

#endif // LRRT_LRRT_HPP_
