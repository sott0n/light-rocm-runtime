#ifndef LRRT_EXAMPLES_COMMON_EXAMPLE_UTILS_H_
#define LRRT_EXAMPLES_COMMON_EXAMPLE_UTILS_H_

#include "lrrt/lrrt.h"

#include <stdio.h>

#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace lrrt_example {

inline void check(lr_status_t status, const char *operation) {
  if (status == LR_SUCCESS) {
    return;
  }
  throw std::runtime_error(std::string(operation) + " failed: " +
                           lr_status_string(status));
}

inline std::vector<unsigned char> read_file(const char *path) {
  FILE *file = fopen(path, "rb");
  if (!file) {
    throw std::runtime_error(std::string("failed to open ") + path);
  }
  if (fseek(file, 0, SEEK_END) != 0) {
    fclose(file);
    throw std::runtime_error(std::string("failed to seek ") + path);
  }
  long length = ftell(file);
  if (length <= 0) {
    fclose(file);
    throw std::runtime_error(std::string("empty file ") + path);
  }
  rewind(file);

  std::vector<unsigned char> data(static_cast<size_t>(length));
  if (fread(data.data(), 1, data.size(), file) != data.size()) {
    fclose(file);
    throw std::runtime_error(std::string("failed to read ") + path);
  }
  fclose(file);
  return data;
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
  ~DeviceBuffer() {
    if (ptr_) {
      lr_free(device_, ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  void *data() const { return ptr_; }
  size_t size() const { return size_; }

 private:
  lr_device_t device_;
  void *ptr_;
  size_t size_;
};

class Module {
 public:
  Module(lr_device_t device, const std::vector<unsigned char> &image)
      : module_(nullptr) {
    check(lr_module_load_hsaco(device, image.data(), image.size(), &module_),
          "lr_module_load_hsaco");
  }
  ~Module() {
    if (module_) {
      lr_module_destroy(module_);
    }
  }

  Module(const Module &) = delete;
  Module &operator=(const Module &) = delete;

  lr_kernel_t *kernel(const char *name) const {
    lr_kernel_t *kernel = nullptr;
    check(lr_kernel_get(module_, name, &kernel), "lr_kernel_get");
    return kernel;
  }

 private:
  lr_module_t *module_;
};

}  // namespace lrrt_example

#endif  // LRRT_EXAMPLES_COMMON_EXAMPLE_UTILS_H_
