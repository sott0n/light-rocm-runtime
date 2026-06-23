#ifndef LRRT_MODULE_HPP_
#define LRRT_MODULE_HPP_

#include "lrrt/runtime.hpp"

#include <vector>

namespace lrrt {

class Kernel {
public:
  explicit Kernel(lr_kernel_t *kernel) : kernel_(kernel) {}

  lr_kernel_t *get() const { return kernel_; }

private:
  lr_kernel_t *kernel_;
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

  Module(Device device, const void *image, size_t image_size)
      : Module(device.get(), image, image_size) {}

  Module(Device device, const std::vector<unsigned char> &image)
      : Module(device.get(), image.data(), image.size()) {}

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

  Kernel kernel(const char *name) const {
    lr_kernel_t *kernel = nullptr;
    check(lr_kernel_get(module_, name, &kernel), "lr_kernel_get");
    return Kernel(kernel);
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

#endif // LRRT_MODULE_HPP_
