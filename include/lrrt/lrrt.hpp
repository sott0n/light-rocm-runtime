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

class Device {
public:
  explicit Device(lr_device_t device) : device_(device) {}

  lr_device_t get() const { return device_; }
  uint32_t index() const { return device_.index; }
  void synchronize() const { check(lr_synchronize(device_), "lr_synchronize"); }

private:
  lr_device_t device_;
};

inline void synchronize(Device device) {
  check(lr_synchronize(device.get()), "lr_synchronize");
}

class Event {
public:
  explicit Event(lr_device_t device) : event_(nullptr) {
    check(lr_event_create(device, &event_), "lr_event_create");
  }

  explicit Event(Device device) : Event(device.get()) {}

  ~Event() { reset(); }

  Event(Event &&other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
  }

  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;

  Event &operator=(Event &&other) noexcept {
    if (this != &other) {
      reset();
      event_ = other.event_;
      other.event_ = nullptr;
    }
    return *this;
  }

  lr_event_t *get() const { return event_; }

  void record() const { check(lr_event_record(event_), "lr_event_record"); }

  void synchronize() const {
    check(lr_event_synchronize(event_), "lr_event_synchronize");
  }

private:
  void reset() noexcept {
    if (event_) {
      lr_event_destroy(event_);
      event_ = nullptr;
    }
  }

  lr_event_t *event_;
};

inline std::vector<lr_event_t *>
event_handles(const std::vector<const Event *> &dependencies) {
  std::vector<lr_event_t *> handles;
  handles.reserve(dependencies.size());
  for (const Event *dependency : dependencies) {
    handles.push_back(dependency ? dependency->get() : nullptr);
  }
  return handles;
}

inline uint64_t elapsed_time_ns(const Event &start, const Event &end) {
  uint64_t elapsed_ns = 0;
  check(lr_event_elapsed_time_ns(start.get(), end.get(), &elapsed_ns),
        "lr_event_elapsed_time_ns");
  return elapsed_ns;
}

class Runtime {
public:
  Runtime() { check(lr_init(), "lr_init"); }
  ~Runtime() { lr_shutdown(); }

  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

  uint32_t device_count() const {
    uint32_t count = 0;
    check(lr_device_count(&count), "lr_device_count");
    return count;
  }

  Device open_device(uint32_t index) const {
    lr_device_t device = {0};
    check(lr_device_open(index, &device), "lr_device_open");
    return Device(device);
  }
};

class DeviceBuffer {
public:
  DeviceBuffer(lr_device_t device, size_t size)
      : device_(device), ptr_(nullptr), size_(size) {
    check(lr_malloc(device_, size_, &ptr_), "lr_malloc");
  }

  DeviceBuffer(Device device, size_t size) : DeviceBuffer(device.get(), size) {}

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
  lr_device_t device() const { return device_; }

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

inline void copy_to_device(DeviceBuffer &dst, const void *src, size_t size) {
  check(
      lr_memcpy(dst.device(), dst.data(), src, size, LR_MEMCPY_HOST_TO_DEVICE),
      "lr_memcpy host to device");
}

inline void copy_to_device_async(DeviceBuffer &dst, const void *src,
                                 size_t size, const Event &event) {
  check(lr_memcpy_async(dst.device(), dst.data(), src, size,
                        LR_MEMCPY_HOST_TO_DEVICE, event.get()),
        "lr_memcpy_async host to device");
}

inline void
copy_to_device_async(DeviceBuffer &dst, const void *src, size_t size,
                     const Event &event,
                     const std::vector<const Event *> &dependencies) {
  std::vector<lr_event_t *> handles = event_handles(dependencies);
  check(lr_memcpy_async_with_dependencies(dst.device(), dst.data(), src, size,
                                          LR_MEMCPY_HOST_TO_DEVICE, event.get(),
                                          handles.data(), handles.size()),
        "lr_memcpy_async_with_dependencies host to device");
}

template <typename T, size_t N>
inline void copy_to_device(DeviceBuffer &dst, const T (&src)[N]) {
  copy_to_device(dst, src, sizeof(src));
}

template <typename T, size_t N>
inline void copy_to_device_async(DeviceBuffer &dst, const T (&src)[N],
                                 const Event &event) {
  copy_to_device_async(dst, src, sizeof(src), event);
}

template <typename T>
inline void copy_to_device(DeviceBuffer &dst, const std::vector<T> &src) {
  if (src.empty()) {
    return;
  }
  copy_to_device(dst, src.data(), src.size() * sizeof(T));
}

template <typename T>
inline void copy_to_device_async(DeviceBuffer &dst, const std::vector<T> &src,
                                 const Event &event) {
  if (src.empty()) {
    return;
  }
  copy_to_device_async(dst, src.data(), src.size() * sizeof(T), event);
}

inline void copy_to_host(void *dst, const DeviceBuffer &src, size_t size) {
  check(
      lr_memcpy(src.device(), dst, src.data(), size, LR_MEMCPY_DEVICE_TO_HOST),
      "lr_memcpy device to host");
}

inline void copy_to_host_async(void *dst, const DeviceBuffer &src, size_t size,
                               const Event &event) {
  check(lr_memcpy_async(src.device(), dst, src.data(), size,
                        LR_MEMCPY_DEVICE_TO_HOST, event.get()),
        "lr_memcpy_async device to host");
}

inline void copy_to_host_async(void *dst, const DeviceBuffer &src, size_t size,
                               const Event &event,
                               const std::vector<const Event *> &dependencies) {
  std::vector<lr_event_t *> handles = event_handles(dependencies);
  check(lr_memcpy_async_with_dependencies(src.device(), dst, src.data(), size,
                                          LR_MEMCPY_DEVICE_TO_HOST, event.get(),
                                          handles.data(), handles.size()),
        "lr_memcpy_async_with_dependencies device to host");
}

template <typename T, size_t N>
inline void copy_to_host(T (&dst)[N], const DeviceBuffer &src) {
  copy_to_host(dst, src, sizeof(dst));
}

template <typename T, size_t N>
inline void copy_to_host_async(T (&dst)[N], const DeviceBuffer &src,
                               const Event &event) {
  copy_to_host_async(dst, src, sizeof(dst), event);
}

template <typename T>
inline void copy_to_host(std::vector<T> &dst, const DeviceBuffer &src) {
  if (dst.empty()) {
    return;
  }
  copy_to_host(dst.data(), src, dst.size() * sizeof(T));
}

template <typename T>
inline void copy_to_host_async(std::vector<T> &dst, const DeviceBuffer &src,
                               const Event &event) {
  if (dst.empty()) {
    return;
  }
  copy_to_host_async(dst.data(), src, dst.size() * sizeof(T), event);
}

inline void copy_device_to_device(DeviceBuffer &dst, const DeviceBuffer &src,
                                  size_t size) {
  check(lr_memcpy(dst.device(), dst.data(), src.data(), size,
                  LR_MEMCPY_DEVICE_TO_DEVICE),
        "lr_memcpy device to device");
}

inline void copy_device_to_device_async(DeviceBuffer &dst,
                                        const DeviceBuffer &src, size_t size,
                                        const Event &event) {
  check(lr_memcpy_async(dst.device(), dst.data(), src.data(), size,
                        LR_MEMCPY_DEVICE_TO_DEVICE, event.get()),
        "lr_memcpy_async device to device");
}

inline void
copy_device_to_device_async(DeviceBuffer &dst, const DeviceBuffer &src,
                            size_t size, const Event &event,
                            const std::vector<const Event *> &dependencies) {
  std::vector<lr_event_t *> handles = event_handles(dependencies);
  check(lr_memcpy_async_with_dependencies(dst.device(), dst.data(), src.data(),
                                          size, LR_MEMCPY_DEVICE_TO_DEVICE,
                                          event.get(), handles.data(),
                                          handles.size()),
        "lr_memcpy_async_with_dependencies device to device");
}

class Kernel {
public:
  explicit Kernel(lr_kernel_t *kernel) : kernel_(kernel) {}

  lr_kernel_t *get() const { return kernel_; }

private:
  lr_kernel_t *kernel_;
};

inline void launch(lr_kernel_t *kernel, const lr_launch_config_t &config,
                   const void *args, size_t args_size) {
  check(lr_launch(kernel, &config, args, args_size), "lr_launch");
}

inline void launch(lr_kernel_t *kernel, const lr_launch_config_t &config,
                   const void *args, size_t args_size,
                   const std::vector<const Event *> &dependencies) {
  std::vector<lr_event_t *> handles = event_handles(dependencies);
  check(lr_launch_with_dependencies(kernel, &config, args, args_size,
                                    handles.data(), handles.size()),
        "lr_launch_with_dependencies");
}

template <typename Args>
inline void launch(lr_kernel_t *kernel, const lr_launch_config_t &config,
                   const Args &args,
                   const std::vector<const Event *> &dependencies) {
  launch(kernel, config, &args, sizeof(args), dependencies);
}

template <typename Args>
inline void launch(lr_kernel_t *kernel, const lr_launch_config_t &config,
                   const Args &args) {
  launch(kernel, config, &args, sizeof(args));
}

inline void launch(const Kernel &kernel, const lr_launch_config_t &config,
                   const void *args, size_t args_size) {
  launch(kernel.get(), config, args, args_size);
}

inline void launch(const Kernel &kernel, const lr_launch_config_t &config,
                   const void *args, size_t args_size,
                   const std::vector<const Event *> &dependencies) {
  launch(kernel.get(), config, args, args_size, dependencies);
}

template <typename Args>
inline void launch(const Kernel &kernel, const lr_launch_config_t &config,
                   const Args &args) {
  launch(kernel.get(), config, args);
}

template <typename Args>
inline void launch(const Kernel &kernel, const lr_launch_config_t &config,
                   const Args &args,
                   const std::vector<const Event *> &dependencies) {
  launch(kernel.get(), config, args, dependencies);
}

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

#endif // LRRT_LRRT_HPP_
