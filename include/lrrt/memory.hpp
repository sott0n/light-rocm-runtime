#ifndef LRRT_MEMORY_HPP_
#define LRRT_MEMORY_HPP_

#include "lrrt/runtime.hpp"

#include <vector>

namespace lrrt {

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

} // namespace lrrt

#endif // LRRT_MEMORY_HPP_
