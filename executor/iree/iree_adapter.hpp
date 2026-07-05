#ifndef LRRT_EXECUTOR_IREE_IREE_ADAPTER_HPP_
#define LRRT_EXECUTOR_IREE_IREE_ADAPTER_HPP_

#include "lrrt/lrrt.hpp"

#include <stddef.h>
#include <stdint.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lrrt::executor::iree {

class UnsupportedFeature : public std::runtime_error {
public:
  explicit UnsupportedFeature(const std::string &feature)
      : std::runtime_error("unsupported IREE HAL adapter feature: " + feature) {
  }
};

class Fence {
public:
  explicit Fence(lrrt::Device device) : event_(device) {}

  lr_event_t *get() const { return event_.get(); }

  void signal_from(const lrrt::Queue &queue) const { event_.record(queue); }
  void wait() const { event_.synchronize(); }

private:
  lrrt::Event event_;
};

class Buffer {
public:
  Buffer(lrrt::Device device, size_t size)
      : device_(device), buffer_(device, size) {}

  void *device_ptr() const { return buffer_.data(); }
  size_t size() const { return buffer_.size(); }
  lr_device_t device() const { return device_.get(); }

  void write(const void *src, size_t size) {
    lrrt::copy_to_device(buffer_, src, size);
  }

  void read(void *dst, size_t size) const {
    lrrt::copy_to_host(dst, buffer_, size);
  }

private:
  lrrt::Device device_;
  lrrt::DeviceBuffer buffer_;
};

class Executable {
public:
  Executable(lrrt::Device device, const void *image, size_t image_size)
      : module_(device, image, image_size) {}

  lrrt::Kernel entry_point(const char *name) const {
    return module_.kernel(name);
  }

private:
  lrrt::Module module_;
};

class CommandQueue {
public:
  explicit CommandQueue(lrrt::Device device) : queue_(device) {}

  lr_queue_t *get() const { return queue_.get(); }

  void dispatch(const lrrt::Kernel &kernel, const lr_launch_config_t &config,
                const void *args, size_t args_size,
                const std::vector<const Fence *> &wait_fences = {}) const {
    std::vector<lr_event_t *> dependencies;
    dependencies.reserve(wait_fences.size());
    for (const Fence *fence : wait_fences) {
      dependencies.push_back(fence ? fence->get() : nullptr);
    }
    lrrt::check(lr_launch_on_queue_with_dependencies(
                    queue_.get(), kernel.get(), &config, args, args_size,
                    dependencies.data(), dependencies.size()),
                "lr_launch_on_queue_with_dependencies");
  }

  void signal(Fence &fence) const { fence.signal_from(queue_); }
  void wait_idle() const { queue_.synchronize(); }

private:
  lrrt::Queue queue_;
};

class Device {
public:
  explicit Device(uint32_t index)
      : runtime_(), device_(runtime_.open_device(index)), queue_(device_) {}

  lrrt::Device get() const { return device_; }
  CommandQueue &queue() { return queue_; }
  const CommandQueue &queue() const { return queue_; }

  Buffer allocate_buffer(size_t size) const { return Buffer(device_, size); }

  Fence create_fence() const { return Fence(device_); }

  Executable load_executable(const void *image, size_t image_size) const {
    return Executable(device_, image, image_size);
  }

private:
  lrrt::Runtime runtime_;
  lrrt::Device device_;
  CommandQueue queue_;
};

inline void require_unsupported(const std::string &feature) {
  throw UnsupportedFeature(feature);
}

} // namespace lrrt::executor::iree

#endif // LRRT_EXECUTOR_IREE_IREE_ADAPTER_HPP_
