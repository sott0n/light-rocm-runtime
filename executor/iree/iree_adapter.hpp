#ifndef LRRT_EXECUTOR_IREE_IREE_ADAPTER_HPP_
#define LRRT_EXECUTOR_IREE_IREE_ADAPTER_HPP_

#include "lrrt/lrrt.hpp"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cstring>
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

struct BindingMetadata {
  uint32_t index = 0;
  std::string type;
  std::vector<std::string> flags;

  bool has_flag(const std::string &flag) const {
    return std::find(flags.begin(), flags.end(), flag) != flags.end();
  }
};

struct KernelMetadata {
  std::string symbol;
  std::vector<std::string> attributes;

  bool has_attribute(const std::string &attribute) const {
    return std::find(attributes.begin(), attributes.end(), attribute) !=
           attributes.end();
  }
};

struct DispatchMetadata {
  std::string executable;
  std::string variant;
  std::string symbol;
};

struct ExportMetadata {
  std::string symbol;
  uint32_t ordinal = 0;
  std::array<uint32_t, 3> workgroup_size = {1, 1, 1};
  uint32_t subgroup_size = 0;
  std::vector<BindingMetadata> bindings;
  KernelMetadata kernel;
  DispatchMetadata dispatch;

  lr_launch_config_t launch_config(lr_dim3_t grid,
                                   uint32_t shared_memory_bytes = 0) const {
    return lr_launch_config_t{
        grid,
        lr_dim3_t{workgroup_size[0], workgroup_size[1], workgroup_size[2]},
        shared_memory_bytes,
    };
  }

  lr_launch_config_t
  launch_config_for_workgroups(lr_dim3_t workgroup_count,
                               uint32_t shared_memory_bytes = 0) const {
    return launch_config(
        lr_dim3_t{
            workgroup_count.x * workgroup_size[0],
            workgroup_count.y * workgroup_size[1],
            workgroup_count.z * workgroup_size[2],
        },
        shared_memory_bytes);
  }
};

class KernargBuilder {
public:
  explicit KernargBuilder(const ExportMetadata &export_metadata)
      : export_metadata_(export_metadata) {}

  std::vector<unsigned char>
  pack_global_buffers(const std::vector<void *> &device_pointers) const {
    if (device_pointers.size() != export_metadata_.bindings.size()) {
      throw std::runtime_error("IREE kernarg buffer count mismatch");
    }

    std::vector<unsigned char> kernargs(device_pointers.size() *
                                        sizeof(void *));
    for (size_t i = 0; i < export_metadata_.bindings.size(); ++i) {
      const BindingMetadata &binding = export_metadata_.bindings[i];
      if (binding.index != i) {
        throw std::runtime_error("IREE binding indices must be contiguous");
      }
      if (binding.type != "storage_buffer" || !binding.has_flag("Indirect")) {
        throw std::runtime_error("unsupported IREE binding for kernarg pack");
      }
      std::memcpy(kernargs.data() + i * sizeof(void *), &device_pointers[i],
                  sizeof(void *));
    }
    return kernargs;
  }

private:
  const ExportMetadata &export_metadata_;
};

struct ExecutableMetadata {
  std::string target;
  std::string executable;
  std::string variant;
  std::vector<ExportMetadata> exports;

  const ExportMetadata *find_export_by_symbol(const std::string &symbol) const {
    const auto it =
        std::find_if(exports.begin(), exports.end(),
                     [&symbol](const ExportMetadata &export_metadata) {
                       return export_metadata.symbol == symbol;
                     });
    return it == exports.end() ? nullptr : &*it;
  }

  const ExportMetadata *find_export_by_ordinal(uint32_t ordinal) const {
    const auto it =
        std::find_if(exports.begin(), exports.end(),
                     [ordinal](const ExportMetadata &export_metadata) {
                       return export_metadata.ordinal == ordinal;
                     });
    return it == exports.end() ? nullptr : &*it;
  }

  const ExportMetadata &
  require_export_by_symbol(const std::string &symbol) const {
    const ExportMetadata *export_metadata = find_export_by_symbol(symbol);
    if (!export_metadata) {
      throw std::runtime_error("missing IREE executable export symbol: " +
                               symbol);
    }
    return *export_metadata;
  }

  const ExportMetadata &require_export_by_ordinal(uint32_t ordinal) const {
    const ExportMetadata *export_metadata = find_export_by_ordinal(ordinal);
    if (!export_metadata) {
      throw std::runtime_error("missing IREE executable export ordinal: " +
                               std::to_string(ordinal));
    }
    return *export_metadata;
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
