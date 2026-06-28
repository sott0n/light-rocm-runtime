#ifndef LRRT_EXECUTOR_TRITON_EXECUTOR_HPP_
#define LRRT_EXECUTOR_TRITON_EXECUTOR_HPP_

#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include <stddef.h>
#include <stdint.h>

#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace lrrt::executor::triton {

class BundleSet {
public:
  explicit BundleSet(lrrt::Device device) : device_(device) {}

  void add(const std::string &name, const char *manifest_path,
           const std::string &kernel_name) {
    bundles_[name] = std::make_unique<lrrt::Bundle>(device_, manifest_path,
                                                    kernel_name.c_str());
  }

  lrrt::Bundle &get(const std::string &name) const {
    auto bundle = bundles_.find(name);
    if (bundle == bundles_.end()) {
      throw std::runtime_error("unknown Triton executor bundle: " + name);
    }
    return *bundle->second;
  }

private:
  lrrt::Device device_;
  std::unordered_map<std::string, std::unique_ptr<lrrt::Bundle>> bundles_;
};

class BufferSet {
public:
  explicit BufferSet(lrrt::Device device) : device_(device) {}

  void allocate_bytes(const std::string &name, size_t bytes) {
    buffers_[name] = std::make_unique<lrrt::DeviceBuffer>(device_, bytes);
  }

  template <typename T> void allocate(const std::string &name, size_t count) {
    allocate_bytes(name, count * sizeof(T));
  }

  lrrt::DeviceBuffer &get(const std::string &name) const {
    auto buffer = buffers_.find(name);
    if (buffer == buffers_.end()) {
      throw std::runtime_error("unknown Triton executor buffer: " + name);
    }
    return *buffer->second;
  }

  template <typename T>
  T *ptr(const std::string &name, size_t element_offset = 0) const {
    lrrt::DeviceBuffer &buffer = get(name);
    size_t byte_offset = element_offset * sizeof(T);
    if (byte_offset > buffer.size()) {
      throw std::runtime_error(
          "Triton executor buffer offset is out of range: " + name);
    }
    auto *base = static_cast<unsigned char *>(buffer.data());
    return reinterpret_cast<T *>(base + byte_offset);
  }

  template <typename T>
  void copy_to(const std::string &name, const std::vector<T> &src) const {
    lrrt::copy_to_device(get(name), src);
  }

  template <typename T>
  void copy_from(std::vector<T> &dst, const std::string &name) const {
    lrrt::copy_to_host(dst, get(name));
  }

private:
  lrrt::Device device_;
  std::unordered_map<std::string, std::unique_ptr<lrrt::DeviceBuffer>> buffers_;
};

struct LaunchArg {
  std::string name;
  std::vector<unsigned char> value;
};

template <typename T> LaunchArg arg(const char *name, const T &value) {
  const auto *bytes = reinterpret_cast<const unsigned char *>(&value);
  return LaunchArg{std::string(name),
                   std::vector<unsigned char>(bytes, bytes + sizeof(T))};
}

inline void launch(const lrrt::Queue &queue, lrrt::Bundle &bundle, uint32_t n,
                   std::initializer_list<LaunchArg> args) {
  lrrt::KernargBuffer kernargs = bundle.make_args();
  for (const LaunchArg &arg : args) {
    kernargs.set_raw(arg.name.c_str(), arg.value.data(), arg.value.size());
  }
  kernargs.bind_optional_nulls();
  bundle.launch(queue, n, kernargs);
}

} // namespace lrrt::executor::triton

#endif // LRRT_EXECUTOR_TRITON_EXECUTOR_HPP_
