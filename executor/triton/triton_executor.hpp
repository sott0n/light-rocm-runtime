#ifndef LRRT_EXECUTOR_TRITON_EXECUTOR_HPP_
#define LRRT_EXECUTOR_TRITON_EXECUTOR_HPP_

#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include <stddef.h>
#include <stdint.h>

#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lrrt::executor::triton {

namespace detail {

inline std::string join_names(const std::vector<std::string> &names) {
  if (names.empty()) {
    return "<none>";
  }

  std::string result;
  for (size_t i = 0; i < names.size(); ++i) {
    if (i != 0) {
      result += ", ";
    }
    result += names[i];
  }
  return result;
}

template <typename Map> std::vector<std::string> keys(const Map &values) {
  std::vector<std::string> names;
  names.reserve(values.size());
  for (const auto &entry : values) {
    names.push_back(entry.first);
  }
  return names;
}

inline std::string kernel_label(const lrrt::KernelManifest &manifest) {
  return manifest.name + " (" + manifest.symbol + ")";
}

inline const char *display_string(const char *value) {
  return value ? value : "<null>";
}

} // namespace detail

class BundleSet {
public:
  explicit BundleSet(lrrt::Device device) : device_(device) {}

  void add(const std::string &name, const char *manifest_path,
           const std::string &kernel_name) {
    try {
      bundles_[name] = std::make_unique<lrrt::Bundle>(device_, manifest_path,
                                                      kernel_name.c_str());
    } catch (const std::exception &error) {
      throw std::runtime_error(
          "failed to load Triton executor bundle '" + name +
          "' from manifest '" + detail::display_string(manifest_path) +
          "' for kernel '" + kernel_name + "': " + error.what());
    }
  }

  lrrt::Bundle &get(const std::string &name) const {
    auto bundle = bundles_.find(name);
    if (bundle == bundles_.end()) {
      throw std::runtime_error("unknown Triton executor bundle '" + name +
                               "'; available bundles: " +
                               detail::join_names(detail::keys(bundles_)));
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
    try {
      buffers_[name] = std::make_unique<lrrt::DeviceBuffer>(device_, bytes);
    } catch (const std::exception &error) {
      throw std::runtime_error("failed to allocate Triton executor buffer '" +
                               name + "' (" + std::to_string(bytes) +
                               " bytes): " + error.what());
    }
  }

  template <typename T> void allocate(const std::string &name, size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::overflow_error("Triton executor buffer allocation overflows "
                                "size_t for buffer '" +
                                name + "'");
    }
    allocate_bytes(name, count * sizeof(T));
  }

  lrrt::DeviceBuffer &get(const std::string &name) const {
    auto buffer = buffers_.find(name);
    if (buffer == buffers_.end()) {
      throw std::runtime_error("unknown Triton executor buffer '" + name +
                               "'; available buffers: " +
                               detail::join_names(detail::keys(buffers_)));
    }
    return *buffer->second;
  }

  template <typename T>
  T *ptr(const std::string &name, size_t element_offset = 0) const {
    lrrt::DeviceBuffer &buffer = get(name);
    if (element_offset > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::overflow_error("Triton executor buffer offset overflows "
                                "size_t for buffer '" +
                                name + "'");
    }
    size_t byte_offset = element_offset * sizeof(T);
    if (byte_offset > buffer.size()) {
      throw std::out_of_range(
          "Triton executor buffer offset is out of range for '" + name +
          "': element_offset=" + std::to_string(element_offset) +
          ", element_size=" + std::to_string(sizeof(T)) +
          ", byte_offset=" + std::to_string(byte_offset) +
          ", buffer_size=" + std::to_string(buffer.size()));
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
  if (!name || name[0] == '\0') {
    throw std::invalid_argument(
        "Triton executor launch argument name is empty");
  }
  const auto *bytes = reinterpret_cast<const unsigned char *>(&value);
  return LaunchArg{std::string(name),
                   std::vector<unsigned char>(bytes, bytes + sizeof(T))};
}

inline void launch(const lrrt::Queue &queue, lrrt::Bundle &bundle, uint32_t n,
                   std::initializer_list<LaunchArg> args,
                   const std::vector<const lrrt::Event *> &dependencies = {}) {
  lrrt::KernargBuffer kernargs = bundle.make_args();
  std::unordered_set<std::string> bound_names;
  for (const LaunchArg &arg : args) {
    if (!bound_names.insert(arg.name).second) {
      throw std::runtime_error("duplicate Triton executor launch argument '" +
                               arg.name + "' for kernel " +
                               detail::kernel_label(bundle.manifest()));
    }
    try {
      kernargs.set_raw(arg.name.c_str(), arg.value.data(), arg.value.size());
    } catch (const std::exception &error) {
      throw std::runtime_error(
          "failed to bind Triton executor launch argument '" + arg.name +
          "' for kernel " + detail::kernel_label(bundle.manifest()) + ": " +
          error.what());
    }
  }
  try {
    kernargs.bind_optional_nulls();
    bundle.launch(queue, n, kernargs, dependencies);
  } catch (const std::exception &error) {
    throw std::runtime_error("failed to launch Triton executor kernel " +
                             detail::kernel_label(bundle.manifest()) + ": " +
                             error.what());
  }
}

} // namespace lrrt::executor::triton

#endif // LRRT_EXECUTOR_TRITON_EXECUTOR_HPP_
