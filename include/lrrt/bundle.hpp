#ifndef LRRT_BUNDLE_HPP_
#define LRRT_BUNDLE_HPP_

#include "lrrt/lrrt.hpp"

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

namespace lrrt {

struct KernelArgument {
  std::string name;
  std::string type;
  size_t offset;
  size_t size;
  bool optional;
};

struct KernelManifest {
  std::string target;
  std::string name;
  std::string symbol;
  std::string code_object;
  uint32_t block[3];
  uint32_t grid_multiplier;
  uint32_t grid_divisor;
  size_t kernarg_size;
  uint32_t shared_memory_bytes;
  std::vector<size_t> arg_offsets;
  std::vector<KernelArgument> args;
};

LRRT_API KernelManifest parse_bundle_manifest(const void *data, size_t size);
LRRT_API KernelManifest parse_bundle_manifest(const void *data, size_t size,
                                              const char *kernel_name);
LRRT_API std::vector<KernelManifest> parse_bundle_manifests(const void *data,
                                                            size_t size);

inline KernelManifest
parse_bundle_manifest(const std::vector<unsigned char> &data) {
  return parse_bundle_manifest(data.data(), data.size());
}

inline KernelManifest
parse_bundle_manifest(const std::vector<unsigned char> &data,
                      const char *kernel_name) {
  return parse_bundle_manifest(data.data(), data.size(), kernel_name);
}

inline std::vector<KernelManifest>
parse_bundle_manifests(const std::vector<unsigned char> &data) {
  return parse_bundle_manifests(data.data(), data.size());
}

LRRT_API lr_launch_config_t
launch_config_from_manifest(const KernelManifest &manifest, uint32_t n);

LRRT_API void require_kernarg_layout(const KernelManifest &manifest,
                                     size_t kernarg_size,
                                     const std::vector<size_t> &arg_offsets);

class LRRT_API KernargBuffer {
public:
  explicit KernargBuffer(const KernelManifest &manifest);

  void set_raw(size_t index, const void *value, size_t value_size);
  void set_raw(const char *name, const void *value, size_t value_size);

  template <typename T> void set(size_t index, const T &value) {
    set_raw(index, &value, sizeof(T));
  }

  template <typename T> void set(int index, const T &value) {
    if (index < 0) {
      throw std::out_of_range("bundle kernarg index is out of range");
    }
    set_raw(static_cast<size_t>(index), &value, sizeof(T));
  }

  template <typename T> void set(const char *name, const T &value) {
    set_raw(name, &value, sizeof(T));
  }

  template <typename T> void set(const std::string &name, const T &value) {
    set_raw(name.c_str(), &value, sizeof(T));
  }

  void validate() const;

  const void *data() const { return data_.data(); }
  void *data() { return data_.data(); }
  size_t size() const { return data_.size(); }
  const std::vector<unsigned char> &bytes() const { return data_; }

private:
  std::vector<unsigned char> data_;
  std::vector<KernelArgument> args_;
  std::vector<bool> bound_;
};

class LRRT_API Bundle {
public:
  Bundle(Device device, const char *manifest_path);
  Bundle(Device device, const char *manifest_path, const char *kernel_name);

  const KernelManifest &manifest() const { return manifest_; }
  Kernel kernel() const { return kernel_; }
  KernargBuffer make_args() const;
  lr_launch_config_t launch_config(uint32_t n) const;
  void launch(uint32_t n, const KernargBuffer &args) const;
  void launch(uint32_t n, const KernargBuffer &args,
              const std::vector<const Event *> &dependencies) const;
  void launch(const Queue &queue, uint32_t n, const KernargBuffer &args,
              const std::vector<const Event *> &dependencies = {}) const;

private:
  static KernelManifest load_manifest(const char *manifest_path,
                                      const char *kernel_name);
  static std::vector<unsigned char>
  read_code_object(const char *manifest_path, const KernelManifest &manifest);

  KernelManifest manifest_;
  Module module_;
  Kernel kernel_;
};

} // namespace lrrt

#endif // LRRT_BUNDLE_HPP_
