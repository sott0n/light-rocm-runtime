#ifndef LRRT_BUNDLE_HPP_
#define LRRT_BUNDLE_HPP_

#include "lrrt/lrrt.hpp"

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

namespace lrrt {

struct KernelManifest {
  std::string name;
  std::string symbol;
  std::string code_object;
  uint32_t block[3];
  uint32_t grid_multiplier;
  uint32_t grid_divisor;
  size_t kernarg_size;
  uint32_t shared_memory_bytes;
  std::vector<size_t> arg_offsets;
};

LRRT_API KernelManifest parse_bundle_manifest(const void *data, size_t size);

inline KernelManifest
parse_bundle_manifest(const std::vector<unsigned char> &data) {
  return parse_bundle_manifest(data.data(), data.size());
}

LRRT_API lr_launch_config_t
launch_config_from_manifest(const KernelManifest &manifest, uint32_t n);

LRRT_API void require_kernarg_layout(const KernelManifest &manifest,
                                     size_t kernarg_size,
                                     const std::vector<size_t> &arg_offsets);

class LRRT_API Bundle {
public:
  Bundle(Device device, const char *manifest_path);

  const KernelManifest &manifest() const { return manifest_; }
  Kernel kernel() const { return kernel_; }
  lr_launch_config_t launch_config(uint32_t n) const;

private:
  static KernelManifest load_manifest(const char *manifest_path);
  static std::vector<unsigned char>
  read_code_object(const char *manifest_path, const KernelManifest &manifest);

  KernelManifest manifest_;
  Module module_;
  Kernel kernel_;
};

} // namespace lrrt

#endif // LRRT_BUNDLE_HPP_
