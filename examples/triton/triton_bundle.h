#ifndef LRRT_EXAMPLES_TRITON_BUNDLE_H_
#define LRRT_EXAMPLES_TRITON_BUNDLE_H_

#include "../common/example_utils.h"
#include "triton_manifest.h"
#include "lrrt/lrrt.hpp"

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

namespace lrrt_example {
namespace triton {

inline std::string bundle_file_path(const char *manifest_path,
                                    const std::string &file_name) {
  std::string path(manifest_path);
  size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) {
    return file_name;
  }
  return path.substr(0, slash + 1) + file_name;
}

class Bundle {
public:
  Bundle(lrrt::Device device, const char *manifest_path)
      : manifest_(load_manifest(manifest_path)),
        module_(device, read_code_object(manifest_path, manifest_)),
        kernel_(module_.kernel(manifest_.symbol.c_str())) {}

  const KernelManifest &manifest() const { return manifest_; }
  lrrt::Kernel kernel() const { return kernel_; }

  lr_launch_config_t launch_config(uint32_t n) const {
    return launch_config_from_manifest(manifest_, n);
  }

private:
  static KernelManifest load_manifest(const char *manifest_path) {
    std::vector<unsigned char> manifest = read_file(manifest_path);
    return parse_first_kernel_manifest(manifest);
  }

  static std::vector<unsigned char> read_code_object(
      const char *manifest_path, const KernelManifest &manifest) {
    std::string hsaco_path =
        bundle_file_path(manifest_path, manifest.code_object);
    return read_file(hsaco_path.c_str());
  }

  KernelManifest manifest_;
  lrrt::Module module_;
  lrrt::Kernel kernel_;
};

inline void require_kernarg_layout(const KernelManifest &manifest,
                                   size_t kernarg_size,
                                   const std::vector<size_t> &arg_offsets) {
  if (manifest.kernarg_size != kernarg_size ||
      manifest.arg_offsets != arg_offsets) {
    throw std::runtime_error("Triton manifest does not match C++ kernarg");
  }
}

} // namespace triton
} // namespace lrrt_example

#endif // LRRT_EXAMPLES_TRITON_BUNDLE_H_
