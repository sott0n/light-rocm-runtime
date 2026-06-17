#include "../common/example_utils.h"
#include "triton_manifest.h"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string>

#ifndef LRRT_TRITON_SCALE_MANIFEST
#define LRRT_TRITON_SCALE_MANIFEST "manifest.json"
#endif

typedef struct triton_scale_args_t {
  const float *x;
  float *out;
  float factor;
  int32_t n;
  void *triton_scratch_0;
  void *triton_scratch_1;
} triton_scale_args_t;

static_assert(sizeof(triton_scale_args_t) == 40,
              "Triton scale kernarg layout must match manifest");

static std::string bundle_file_path(const char *manifest_path,
                                    const std::string &file_name) {
  std::string path(manifest_path);
  size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) {
    return file_name;
  }
  return path.substr(0, slash + 1) + file_name;
}

int main(void) {
  try {
    lrrt::Runtime runtime;

    uint32_t count = runtime.device_count();
    printf("devices: %u\n", count);
    if (count == 0) {
      return 0;
    }

    lrrt::Device device = runtime.open_device(0);
    printf("opened device: %u\n", device.index());

    const uint32_t n = 1024;
    const float factor = 1.75f;
    float x[n];
    float out[n];
    for (uint32_t i = 0; i < n; ++i) {
      x[i] = (float)i;
      out[i] = 0.0f;
    }

    lrrt::DeviceBuffer device_x(device, sizeof(x));
    lrrt::DeviceBuffer device_out(device, sizeof(out));
    lrrt::copy_to_device(device_x, x);

    std::vector<unsigned char> manifest =
        lrrt_example::read_file(LRRT_TRITON_SCALE_MANIFEST);
    lrrt_example::triton::KernelManifest kernel_manifest =
        lrrt_example::triton::parse_first_kernel_manifest(manifest);
    if (kernel_manifest.kernarg_size != sizeof(triton_scale_args_t) ||
        kernel_manifest.arg_offsets.size() != 6 ||
        kernel_manifest.arg_offsets[0] != offsetof(triton_scale_args_t, x) ||
        kernel_manifest.arg_offsets[1] != offsetof(triton_scale_args_t, out) ||
        kernel_manifest.arg_offsets[2] !=
            offsetof(triton_scale_args_t, factor) ||
        kernel_manifest.arg_offsets[3] != offsetof(triton_scale_args_t, n) ||
        kernel_manifest.arg_offsets[4] !=
            offsetof(triton_scale_args_t, triton_scratch_0) ||
        kernel_manifest.arg_offsets[5] !=
            offsetof(triton_scale_args_t, triton_scratch_1)) {
      throw std::runtime_error("Triton manifest does not match C++ kernarg");
    }
    printf("loaded Triton manifest for kernel: %s\n",
           kernel_manifest.name.c_str());

    std::string hsaco_path =
        bundle_file_path(LRRT_TRITON_SCALE_MANIFEST,
                         kernel_manifest.code_object);
    std::vector<unsigned char> hsaco =
        lrrt_example::read_file(hsaco_path.c_str());
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel(kernel_manifest.symbol.c_str());

    triton_scale_args_t kernel_args = {
        (const float *)device_x.data(),
        (float *)device_out.data(),
        factor,
        (int32_t)n,
        nullptr,
        nullptr,
    };

    lr_launch_config_t config =
        lrrt_example::triton::launch_config_from_manifest(kernel_manifest, n);
    lrrt::launch(kernel, config, kernel_args);
    device.synchronize();
    lrrt::copy_to_host(out, device_out);

    for (uint32_t i = 0; i < n; ++i) {
      if (fabsf(out[i] - (factor * x[i])) > 0.001f) {
        throw std::runtime_error("triton_scale result mismatch");
      }
    }

    printf("triton_scale: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
