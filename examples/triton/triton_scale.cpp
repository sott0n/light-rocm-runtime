#include "triton_bundle.h"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

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

    lrrt_example::triton::Bundle bundle(device, LRRT_TRITON_SCALE_MANIFEST);
    const lrrt_example::triton::KernelManifest &kernel_manifest =
        bundle.manifest();
    lrrt_example::triton::require_kernarg_layout(
        kernel_manifest, sizeof(triton_scale_args_t),
        {
            offsetof(triton_scale_args_t, x),
            offsetof(triton_scale_args_t, out),
            offsetof(triton_scale_args_t, factor),
            offsetof(triton_scale_args_t, n),
            offsetof(triton_scale_args_t, triton_scratch_0),
            offsetof(triton_scale_args_t, triton_scratch_1),
        });
    printf("loaded Triton manifest for kernel: %s\n",
           kernel_manifest.name.c_str());

    triton_scale_args_t kernel_args = {
        (const float *)device_x.data(),
        (float *)device_out.data(),
        factor,
        (int32_t)n,
        nullptr,
        nullptr,
    };

    lrrt::launch(bundle.kernel(), bundle.launch_config(n), kernel_args);
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
