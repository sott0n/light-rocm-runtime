#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#ifndef LRRT_TRITON_SAXPY_MANIFEST
#define LRRT_TRITON_SAXPY_MANIFEST "manifest.json"
#endif

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
    const float a = 2.0f;
    float x[n];
    float y[n];
    float out[n];
    for (uint32_t i = 0; i < n; ++i) {
      x[i] = (float)i;
      y[i] = (float)(i * 3);
      out[i] = 0.0f;
    }

    lrrt::DeviceBuffer device_x(device, sizeof(x));
    lrrt::DeviceBuffer device_y(device, sizeof(y));
    lrrt::DeviceBuffer device_out(device, sizeof(out));
    lrrt::copy_to_device(device_x, x);
    lrrt::copy_to_device(device_y, y);

    lrrt::Bundle bundle(device, LRRT_TRITON_SAXPY_MANIFEST);
    const lrrt::KernelManifest &kernel_manifest = bundle.manifest();
    printf("loaded Triton manifest for kernel: %s\n",
           kernel_manifest.name.c_str());

    lrrt::KernargBuffer kernel_args(kernel_manifest);
    kernel_args.set("x", (const float *)device_x.data());
    kernel_args.set("y", (const float *)device_y.data());
    kernel_args.set("out", (float *)device_out.data());
    kernel_args.set("a", a);
    kernel_args.set("n", (int32_t)n);
    void *scratch = nullptr;
    kernel_args.set("_triton_scratch_0", scratch);
    kernel_args.set("_triton_scratch_1", scratch);
    kernel_args.validate();

    lrrt::launch(bundle.kernel(), bundle.launch_config(n), kernel_args.data(),
                 kernel_args.size());
    device.synchronize();
    lrrt::copy_to_host(out, device_out);

    for (uint32_t i = 0; i < n; ++i) {
      if (fabsf(out[i] - (a * x[i] + y[i])) > 0.001f) {
        throw std::runtime_error("triton_saxpy result mismatch");
      }
    }

    printf("triton_saxpy: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
