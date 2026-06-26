#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#ifndef LRRT_TRITON_SILU_MUL_MANIFEST
#define LRRT_TRITON_SILU_MUL_MANIFEST "manifest.json"
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

    const uint32_t n = 1003;
    std::vector<float> gate(n);
    std::vector<float> up(n);
    std::vector<float> out(n, 0.0f);
    for (uint32_t i = 0; i < n; ++i) {
      gate[i] = 0.125f * (float)((int32_t)(i % 33) - 16);
      up[i] = 0.25f * (float)((int32_t)(i % 17) - 8);
    }

    lrrt::DeviceBuffer device_gate(device, gate.size() * sizeof(float));
    lrrt::DeviceBuffer device_up(device, up.size() * sizeof(float));
    lrrt::DeviceBuffer device_out(device, out.size() * sizeof(float));
    lrrt::copy_to_device(device_gate, gate);
    lrrt::copy_to_device(device_up, up);

    lrrt::Bundle bundle(device, LRRT_TRITON_SILU_MUL_MANIFEST);
    const lrrt::KernelManifest &kernel_manifest = bundle.manifest();
    printf("loaded Triton manifest for kernel: %s\n",
           kernel_manifest.name.c_str());

    lrrt::KernargBuffer kernel_args = bundle.make_args();
    kernel_args.set("gate", (const float *)device_gate.data());
    kernel_args.set("up", (const float *)device_up.data());
    kernel_args.set("out", (float *)device_out.data());
    kernel_args.set("n", (int32_t)n);
    void *scratch = nullptr;
    kernel_args.set("_triton_scratch_0", scratch);
    kernel_args.set("_triton_scratch_1", scratch);

    bundle.launch(n, kernel_args);
    device.synchronize();
    lrrt::copy_to_host(out, device_out);

    float max_diff = 0.0f;
    uint32_t max_index = 0;
    float max_expected = 0.0f;
    float max_actual = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
      float activated = gate[i] / (1.0f + expf(-gate[i]));
      float expected = activated * up[i];
      float diff = fabsf(out[i] - expected);
      if (diff > max_diff) {
        max_diff = diff;
        max_index = i;
        max_expected = expected;
        max_actual = out[i];
      }
    }
    if (max_diff > 0.001f) {
      fprintf(stderr,
              "triton_silu_mul result mismatch at %u: actual=%f "
              "expected=%f diff=%f\n",
              max_index, max_actual, max_expected, max_diff);
      throw std::runtime_error("triton_silu_mul result mismatch");
    }

    printf("triton_silu_mul: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
