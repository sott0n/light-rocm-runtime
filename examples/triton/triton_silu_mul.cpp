#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#ifndef LRRT_TRITON_SILU_MUL_MANIFEST
#define LRRT_TRITON_SILU_MUL_MANIFEST "manifest.json"
#endif

typedef struct triton_silu_mul_args_t {
  const float *gate;
  const float *up;
  float *out;
  int32_t n;
  int32_t padding;
  void *triton_scratch_0;
  void *triton_scratch_1;
} triton_silu_mul_args_t;

static_assert(sizeof(triton_silu_mul_args_t) == 48,
              "Triton silu_mul kernarg layout must match manifest");

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
    lrrt::require_kernarg_layout(
        kernel_manifest, sizeof(triton_silu_mul_args_t),
        {
            offsetof(triton_silu_mul_args_t, gate),
            offsetof(triton_silu_mul_args_t, up),
            offsetof(triton_silu_mul_args_t, out),
            offsetof(triton_silu_mul_args_t, n),
            offsetof(triton_silu_mul_args_t, triton_scratch_0),
            offsetof(triton_silu_mul_args_t, triton_scratch_1),
        });
    printf("loaded Triton manifest for kernel: %s\n",
           kernel_manifest.name.c_str());

    triton_silu_mul_args_t kernel_args = {
        (const float *)device_gate.data(),
        (const float *)device_up.data(),
        (float *)device_out.data(),
        (int32_t)n,
        0,
        nullptr,
        nullptr,
    };

    lrrt::launch(bundle.kernel(), bundle.launch_config(n), kernel_args);
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
