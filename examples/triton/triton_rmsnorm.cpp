#include "triton_bundle.h"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#ifndef LRRT_TRITON_RMSNORM_MANIFEST
#define LRRT_TRITON_RMSNORM_MANIFEST "manifest.json"
#endif

typedef struct triton_rmsnorm_args_t {
  const float *x;
  const float *weight;
  float *out;
  float eps;
  int32_t rows;
  void *triton_scratch_0;
  void *triton_scratch_1;
} triton_rmsnorm_args_t;

static_assert(sizeof(triton_rmsnorm_args_t) == 48,
              "Triton rmsnorm kernarg layout must match manifest");

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

    const uint32_t rows = 4;
    const uint32_t hidden = 128;
    const uint32_t elements = rows * hidden;
    const float eps = 1.0e-5f;
    float x[elements];
    float weight[hidden];
    float out[elements];
    for (uint32_t i = 0; i < hidden; ++i) {
      weight[i] = 1.0f + 0.001f * (float)(i % 29);
    }
    for (uint32_t i = 0; i < elements; ++i) {
      x[i] = 0.125f * (float)((int32_t)(i % 17) - 8);
      out[i] = 0.0f;
    }

    lrrt::DeviceBuffer device_x(device, sizeof(x));
    lrrt::DeviceBuffer device_weight(device, sizeof(weight));
    lrrt::DeviceBuffer device_out(device, sizeof(out));
    lrrt::copy_to_device(device_x, x);
    lrrt::copy_to_device(device_weight, weight);

    lrrt_example::triton::Bundle bundle(device, LRRT_TRITON_RMSNORM_MANIFEST);
    const lrrt_example::triton::KernelManifest &kernel_manifest =
        bundle.manifest();
    lrrt_example::triton::require_kernarg_layout(
        kernel_manifest, sizeof(triton_rmsnorm_args_t),
        {
            offsetof(triton_rmsnorm_args_t, x),
            offsetof(triton_rmsnorm_args_t, weight),
            offsetof(triton_rmsnorm_args_t, out),
            offsetof(triton_rmsnorm_args_t, eps),
            offsetof(triton_rmsnorm_args_t, rows),
            offsetof(triton_rmsnorm_args_t, triton_scratch_0),
            offsetof(triton_rmsnorm_args_t, triton_scratch_1),
        });
    printf("loaded Triton manifest for kernel: %s\n",
           kernel_manifest.name.c_str());

    triton_rmsnorm_args_t kernel_args = {
        (const float *)device_x.data(),
        (const float *)device_weight.data(),
        (float *)device_out.data(),
        eps,
        (int32_t)rows,
        nullptr,
        nullptr,
    };

    lrrt::launch(bundle.kernel(), bundle.launch_config(rows), kernel_args);
    device.synchronize();
    lrrt::copy_to_host(out, device_out);

    float max_diff = 0.0f;
    uint32_t max_index = 0;
    float max_expected = 0.0f;
    float max_actual = 0.0f;
    for (uint32_t row = 0; row < rows; ++row) {
      float sum_square = 0.0f;
      for (uint32_t col = 0; col < hidden; ++col) {
        float value = x[row * hidden + col];
        sum_square += value * value;
      }
      float scale = 1.0f / sqrtf(sum_square / (float)hidden + eps);
      for (uint32_t col = 0; col < hidden; ++col) {
        uint32_t index = row * hidden + col;
        float expected = x[index] * scale * weight[col];
        float diff = fabsf(out[index] - expected);
        if (diff > max_diff) {
          max_diff = diff;
          max_index = index;
          max_expected = expected;
          max_actual = out[index];
        }
      }
    }
    if (max_diff > 0.002f) {
      fprintf(stderr,
              "triton_rmsnorm result mismatch at %u: actual=%f expected=%f "
              "diff=%f\n",
              max_index, max_actual, max_expected, max_diff);
      throw std::runtime_error("triton_rmsnorm result mismatch");
    }

    printf("triton_rmsnorm: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
