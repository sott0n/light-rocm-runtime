#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string>
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
  int32_t hidden;
  void *triton_global_scratch;
  void *triton_profile_scratch;
} triton_rmsnorm_args_t;

static_assert(sizeof(triton_rmsnorm_args_t) == 56,
              "Triton rmsnorm kernarg layout must match manifest");

static void run_case(lrrt::Device &device, uint32_t rows, uint32_t hidden) {
  uint32_t block_size = hidden <= 1024 ? 1024 : hidden <= 2048 ? 2048 : 4096;
  std::string kernel_name = "rmsnorm_fp32_" + std::to_string(block_size);
  lrrt::Bundle bundle(device, LRRT_TRITON_RMSNORM_MANIFEST,
                      kernel_name.c_str());
  lrrt::require_kernarg_layout(
      bundle.manifest(), sizeof(triton_rmsnorm_args_t),
      {
          offsetof(triton_rmsnorm_args_t, x),
          offsetof(triton_rmsnorm_args_t, weight),
          offsetof(triton_rmsnorm_args_t, out),
          offsetof(triton_rmsnorm_args_t, eps),
          offsetof(triton_rmsnorm_args_t, rows),
          offsetof(triton_rmsnorm_args_t, hidden),
          offsetof(triton_rmsnorm_args_t, triton_global_scratch),
          offsetof(triton_rmsnorm_args_t, triton_profile_scratch),
      });

  const uint32_t elements = rows * hidden;
  const float eps = 1.0e-5f;
  std::vector<float> x(elements);
  std::vector<float> weight(hidden);
  std::vector<float> out(elements, 0.0f);
  for (uint32_t i = 0; i < hidden; ++i) {
    weight[i] = 1.0f + 0.001f * (float)(i % 29);
  }
  for (uint32_t i = 0; i < elements; ++i) {
    x[i] = 0.117f * (float)((int32_t)(i % 17) - 8);
  }

  lrrt::DeviceBuffer device_x(device, x.size() * sizeof(float));
  lrrt::DeviceBuffer device_weight(device, weight.size() * sizeof(float));
  lrrt::DeviceBuffer device_out(device, out.size() * sizeof(float));
  lrrt::copy_to_device(device_x, x);
  lrrt::copy_to_device(device_weight, weight);

  triton_rmsnorm_args_t kernel_args = {
      (const float *)device_x.data(),
      (const float *)device_weight.data(),
      (float *)device_out.data(),
      eps,
      (int32_t)rows,
      (int32_t)hidden,
      nullptr,
      nullptr,
  };
  lrrt::launch(bundle.kernel(), bundle.launch_config(rows), kernel_args);
  device.synchronize();
  lrrt::copy_to_host(out, device_out);

  float max_diff = 0.0f;
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
      max_diff = fmaxf(max_diff, fabsf(out[index] - expected));
    }
  }
  if (max_diff > 0.002f) {
    throw std::runtime_error("triton_rmsnorm result mismatch");
  }
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
    for (uint32_t hidden : {1003u, 1536u, 3072u}) {
      run_case(device, 4, hidden);
    }

    printf("triton_rmsnorm: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
