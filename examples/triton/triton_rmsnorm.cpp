#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdexcept>
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
  int32_t hidden;
  void *triton_global_scratch;
  void *triton_profile_scratch;
} triton_rmsnorm_args_t;

static_assert(sizeof(triton_rmsnorm_args_t) == 56,
              "Triton rmsnorm kernarg layout must match manifest");

static constexpr uint32_t kMaxHiddenSize = 4096;

static void run_case(lrrt::Device &device, lrrt::Bundle &bundle, uint32_t rows,
                     uint32_t hidden) {
  if (hidden == 0 || hidden > kMaxHiddenSize) {
    throw std::invalid_argument("RMSNorm hidden size must be in [1, 4096]");
  }
  const uint32_t elements = rows * hidden;
  const float eps = 1.0e-5f;
  std::vector<float> x(elements);
  std::vector<float> weight(hidden);
  std::vector<float> out(elements, 0.0f);
  for (uint32_t i = 0; i < hidden; ++i) {
    weight[i] = 1.0f + 0.001f * (float)(i % 29);
  }
  for (uint32_t i = 0; i < elements; ++i) {
    x[i] = 0.125f * (float)((int32_t)(i % 17) - 8);
  }

  lrrt::DeviceBuffer device_x(device, x.size() * sizeof(float));
  lrrt::DeviceBuffer device_weight(device, weight.size() * sizeof(float));
  lrrt::DeviceBuffer device_out(device, out.size() * sizeof(float));
  lrrt::copy_to_device(device_x, x.data(), x.size() * sizeof(float));
  lrrt::copy_to_device(device_weight, weight.data(),
                       weight.size() * sizeof(float));

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
  lrrt::copy_to_host(out.data(), device_out, out.size() * sizeof(float));

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
            "triton_rmsnorm hidden=%u mismatch at %u: actual=%f "
            "expected=%f diff=%f\n",
            hidden, max_index, max_actual, max_expected, max_diff);
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

    lrrt::Bundle rmsnorm_1024(device, LRRT_TRITON_RMSNORM_MANIFEST,
                              "rmsnorm_1024");
    lrrt::Bundle rmsnorm_2048(device, LRRT_TRITON_RMSNORM_MANIFEST,
                              "rmsnorm_2048");
    lrrt::Bundle rmsnorm_4096(device, LRRT_TRITON_RMSNORM_MANIFEST,
                              "rmsnorm_4096");
    const std::vector<size_t> arg_offsets = {
        offsetof(triton_rmsnorm_args_t, x),
        offsetof(triton_rmsnorm_args_t, weight),
        offsetof(triton_rmsnorm_args_t, out),
        offsetof(triton_rmsnorm_args_t, eps),
        offsetof(triton_rmsnorm_args_t, rows),
        offsetof(triton_rmsnorm_args_t, hidden),
        offsetof(triton_rmsnorm_args_t, triton_global_scratch),
        offsetof(triton_rmsnorm_args_t, triton_profile_scratch),
    };
    for (lrrt::Bundle *bundle : {&rmsnorm_1024, &rmsnorm_2048, &rmsnorm_4096}) {
      lrrt::require_kernarg_layout(bundle->manifest(),
                                   sizeof(triton_rmsnorm_args_t), arg_offsets);
      printf("loaded Triton manifest for kernel: %s\n",
             bundle->manifest().name.c_str());
    }

    for (uint32_t hidden : {1u, 127u, 768u, 1003u, 1024u, 1025u, 1536u, 2048u,
                            2049u, 3072u, 4096u}) {
      lrrt::Bundle *bundle = &rmsnorm_4096;
      if (hidden <= 1024) {
        bundle = &rmsnorm_1024;
      } else if (hidden <= 2048) {
        bundle = &rmsnorm_2048;
      }
      run_case(device, *bundle, 4, hidden);
    }

    printf("triton_rmsnorm: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
