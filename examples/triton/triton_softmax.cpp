#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#ifndef LRRT_TRITON_SOFTMAX_MANIFEST
#define LRRT_TRITON_SOFTMAX_MANIFEST "manifest.json"
#endif

static void run_case(lrrt::Device &device, uint32_t rows, uint32_t hidden) {
  uint32_t block_size = hidden <= 1024 ? 1024 : hidden <= 2048 ? 2048 : 4096;
  std::string kernel_name = "softmax_fp32_" + std::to_string(block_size);
  lrrt::Bundle bundle(device, LRRT_TRITON_SOFTMAX_MANIFEST,
                      kernel_name.c_str());

  const uint32_t elements = rows * hidden;
  std::vector<float> x(elements);
  std::vector<float> out(elements, 0.0f);
  for (uint32_t i = 0; i < elements; ++i) {
    int32_t lane = (int32_t)(i % 31) - 15;
    x[i] = 0.125f * (float)lane + 0.01f * (float)(i / hidden);
  }

  lrrt::DeviceBuffer device_x(device, x.size() * sizeof(float));
  lrrt::DeviceBuffer device_out(device, out.size() * sizeof(float));
  lrrt::copy_to_device(device_x, x);

  lrrt::KernargBuffer kernel_args = bundle.make_args();
  kernel_args.set("x", (const float *)device_x.data());
  kernel_args.set("out", (float *)device_out.data());
  kernel_args.set("rows", (int32_t)rows);
  kernel_args.set("hidden", (int32_t)hidden);
  kernel_args.bind_optional_nulls();
  bundle.launch(rows, kernel_args);
  device.synchronize();
  lrrt::copy_to_host(out, device_out);

  float max_diff = 0.0f;
  float max_sum_diff = 0.0f;
  for (uint32_t row = 0; row < rows; ++row) {
    uint32_t base = row * hidden;
    float max_value = x[base];
    for (uint32_t col = 1; col < hidden; ++col) {
      max_value = fmaxf(max_value, x[base + col]);
    }

    float denominator = 0.0f;
    for (uint32_t col = 0; col < hidden; ++col) {
      denominator += expf(x[base + col] - max_value);
    }

    float row_sum = 0.0f;
    for (uint32_t col = 0; col < hidden; ++col) {
      float expected = expf(x[base + col] - max_value) / denominator;
      float actual = out[base + col];
      max_diff = fmaxf(max_diff, fabsf(actual - expected));
      row_sum += actual;
    }
    max_sum_diff = fmaxf(max_sum_diff, fabsf(row_sum - 1.0f));
  }
  if (max_diff > 0.0005f || max_sum_diff > 0.0005f) {
    fprintf(stderr,
            "triton_softmax hidden=%u mismatch: max_diff=%f "
            "max_sum_diff=%f\n",
            hidden, max_diff, max_sum_diff);
    throw std::runtime_error("triton_softmax result mismatch");
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
    for (uint32_t hidden : {513u, 1536u, 3072u}) {
      run_case(device, 4, hidden);
    }

    printf("triton_softmax: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
