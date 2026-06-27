#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#ifndef LRRT_TRITON_MATVEC_MANIFEST
#define LRRT_TRITON_MATVEC_MANIFEST "manifest.json"
#endif

static void run_case(lrrt::Device &device, uint32_t outputs, uint32_t hidden) {
  uint32_t block_size = hidden <= 1024 ? 1024 : hidden <= 2048 ? 2048 : 4096;
  std::string kernel_name = "matvec_fp32_" + std::to_string(block_size);
  lrrt::Bundle bundle(device, LRRT_TRITON_MATVEC_MANIFEST, kernel_name.c_str());

  std::vector<float> x(hidden);
  std::vector<float> weight(outputs * hidden);
  std::vector<float> out(outputs, 0.0f);
  for (uint32_t i = 0; i < hidden; ++i) {
    x[i] = 0.03125f * (float)((int32_t)(i % 23) - 11);
  }
  for (uint32_t row = 0; row < outputs; ++row) {
    for (uint32_t col = 0; col < hidden; ++col) {
      uint32_t index = row * hidden + col;
      int32_t lane = (int32_t)((index + row * 7) % 19) - 9;
      weight[index] = 0.015625f * (float)lane;
    }
  }

  lrrt::DeviceBuffer device_x(device, x.size() * sizeof(float));
  lrrt::DeviceBuffer device_weight(device, weight.size() * sizeof(float));
  lrrt::DeviceBuffer device_out(device, out.size() * sizeof(float));
  lrrt::copy_to_device(device_x, x);
  lrrt::copy_to_device(device_weight, weight);

  lrrt::KernargBuffer kernel_args = bundle.make_args();
  kernel_args.set("x", (const float *)device_x.data());
  kernel_args.set("weight", (const float *)device_weight.data());
  kernel_args.set("out", (float *)device_out.data());
  kernel_args.set("outputs", (int32_t)outputs);
  kernel_args.set("hidden", (int32_t)hidden);
  kernel_args.bind_optional_nulls();
  bundle.launch(outputs, kernel_args);
  device.synchronize();
  lrrt::copy_to_host(out, device_out);

  float max_diff = 0.0f;
  uint32_t max_index = 0;
  float max_expected = 0.0f;
  float max_actual = 0.0f;
  for (uint32_t row = 0; row < outputs; ++row) {
    float expected = 0.0f;
    for (uint32_t col = 0; col < hidden; ++col) {
      expected += x[col] * weight[row * hidden + col];
    }
    float diff = fabsf(out[row] - expected);
    if (diff > max_diff) {
      max_diff = diff;
      max_index = row;
      max_expected = expected;
      max_actual = out[row];
    }
  }
  if (max_diff > 0.002f) {
    fprintf(stderr,
            "triton_matvec hidden=%u mismatch at %u: actual=%f "
            "expected=%f diff=%f\n",
            hidden, max_index, max_actual, max_expected, max_diff);
    throw std::runtime_error("triton_matvec result mismatch");
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
    for (uint32_t hidden : {768u, 1536u, 3072u}) {
      run_case(device, 17, hidden);
    }

    printf("triton_matvec: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
