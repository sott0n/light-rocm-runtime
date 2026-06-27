#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#ifndef LRRT_TRITON_VALUE_AGGREGATION_MANIFEST
#define LRRT_TRITON_VALUE_AGGREGATION_MANIFEST "manifest.json"
#endif

static uint32_t select_block_keys(uint32_t keys) {
  if (keys <= 1024) {
    return 1024;
  }
  if (keys <= 2048) {
    return 2048;
  }
  return 4096;
}

static void run_case(lrrt::Device &device, uint32_t keys, uint32_t head_dim) {
  uint32_t block_keys = select_block_keys(keys);
  std::string kernel_name =
      "value_aggregation_fp32_" + std::to_string(block_keys);
  lrrt::Bundle bundle(device, LRRT_TRITON_VALUE_AGGREGATION_MANIFEST,
                      kernel_name.c_str());

  std::vector<float> probs(keys);
  std::vector<float> v(keys * head_dim);
  std::vector<float> out(head_dim, 0.0f);

  float denominator = 0.0f;
  for (uint32_t row = 0; row < keys; ++row) {
    float value = 1.0f + 0.03125f * (float)((row * 7) % 17);
    probs[row] = value;
    denominator += value;
  }
  for (uint32_t row = 0; row < keys; ++row) {
    probs[row] /= denominator;
  }
  for (uint32_t row = 0; row < keys; ++row) {
    for (uint32_t col = 0; col < head_dim; ++col) {
      uint32_t index = row * head_dim + col;
      int32_t lane = (int32_t)((index + row * 13 + col * 5) % 37) - 18;
      v[index] = 0.015625f * (float)lane;
    }
  }

  lrrt::DeviceBuffer device_probs(device, probs.size() * sizeof(float));
  lrrt::DeviceBuffer device_v(device, v.size() * sizeof(float));
  lrrt::DeviceBuffer device_out(device, out.size() * sizeof(float));
  lrrt::copy_to_device(device_probs, probs);
  lrrt::copy_to_device(device_v, v);

  lrrt::KernargBuffer kernel_args = bundle.make_args();
  kernel_args.set("probs", (const float *)device_probs.data());
  kernel_args.set("v", (const float *)device_v.data());
  kernel_args.set("out", (float *)device_out.data());
  kernel_args.set("keys", (int32_t)keys);
  kernel_args.set("head_dim", (int32_t)head_dim);
  kernel_args.bind_optional_nulls();
  bundle.launch(head_dim, kernel_args);
  device.synchronize();
  lrrt::copy_to_host(out, device_out);

  float max_diff = 0.0f;
  uint32_t max_index = 0;
  float max_expected = 0.0f;
  float max_actual = 0.0f;
  for (uint32_t col = 0; col < head_dim; ++col) {
    float expected = 0.0f;
    for (uint32_t row = 0; row < keys; ++row) {
      expected += probs[row] * v[row * head_dim + col];
    }

    float diff = fabsf(out[col] - expected);
    if (diff > max_diff) {
      max_diff = diff;
      max_index = col;
      max_expected = expected;
      max_actual = out[col];
    }
  }
  if (max_diff > 0.001f) {
    fprintf(stderr,
            "triton_value_aggregation keys=%u head_dim=%u mismatch at %u: "
            "actual=%f expected=%f diff=%f\n",
            keys, head_dim, max_index, max_actual, max_expected, max_diff);
    throw std::runtime_error("triton_value_aggregation result mismatch");
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
    run_case(device, 7, 64);
    run_case(device, 1536, 128);
    run_case(device, 3072, 192);

    printf("triton_value_aggregation: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
