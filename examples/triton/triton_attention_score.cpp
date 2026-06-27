#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#ifndef LRRT_TRITON_ATTENTION_SCORE_MANIFEST
#define LRRT_TRITON_ATTENTION_SCORE_MANIFEST "manifest.json"
#endif

static uint32_t select_block_size(uint32_t head_dim) {
  if (head_dim <= 64) {
    return 64;
  }
  if (head_dim <= 128) {
    return 128;
  }
  return 256;
}

static void run_case(lrrt::Device &device, uint32_t keys, uint32_t head_dim) {
  uint32_t block_size = select_block_size(head_dim);
  std::string kernel_name =
      "attention_score_fp32_" + std::to_string(block_size);
  lrrt::Bundle bundle(device, LRRT_TRITON_ATTENTION_SCORE_MANIFEST,
                      kernel_name.c_str());

  std::vector<float> q(head_dim);
  std::vector<float> k(keys * head_dim);
  std::vector<float> out(keys, 0.0f);
  const float scale = 1.0f / sqrtf((float)head_dim);

  for (uint32_t i = 0; i < head_dim; ++i) {
    q[i] = 0.03125f * (float)((int32_t)((i * 5) % 29) - 14);
  }
  for (uint32_t row = 0; row < keys; ++row) {
    for (uint32_t col = 0; col < head_dim; ++col) {
      uint32_t index = row * head_dim + col;
      int32_t lane = (int32_t)((index + row * 11 + col * 3) % 31) - 15;
      k[index] = 0.015625f * (float)lane;
    }
  }

  lrrt::DeviceBuffer device_q(device, q.size() * sizeof(float));
  lrrt::DeviceBuffer device_k(device, k.size() * sizeof(float));
  lrrt::DeviceBuffer device_out(device, out.size() * sizeof(float));
  lrrt::copy_to_device(device_q, q);
  lrrt::copy_to_device(device_k, k);

  lrrt::KernargBuffer kernel_args = bundle.make_args();
  kernel_args.set("q", (const float *)device_q.data());
  kernel_args.set("k", (const float *)device_k.data());
  kernel_args.set("out", (float *)device_out.data());
  kernel_args.set("keys", (int32_t)keys);
  kernel_args.set("head_dim", (int32_t)head_dim);
  kernel_args.set("scale", scale);
  kernel_args.bind_optional_nulls();
  bundle.launch(keys, kernel_args);
  device.synchronize();
  lrrt::copy_to_host(out, device_out);

  float max_diff = 0.0f;
  uint32_t max_index = 0;
  float max_expected = 0.0f;
  float max_actual = 0.0f;
  for (uint32_t row = 0; row < keys; ++row) {
    float expected = 0.0f;
    for (uint32_t col = 0; col < head_dim; ++col) {
      expected += q[col] * k[row * head_dim + col];
    }
    expected *= scale;

    float diff = fabsf(out[row] - expected);
    if (diff > max_diff) {
      max_diff = diff;
      max_index = row;
      max_expected = expected;
      max_actual = out[row];
    }
  }
  if (max_diff > 0.001f) {
    fprintf(stderr,
            "triton_attention_score head_dim=%u mismatch at %u: actual=%f "
            "expected=%f diff=%f\n",
            head_dim, max_index, max_actual, max_expected, max_diff);
    throw std::runtime_error("triton_attention_score result mismatch");
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
    run_case(device, 17, 128);
    run_case(device, 33, 192);

    printf("triton_attention_score: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
