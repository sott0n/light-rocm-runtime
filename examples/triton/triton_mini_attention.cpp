#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#ifndef LRRT_TRITON_MINI_ATTENTION_SCORE_MANIFEST
#define LRRT_TRITON_MINI_ATTENTION_SCORE_MANIFEST "manifest.json"
#endif

#ifndef LRRT_TRITON_MINI_CAUSAL_SOFTMAX_MANIFEST
#define LRRT_TRITON_MINI_CAUSAL_SOFTMAX_MANIFEST "manifest.json"
#endif

#ifndef LRRT_TRITON_MINI_VALUE_AGGREGATION_MANIFEST
#define LRRT_TRITON_MINI_VALUE_AGGREGATION_MANIFEST "manifest.json"
#endif

static uint32_t select_head_block(uint32_t head_dim) {
  if (head_dim <= 64) {
    return 64;
  }
  if (head_dim <= 128) {
    return 128;
  }
  return 256;
}

static uint32_t select_keys_block(uint32_t keys) {
  if (keys <= 1024) {
    return 1024;
  }
  if (keys <= 2048) {
    return 2048;
  }
  return 4096;
}

static void run_case(lrrt::Device &device, uint32_t keys, uint32_t head_dim,
                     uint32_t valid_keys) {
  uint32_t head_block = select_head_block(head_dim);
  uint32_t keys_block = select_keys_block(keys);
  std::string score_name = "attention_score_fp32_" + std::to_string(head_block);
  std::string softmax_name =
      "causal_softmax_fp32_" + std::to_string(keys_block);
  std::string aggregation_name =
      "value_aggregation_fp32_" + std::to_string(keys_block);

  lrrt::Bundle score_bundle(device, LRRT_TRITON_MINI_ATTENTION_SCORE_MANIFEST,
                            score_name.c_str());
  lrrt::Bundle softmax_bundle(device, LRRT_TRITON_MINI_CAUSAL_SOFTMAX_MANIFEST,
                              softmax_name.c_str());
  lrrt::Bundle aggregation_bundle(device,
                                  LRRT_TRITON_MINI_VALUE_AGGREGATION_MANIFEST,
                                  aggregation_name.c_str());

  std::vector<float> q(head_dim);
  std::vector<float> k(keys * head_dim);
  std::vector<float> v(keys * head_dim);
  std::vector<float> scores(keys, 0.0f);
  std::vector<float> probs(keys, 0.0f);
  std::vector<float> out(head_dim, 0.0f);
  const float scale = 1.0f / sqrtf((float)head_dim);

  for (uint32_t col = 0; col < head_dim; ++col) {
    q[col] = 0.03125f * (float)((int32_t)((col * 5) % 29) - 14);
  }
  for (uint32_t row = 0; row < keys; ++row) {
    for (uint32_t col = 0; col < head_dim; ++col) {
      uint32_t index = row * head_dim + col;
      int32_t k_lane = (int32_t)((index + row * 11 + col * 3) % 31) - 15;
      int32_t v_lane = (int32_t)((index + row * 13 + col * 5) % 37) - 18;
      k[index] = 0.015625f * (float)k_lane;
      v[index] = 0.015625f * (float)v_lane;
    }
  }

  lrrt::DeviceBuffer device_q(device, q.size() * sizeof(float));
  lrrt::DeviceBuffer device_k(device, k.size() * sizeof(float));
  lrrt::DeviceBuffer device_v(device, v.size() * sizeof(float));
  lrrt::DeviceBuffer device_scores(device, scores.size() * sizeof(float));
  lrrt::DeviceBuffer device_probs(device, probs.size() * sizeof(float));
  lrrt::DeviceBuffer device_out(device, out.size() * sizeof(float));
  lrrt::copy_to_device(device_q, q);
  lrrt::copy_to_device(device_k, k);
  lrrt::copy_to_device(device_v, v);
  lrrt::copy_to_device(device_scores, scores);
  lrrt::copy_to_device(device_probs, probs);

  lrrt::KernargBuffer score_args = score_bundle.make_args();
  score_args.set("q", (const float *)device_q.data());
  score_args.set("k", (const float *)device_k.data());
  score_args.set("out", (float *)device_scores.data());
  score_args.set("keys", (int32_t)keys);
  score_args.set("head_dim", (int32_t)head_dim);
  score_args.set("scale", scale);
  score_args.bind_optional_nulls();
  score_bundle.launch(keys, score_args);

  lrrt::KernargBuffer softmax_args = softmax_bundle.make_args();
  softmax_args.set("x", (const float *)device_scores.data());
  softmax_args.set("out", (float *)device_probs.data());
  softmax_args.set("rows", (int32_t)1);
  softmax_args.set("hidden", (int32_t)keys);
  softmax_args.set("query_start", (int32_t)(valid_keys - 1));
  softmax_args.bind_optional_nulls();
  softmax_bundle.launch(1, softmax_args);

  lrrt::KernargBuffer aggregation_args = aggregation_bundle.make_args();
  aggregation_args.set("probs", (const float *)device_probs.data());
  aggregation_args.set("v", (const float *)device_v.data());
  aggregation_args.set("out", (float *)device_out.data());
  aggregation_args.set("keys", (int32_t)keys);
  aggregation_args.set("head_dim", (int32_t)head_dim);
  aggregation_args.bind_optional_nulls();
  aggregation_bundle.launch(head_dim, aggregation_args);

  device.synchronize();
  lrrt::copy_to_host(out, device_out);

  std::vector<float> reference_scores(keys, 0.0f);
  for (uint32_t row = 0; row < keys; ++row) {
    for (uint32_t col = 0; col < head_dim; ++col) {
      reference_scores[row] += q[col] * k[row * head_dim + col];
    }
    reference_scores[row] *= scale;
  }

  float max_score = reference_scores[0];
  for (uint32_t row = 1; row < valid_keys; ++row) {
    max_score = fmaxf(max_score, reference_scores[row]);
  }
  float denominator = 0.0f;
  for (uint32_t row = 0; row < valid_keys; ++row) {
    denominator += expf(reference_scores[row] - max_score);
  }

  std::vector<float> reference_probs(keys, 0.0f);
  for (uint32_t row = 0; row < valid_keys; ++row) {
    reference_probs[row] =
        expf(reference_scores[row] - max_score) / denominator;
  }

  float max_diff = 0.0f;
  uint32_t max_index = 0;
  float max_expected = 0.0f;
  float max_actual = 0.0f;
  for (uint32_t col = 0; col < head_dim; ++col) {
    float expected = 0.0f;
    for (uint32_t row = 0; row < keys; ++row) {
      expected += reference_probs[row] * v[row * head_dim + col];
    }

    float diff = fabsf(out[col] - expected);
    if (diff > max_diff) {
      max_diff = diff;
      max_index = col;
      max_expected = expected;
      max_actual = out[col];
    }
  }
  if (max_diff > 0.002f) {
    fprintf(stderr,
            "triton_mini_attention keys=%u head_dim=%u valid_keys=%u "
            "mismatch at %u: actual=%f expected=%f diff=%f\n",
            keys, head_dim, valid_keys, max_index, max_actual, max_expected,
            max_diff);
    throw std::runtime_error("triton_mini_attention result mismatch");
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
    run_case(device, 16, 64, 7);
    run_case(device, 1536, 128, 1024);
    run_case(device, 3072, 192, 2049);

    printf("triton_mini_attention: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
