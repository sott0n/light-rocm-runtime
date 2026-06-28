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

#ifndef LRRT_TRITON_MINI_VECTOR_ADD_MANIFEST
#define LRRT_TRITON_MINI_VECTOR_ADD_MANIFEST "manifest.json"
#endif

#ifndef LRRT_TRITON_MINI_KV_CACHE_MANIFEST
#define LRRT_TRITON_MINI_KV_CACHE_MANIFEST "manifest.json"
#endif

#ifndef LRRT_TRITON_MINI_ROPE_MANIFEST
#define LRRT_TRITON_MINI_ROPE_MANIFEST "manifest.json"
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

class MiniAttentionExecutor {
public:
  MiniAttentionExecutor(lrrt::Device &device, uint32_t keys, uint32_t head_dim)
      : queue_(device), keys_(keys), head_dim_(head_dim),
        scale_(1.0f / sqrtf((float)head_dim)),
        score_kernel_name_(make_score_kernel_name(head_dim)),
        softmax_kernel_name_(make_softmax_kernel_name(keys)),
        aggregation_kernel_name_(make_aggregation_kernel_name(keys)),
        kv_update_kernel_name_(make_kv_update_kernel_name(head_dim)),
        rope_kernel_name_(make_rope_kernel_name(head_dim)),
        score_bundle_(device, LRRT_TRITON_MINI_ATTENTION_SCORE_MANIFEST,
                      score_kernel_name_.c_str()),
        softmax_bundle_(device, LRRT_TRITON_MINI_CAUSAL_SOFTMAX_MANIFEST,
                        softmax_kernel_name_.c_str()),
        aggregation_bundle_(device, LRRT_TRITON_MINI_VALUE_AGGREGATION_MANIFEST,
                            aggregation_kernel_name_.c_str()),
        residual_add_bundle_(device, LRRT_TRITON_MINI_VECTOR_ADD_MANIFEST,
                             "vector_add"),
        kv_update_bundle_(device, LRRT_TRITON_MINI_KV_CACHE_MANIFEST,
                          kv_update_kernel_name_.c_str()),
        rope_bundle_(device, LRRT_TRITON_MINI_ROPE_MANIFEST,
                     rope_kernel_name_.c_str()),
        q_(device, head_dim * sizeof(float)),
        q_rope_(device, head_dim * sizeof(float)),
        source_k_(device, keys * head_dim * sizeof(float)),
        source_v_(device, keys * head_dim * sizeof(float)),
        k_rope_(device, head_dim * sizeof(float)),
        k_cache_(device, keys * head_dim * sizeof(float)),
        v_cache_(device, keys * head_dim * sizeof(float)),
        cos_(device, keys * (head_dim / 2) * sizeof(float)),
        sin_(device, keys * (head_dim / 2) * sizeof(float)),
        residual_(device, head_dim * sizeof(float)),
        scores_(device, keys * sizeof(float)),
        probs_(device, keys * sizeof(float)),
        attention_out_(device, head_dim * sizeof(float)),
        out_(device, head_dim * sizeof(float)) {}

  void copy_inputs(const std::vector<float> &q, const std::vector<float> &k,
                   const std::vector<float> &v, const std::vector<float> &cos,
                   const std::vector<float> &sin,
                   const std::vector<float> &residual) {
    if (q.size() != head_dim_ || k.size() != keys_ * head_dim_ ||
        v.size() != keys_ * head_dim_ ||
        cos.size() != keys_ * (head_dim_ / 2) ||
        sin.size() != keys_ * (head_dim_ / 2) || residual.size() != head_dim_) {
      throw std::runtime_error("mini attention input shape mismatch");
    }

    std::vector<float> cache(keys_ * head_dim_, 0.0f);
    std::vector<float> scores(keys_, 0.0f);
    std::vector<float> probs(keys_, 0.0f);
    std::vector<float> out(head_dim_, 0.0f);
    lrrt::copy_to_device(q_, q);
    lrrt::copy_to_device(source_k_, k);
    lrrt::copy_to_device(source_v_, v);
    lrrt::copy_to_device(cos_, cos);
    lrrt::copy_to_device(sin_, sin);
    lrrt::copy_to_device(k_cache_, cache);
    lrrt::copy_to_device(v_cache_, cache);
    lrrt::copy_to_device(residual_, residual);
    lrrt::copy_to_device(scores_, scores);
    lrrt::copy_to_device(probs_, probs);
    lrrt::copy_to_device(attention_out_, out);
    lrrt::copy_to_device(out_, out);
  }

  void run(uint32_t valid_keys) {
    if (valid_keys == 0 || valid_keys > keys_) {
      throw std::runtime_error("mini attention valid_keys is out of range");
    }

    const uint32_t half = head_dim_ / 2;
    const uint32_t query_position = valid_keys - 1;
    lrrt::KernargBuffer q_rope_args = rope_bundle_.make_args();
    q_rope_args.set("x", (const float *)q_.data());
    q_rope_args.set("cos", (const float *)cos_.data() + query_position * half);
    q_rope_args.set("sin", (const float *)sin_.data() + query_position * half);
    q_rope_args.set("out", (float *)q_rope_.data());
    q_rope_args.set("rows", (int32_t)1);
    q_rope_args.set("heads", (int32_t)1);
    q_rope_args.set("head_dim", (int32_t)head_dim_);
    q_rope_args.bind_optional_nulls();
    rope_bundle_.launch(queue_, 1, q_rope_args);

    for (uint32_t position = 0; position < valid_keys; ++position) {
      lrrt::KernargBuffer k_rope_args = rope_bundle_.make_args();
      const float *source_k =
          (const float *)source_k_.data() + position * head_dim_;
      const float *source_cos = (const float *)cos_.data() + position * half;
      const float *source_sin = (const float *)sin_.data() + position * half;
      k_rope_args.set("x", source_k);
      k_rope_args.set("cos", source_cos);
      k_rope_args.set("sin", source_sin);
      k_rope_args.set("out", (float *)k_rope_.data());
      k_rope_args.set("rows", (int32_t)1);
      k_rope_args.set("heads", (int32_t)1);
      k_rope_args.set("head_dim", (int32_t)head_dim_);
      k_rope_args.bind_optional_nulls();
      rope_bundle_.launch(queue_, 1, k_rope_args);

      lrrt::KernargBuffer kv_update_args = kv_update_bundle_.make_args();
      const float *source_v =
          (const float *)source_v_.data() + position * head_dim_;
      kv_update_args.set("k", (const float *)k_rope_.data());
      kv_update_args.set("v", source_v);
      kv_update_args.set("k_cache", (float *)k_cache_.data());
      kv_update_args.set("v_cache", (float *)v_cache_.data());
      kv_update_args.set("position", (int32_t)position);
      kv_update_args.set("max_tokens", (int32_t)keys_);
      kv_update_args.set("head_dim", (int32_t)head_dim_);
      kv_update_args.bind_optional_nulls();
      kv_update_bundle_.launch(queue_, 1, kv_update_args);
    }

    lrrt::KernargBuffer score_args = score_bundle_.make_args();
    score_args.set("q", (const float *)q_rope_.data());
    score_args.set("k", (const float *)k_cache_.data());
    score_args.set("out", (float *)scores_.data());
    score_args.set("keys", (int32_t)keys_);
    score_args.set("head_dim", (int32_t)head_dim_);
    score_args.set("scale", scale_);
    score_args.bind_optional_nulls();
    score_bundle_.launch(queue_, keys_, score_args);

    lrrt::KernargBuffer softmax_args = softmax_bundle_.make_args();
    softmax_args.set("x", (const float *)scores_.data());
    softmax_args.set("out", (float *)probs_.data());
    softmax_args.set("rows", (int32_t)1);
    softmax_args.set("hidden", (int32_t)keys_);
    softmax_args.set("query_start", (int32_t)(valid_keys - 1));
    softmax_args.bind_optional_nulls();
    softmax_bundle_.launch(queue_, 1, softmax_args);

    lrrt::KernargBuffer aggregation_args = aggregation_bundle_.make_args();
    aggregation_args.set("probs", (const float *)probs_.data());
    aggregation_args.set("v", (const float *)v_cache_.data());
    aggregation_args.set("out", (float *)attention_out_.data());
    aggregation_args.set("keys", (int32_t)keys_);
    aggregation_args.set("head_dim", (int32_t)head_dim_);
    aggregation_args.bind_optional_nulls();
    aggregation_bundle_.launch(queue_, head_dim_, aggregation_args);

    lrrt::KernargBuffer residual_add_args = residual_add_bundle_.make_args();
    residual_add_args.set("x", (const float *)residual_.data());
    residual_add_args.set("y", (const float *)attention_out_.data());
    residual_add_args.set("out", (float *)out_.data());
    residual_add_args.set("n", (int32_t)head_dim_);
    residual_add_args.bind_optional_nulls();
    residual_add_bundle_.launch(queue_, head_dim_, residual_add_args);
  }

  void synchronize() const { queue_.synchronize(); }

  void copy_output(std::vector<float> &out) const {
    if (out.size() != head_dim_) {
      throw std::runtime_error("mini attention output shape mismatch");
    }
    lrrt::copy_to_host(out, out_);
  }

private:
  static std::string make_score_kernel_name(uint32_t head_dim) {
    return "attention_score_fp32_" +
           std::to_string(select_head_block(head_dim));
  }

  static std::string make_softmax_kernel_name(uint32_t keys) {
    return "causal_softmax_fp32_" + std::to_string(select_keys_block(keys));
  }

  static std::string make_aggregation_kernel_name(uint32_t keys) {
    return "value_aggregation_fp32_" + std::to_string(select_keys_block(keys));
  }

  static std::string make_kv_update_kernel_name(uint32_t head_dim) {
    return "kv_cache_update_fp32_" +
           std::to_string(select_head_block(head_dim));
  }

  static std::string make_rope_kernel_name(uint32_t head_dim) {
    return "rope_" + std::to_string(select_head_block(head_dim));
  }

  lrrt::Queue queue_;
  uint32_t keys_;
  uint32_t head_dim_;
  float scale_;
  std::string score_kernel_name_;
  std::string softmax_kernel_name_;
  std::string aggregation_kernel_name_;
  std::string kv_update_kernel_name_;
  std::string rope_kernel_name_;
  lrrt::Bundle score_bundle_;
  lrrt::Bundle softmax_bundle_;
  lrrt::Bundle aggregation_bundle_;
  lrrt::Bundle residual_add_bundle_;
  lrrt::Bundle kv_update_bundle_;
  lrrt::Bundle rope_bundle_;
  lrrt::DeviceBuffer q_;
  lrrt::DeviceBuffer q_rope_;
  lrrt::DeviceBuffer source_k_;
  lrrt::DeviceBuffer source_v_;
  lrrt::DeviceBuffer k_rope_;
  lrrt::DeviceBuffer k_cache_;
  lrrt::DeviceBuffer v_cache_;
  lrrt::DeviceBuffer cos_;
  lrrt::DeviceBuffer sin_;
  lrrt::DeviceBuffer residual_;
  lrrt::DeviceBuffer scores_;
  lrrt::DeviceBuffer probs_;
  lrrt::DeviceBuffer attention_out_;
  lrrt::DeviceBuffer out_;
};

static void run_case(lrrt::Device &device, uint32_t keys, uint32_t head_dim,
                     uint32_t valid_keys) {
  std::vector<float> q(head_dim);
  std::vector<float> k(keys * head_dim);
  std::vector<float> v(keys * head_dim);
  std::vector<float> cos(keys * (head_dim / 2));
  std::vector<float> sin(keys * (head_dim / 2));
  std::vector<float> residual(head_dim);
  std::vector<float> out(head_dim, 0.0f);
  const float scale = 1.0f / sqrtf((float)head_dim);
  const uint32_t half = head_dim / 2;

  for (uint32_t col = 0; col < head_dim; ++col) {
    q[col] = 0.03125f * (float)((int32_t)((col * 5) % 29) - 14);
    residual[col] = 0.0625f * (float)((int32_t)((col * 7) % 23) - 11);
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
  for (uint32_t token = 0; token < keys; ++token) {
    for (uint32_t frequency = 0; frequency < half; ++frequency) {
      float exponent = -2.0f * (float)frequency / (float)head_dim;
      float inverse_frequency = powf(10000.0f, exponent);
      float angle = (float)(token + 7) * inverse_frequency;
      uint32_t index = token * half + frequency;
      cos[index] = cosf(angle);
      sin[index] = sinf(angle);
    }
  }

  MiniAttentionExecutor executor(device, keys, head_dim);
  executor.copy_inputs(q, k, v, cos, sin, residual);
  executor.run(valid_keys);
  executor.synchronize();
  executor.copy_output(out);

  std::vector<float> q_rope(head_dim);
  uint32_t query_position = valid_keys - 1;
  for (uint32_t offset = 0; offset < head_dim; ++offset) {
    uint32_t frequency = offset % half;
    uint32_t partner = offset < half ? offset + half : offset - half;
    float rotated = offset < half ? -q[partner] : q[partner];
    q_rope[offset] = q[offset] * cos[query_position * half + frequency] +
                     rotated * sin[query_position * half + frequency];
  }

  std::vector<float> reference_scores(keys, 0.0f);
  for (uint32_t row = 0; row < valid_keys; ++row) {
    for (uint32_t col = 0; col < head_dim; ++col) {
      uint32_t frequency = col % half;
      uint32_t partner = col < half ? col + half : col - half;
      uint32_t base = row * head_dim;
      float rotated = col < half ? -k[base + partner] : k[base + partner];
      float k_rope = k[base + col] * cos[row * half + frequency] +
                     rotated * sin[row * half + frequency];
      reference_scores[row] += q_rope[col] * k_rope;
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
    expected += residual[col];

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
    run_case(device, 96, 128, 64);
    run_case(device, 128, 192, 97);

    printf("triton_mini_attention: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
