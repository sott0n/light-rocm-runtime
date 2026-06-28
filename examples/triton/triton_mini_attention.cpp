#include "triton_executor.hpp"

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

#ifndef LRRT_TRITON_MINI_MATVEC_MANIFEST
#define LRRT_TRITON_MINI_MATVEC_MANIFEST "manifest.json"
#endif

namespace tex = lrrt::executor::triton;

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

static uint32_t select_matvec_block(uint32_t hidden) {
  if (hidden <= 1024) {
    return 1024;
  }
  if (hidden <= 2048) {
    return 2048;
  }
  return 4096;
}

class MiniAttentionExecutor {
public:
  MiniAttentionExecutor(lrrt::Device &device, uint32_t keys, uint32_t hidden,
                        uint32_t head_dim)
      : queue_(device), bundles_(device), buffers_(device), keys_(keys),
        hidden_(hidden), head_dim_(head_dim),
        scale_(1.0f / sqrtf((float)head_dim)) {
    const std::string matvec_kernel_name = make_matvec_kernel_name(hidden);
    bundles_.add("q_projection", LRRT_TRITON_MINI_MATVEC_MANIFEST,
                 matvec_kernel_name);
    bundles_.add("k_projection", LRRT_TRITON_MINI_MATVEC_MANIFEST,
                 matvec_kernel_name);
    bundles_.add("v_projection", LRRT_TRITON_MINI_MATVEC_MANIFEST,
                 matvec_kernel_name);
    bundles_.add("score", LRRT_TRITON_MINI_ATTENTION_SCORE_MANIFEST,
                 make_score_kernel_name(head_dim));
    bundles_.add("softmax", LRRT_TRITON_MINI_CAUSAL_SOFTMAX_MANIFEST,
                 make_softmax_kernel_name(keys));
    bundles_.add("aggregation", LRRT_TRITON_MINI_VALUE_AGGREGATION_MANIFEST,
                 make_aggregation_kernel_name(keys));
    bundles_.add("residual_add", LRRT_TRITON_MINI_VECTOR_ADD_MANIFEST,
                 "vector_add");
    bundles_.add("kv_update", LRRT_TRITON_MINI_KV_CACHE_MANIFEST,
                 make_kv_update_kernel_name(head_dim));
    bundles_.add("rope", LRRT_TRITON_MINI_ROPE_MANIFEST,
                 make_rope_kernel_name(head_dim));

    buffers_.allocate<float>("hidden_states", keys * hidden);
    buffers_.allocate<float>("q_weight", head_dim * hidden);
    buffers_.allocate<float>("k_weight", head_dim * hidden);
    buffers_.allocate<float>("v_weight", head_dim * hidden);
    buffers_.allocate<float>("q", head_dim);
    buffers_.allocate<float>("q_rope", head_dim);
    buffers_.allocate<float>("source_k", keys * head_dim);
    buffers_.allocate<float>("source_v", keys * head_dim);
    buffers_.allocate<float>("k_rope", head_dim);
    buffers_.allocate<float>("k_cache", keys * head_dim);
    buffers_.allocate<float>("v_cache", keys * head_dim);
    buffers_.allocate<float>("cos", keys * (head_dim / 2));
    buffers_.allocate<float>("sin", keys * (head_dim / 2));
    buffers_.allocate<float>("residual", head_dim);
    buffers_.allocate<float>("scores", keys);
    buffers_.allocate<float>("probs", keys);
    buffers_.allocate<float>("attention_out", head_dim);
    buffers_.allocate<float>("out", head_dim);
  }

  void copy_inputs(const std::vector<float> &hidden_states,
                   const std::vector<float> &q_weight,
                   const std::vector<float> &k_weight,
                   const std::vector<float> &v_weight,
                   const std::vector<float> &cos, const std::vector<float> &sin,
                   const std::vector<float> &residual) {
    if (hidden_states.size() != keys_ * hidden_ ||
        q_weight.size() != head_dim_ * hidden_ ||
        k_weight.size() != head_dim_ * hidden_ ||
        v_weight.size() != head_dim_ * hidden_ ||
        cos.size() != keys_ * (head_dim_ / 2) ||
        sin.size() != keys_ * (head_dim_ / 2) || residual.size() != head_dim_) {
      throw std::runtime_error("mini attention input shape mismatch");
    }

    std::vector<float> vector_zero(head_dim_, 0.0f);
    std::vector<float> cache_zero(keys_ * head_dim_, 0.0f);
    std::vector<float> scores(keys_, 0.0f);
    std::vector<float> probs(keys_, 0.0f);
    buffers_.copy_to("hidden_states", hidden_states);
    buffers_.copy_to("q_weight", q_weight);
    buffers_.copy_to("k_weight", k_weight);
    buffers_.copy_to("v_weight", v_weight);
    buffers_.copy_to("q", vector_zero);
    buffers_.copy_to("source_k", cache_zero);
    buffers_.copy_to("source_v", cache_zero);
    buffers_.copy_to("cos", cos);
    buffers_.copy_to("sin", sin);
    buffers_.copy_to("k_cache", cache_zero);
    buffers_.copy_to("v_cache", cache_zero);
    buffers_.copy_to("residual", residual);
    buffers_.copy_to("scores", scores);
    buffers_.copy_to("probs", probs);
    buffers_.copy_to("attention_out", vector_zero);
    buffers_.copy_to("out", vector_zero);
  }

  void run(uint32_t valid_keys) {
    if (valid_keys == 0 || valid_keys > keys_) {
      throw std::runtime_error("mini attention valid_keys is out of range");
    }

    const uint32_t half = head_dim_ / 2;
    const uint32_t query_position = valid_keys - 1;

    tex::launch(
        queue_, bundles_.get("q_projection"), head_dim_,
        {
            tex::arg("x", buffers_.ptr<float>("hidden_states",
                                              query_position * hidden_)),
            tex::arg("weight", buffers_.ptr<float>("q_weight")),
            tex::arg("out", buffers_.ptr<float>("q")),
            tex::arg("outputs", (int32_t)head_dim_),
            tex::arg("hidden", (int32_t)hidden_),
        });
    for (uint32_t position = 0; position < valid_keys; ++position) {
      tex::launch(queue_, bundles_.get("k_projection"), head_dim_,
                  {
                      tex::arg("x", buffers_.ptr<float>("hidden_states",
                                                        position * hidden_)),
                      tex::arg("weight", buffers_.ptr<float>("k_weight")),
                      tex::arg("out", buffers_.ptr<float>(
                                          "source_k", position * head_dim_)),
                      tex::arg("outputs", (int32_t)head_dim_),
                      tex::arg("hidden", (int32_t)hidden_),
                  });
      tex::launch(queue_, bundles_.get("v_projection"), head_dim_,
                  {
                      tex::arg("x", buffers_.ptr<float>("hidden_states",
                                                        position * hidden_)),
                      tex::arg("weight", buffers_.ptr<float>("v_weight")),
                      tex::arg("out", buffers_.ptr<float>(
                                          "source_v", position * head_dim_)),
                      tex::arg("outputs", (int32_t)head_dim_),
                      tex::arg("hidden", (int32_t)hidden_),
                  });
    }

    tex::launch(
        queue_, bundles_.get("rope"), 1,
        {
            tex::arg("x", buffers_.ptr<float>("q")),
            tex::arg("cos", buffers_.ptr<float>("cos", query_position * half)),
            tex::arg("sin", buffers_.ptr<float>("sin", query_position * half)),
            tex::arg("out", buffers_.ptr<float>("q_rope")),
            tex::arg("rows", (int32_t)1),
            tex::arg("heads", (int32_t)1),
            tex::arg("head_dim", (int32_t)head_dim_),
        });

    for (uint32_t position = 0; position < valid_keys; ++position) {
      tex::launch(
          queue_, bundles_.get("rope"), 1,
          {
              tex::arg("x",
                       buffers_.ptr<float>("source_k", position * head_dim_)),
              tex::arg("cos", buffers_.ptr<float>("cos", position * half)),
              tex::arg("sin", buffers_.ptr<float>("sin", position * half)),
              tex::arg("out", buffers_.ptr<float>("k_rope")),
              tex::arg("rows", (int32_t)1),
              tex::arg("heads", (int32_t)1),
              tex::arg("head_dim", (int32_t)head_dim_),
          });

      tex::launch(queue_, bundles_.get("kv_update"), 1,
                  {
                      tex::arg("k", buffers_.ptr<float>("k_rope")),
                      tex::arg("v", buffers_.ptr<float>("source_v",
                                                        position * head_dim_)),
                      tex::arg("k_cache", buffers_.ptr<float>("k_cache")),
                      tex::arg("v_cache", buffers_.ptr<float>("v_cache")),
                      tex::arg("position", (int32_t)position),
                      tex::arg("max_tokens", (int32_t)keys_),
                      tex::arg("head_dim", (int32_t)head_dim_),
                  });
    }

    tex::launch(queue_, bundles_.get("score"), keys_,
                {
                    tex::arg("q", buffers_.ptr<float>("q_rope")),
                    tex::arg("k", buffers_.ptr<float>("k_cache")),
                    tex::arg("out", buffers_.ptr<float>("scores")),
                    tex::arg("keys", (int32_t)keys_),
                    tex::arg("head_dim", (int32_t)head_dim_),
                    tex::arg("scale", scale_),
                });

    tex::launch(queue_, bundles_.get("softmax"), 1,
                {
                    tex::arg("x", buffers_.ptr<float>("scores")),
                    tex::arg("out", buffers_.ptr<float>("probs")),
                    tex::arg("rows", (int32_t)1),
                    tex::arg("hidden", (int32_t)keys_),
                    tex::arg("query_start", (int32_t)(valid_keys - 1)),
                });

    tex::launch(queue_, bundles_.get("aggregation"), head_dim_,
                {
                    tex::arg("probs", buffers_.ptr<float>("probs")),
                    tex::arg("v", buffers_.ptr<float>("v_cache")),
                    tex::arg("out", buffers_.ptr<float>("attention_out")),
                    tex::arg("keys", (int32_t)keys_),
                    tex::arg("head_dim", (int32_t)head_dim_),
                });

    tex::launch(queue_, bundles_.get("residual_add"), head_dim_,
                {
                    tex::arg("x", buffers_.ptr<float>("residual")),
                    tex::arg("y", buffers_.ptr<float>("attention_out")),
                    tex::arg("out", buffers_.ptr<float>("out")),
                    tex::arg("n", (int32_t)head_dim_),
                });
  }

  void synchronize() const { queue_.synchronize(); }

  void copy_output(std::vector<float> &out) const {
    if (out.size() != head_dim_) {
      throw std::runtime_error("mini attention output shape mismatch");
    }
    buffers_.copy_from(out, "out");
  }

private:
  static std::string make_matvec_kernel_name(uint32_t hidden) {
    return "matvec_fp32_" + std::to_string(select_matvec_block(hidden));
  }

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
  tex::BundleSet bundles_;
  tex::BufferSet buffers_;
  uint32_t keys_;
  uint32_t hidden_;
  uint32_t head_dim_;
  float scale_;
};

static void fill_projection_weight(std::vector<float> &weight, uint32_t hidden,
                                   uint32_t seed) {
  for (uint32_t row = 0; row < weight.size() / hidden; ++row) {
    for (uint32_t col = 0; col < hidden; ++col) {
      uint32_t index = row * hidden + col;
      int32_t lane = (int32_t)((index + row * 7 + seed * 11) % 23) - 11;
      weight[index] = 0.0078125f * (float)lane;
    }
  }
}

static void reference_projection(const std::vector<float> &hidden_states,
                                 const std::vector<float> &weight,
                                 uint32_t hidden, uint32_t row,
                                 std::vector<float> &out) {
  for (uint32_t projection = 0; projection < out.size(); ++projection) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < hidden; ++col) {
      sum +=
          hidden_states[row * hidden + col] * weight[projection * hidden + col];
    }
    out[projection] = sum;
  }
}

static void run_case(lrrt::Device &device, uint32_t keys, uint32_t hidden,
                     uint32_t head_dim, uint32_t valid_keys) {
  std::vector<float> hidden_states(keys * hidden);
  std::vector<float> q_weight(head_dim * hidden);
  std::vector<float> k_weight(head_dim * hidden);
  std::vector<float> v_weight(head_dim * hidden);
  std::vector<float> q(head_dim);
  std::vector<float> k(keys * head_dim, 0.0f);
  std::vector<float> v(keys * head_dim, 0.0f);
  std::vector<float> cos(keys * (head_dim / 2));
  std::vector<float> sin(keys * (head_dim / 2));
  std::vector<float> residual(head_dim);
  std::vector<float> out(head_dim, 0.0f);
  const float scale = 1.0f / sqrtf((float)head_dim);
  const uint32_t half = head_dim / 2;

  for (uint32_t i = 0; i < hidden_states.size(); ++i) {
    hidden_states[i] =
        0.03125f * (float)((int32_t)((i * 5 + i / hidden) % 31) - 15);
  }
  fill_projection_weight(q_weight, hidden, 1);
  fill_projection_weight(k_weight, hidden, 2);
  fill_projection_weight(v_weight, hidden, 3);
  for (uint32_t col = 0; col < head_dim; ++col) {
    residual[col] = 0.0625f * (float)((int32_t)((col * 7) % 23) - 11);
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

  MiniAttentionExecutor executor(device, keys, hidden, head_dim);
  executor.copy_inputs(hidden_states, q_weight, k_weight, v_weight, cos, sin,
                       residual);
  executor.run(valid_keys);
  executor.synchronize();
  executor.copy_output(out);

  const uint32_t query_position = valid_keys - 1;
  reference_projection(hidden_states, q_weight, hidden, query_position, q);
  for (uint32_t row = 0; row < valid_keys; ++row) {
    std::vector<float> k_row(head_dim);
    std::vector<float> v_row(head_dim);
    reference_projection(hidden_states, k_weight, hidden, row, k_row);
    reference_projection(hidden_states, v_weight, hidden, row, v_row);
    for (uint32_t col = 0; col < head_dim; ++col) {
      k[row * head_dim + col] = k_row[col];
      v[row * head_dim + col] = v_row[col];
    }
  }

  std::vector<float> q_rope(head_dim);
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
            "triton_mini_attention keys=%u hidden=%u head_dim=%u valid_keys=%u "
            "mismatch at %u: actual=%f expected=%f diff=%f\n",
            keys, hidden, head_dim, valid_keys, max_index, max_actual,
            max_expected, max_diff);
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
    run_case(device, 16, 768, 64, 7);
    run_case(device, 96, 1024, 128, 64);
    run_case(device, 128, 1536, 192, 97);

    printf("triton_mini_attention: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
