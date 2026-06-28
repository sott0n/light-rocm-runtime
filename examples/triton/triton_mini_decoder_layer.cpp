#include "triton_executor.hpp"

#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#ifndef LRRT_TRITON_MINI_LAYER_RMSNORM_MANIFEST
#define LRRT_TRITON_MINI_LAYER_RMSNORM_MANIFEST "manifest.json"
#endif

#ifndef LRRT_TRITON_MINI_LAYER_MATVEC_MANIFEST
#define LRRT_TRITON_MINI_LAYER_MATVEC_MANIFEST "manifest.json"
#endif

#ifndef LRRT_TRITON_MINI_LAYER_ATTENTION_SCORE_MANIFEST
#define LRRT_TRITON_MINI_LAYER_ATTENTION_SCORE_MANIFEST "manifest.json"
#endif

#ifndef LRRT_TRITON_MINI_LAYER_CAUSAL_SOFTMAX_MANIFEST
#define LRRT_TRITON_MINI_LAYER_CAUSAL_SOFTMAX_MANIFEST "manifest.json"
#endif

#ifndef LRRT_TRITON_MINI_LAYER_VALUE_AGGREGATION_MANIFEST
#define LRRT_TRITON_MINI_LAYER_VALUE_AGGREGATION_MANIFEST "manifest.json"
#endif

#ifndef LRRT_TRITON_MINI_LAYER_VECTOR_ADD_MANIFEST
#define LRRT_TRITON_MINI_LAYER_VECTOR_ADD_MANIFEST "manifest.json"
#endif

#ifndef LRRT_TRITON_MINI_LAYER_KV_CACHE_MANIFEST
#define LRRT_TRITON_MINI_LAYER_KV_CACHE_MANIFEST "manifest.json"
#endif

#ifndef LRRT_TRITON_MINI_LAYER_ROPE_MANIFEST
#define LRRT_TRITON_MINI_LAYER_ROPE_MANIFEST "manifest.json"
#endif

#ifndef LRRT_TRITON_MINI_LAYER_SILU_MUL_MANIFEST
#define LRRT_TRITON_MINI_LAYER_SILU_MUL_MANIFEST "manifest.json"
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

static uint32_t select_block(uint32_t size) {
  if (size <= 1024) {
    return 1024;
  }
  if (size <= 2048) {
    return 2048;
  }
  return 4096;
}

class MiniDecoderLayerExecutor {
public:
  MiniDecoderLayerExecutor(lrrt::Device &device, uint32_t keys, uint32_t hidden,
                           uint32_t head_dim, uint32_t intermediate)
      : queue_(device), bundles_(device), buffers_(device), keys_(keys),
        hidden_(hidden), head_dim_(head_dim), intermediate_(intermediate),
        scale_(1.0f / sqrtf((float)head_dim)) {
    bundles_.add("attention_norm", LRRT_TRITON_MINI_LAYER_RMSNORM_MANIFEST,
                 "rmsnorm_fp32_" + std::to_string(select_block(hidden)));
    bundles_.add("mlp_norm", LRRT_TRITON_MINI_LAYER_RMSNORM_MANIFEST,
                 "rmsnorm_fp32_" + std::to_string(select_block(hidden)));
    bundles_.add("q_projection", LRRT_TRITON_MINI_LAYER_MATVEC_MANIFEST,
                 "matvec_fp32_" + std::to_string(select_block(hidden)));
    bundles_.add("k_projection", LRRT_TRITON_MINI_LAYER_MATVEC_MANIFEST,
                 "matvec_fp32_" + std::to_string(select_block(hidden)));
    bundles_.add("v_projection", LRRT_TRITON_MINI_LAYER_MATVEC_MANIFEST,
                 "matvec_fp32_" + std::to_string(select_block(hidden)));
    bundles_.add("out_projection", LRRT_TRITON_MINI_LAYER_MATVEC_MANIFEST,
                 "matvec_fp32_" + std::to_string(select_block(head_dim)));
    bundles_.add("gate_projection", LRRT_TRITON_MINI_LAYER_MATVEC_MANIFEST,
                 "matvec_fp32_" + std::to_string(select_block(hidden)));
    bundles_.add("up_projection", LRRT_TRITON_MINI_LAYER_MATVEC_MANIFEST,
                 "matvec_fp32_" + std::to_string(select_block(hidden)));
    bundles_.add("down_projection", LRRT_TRITON_MINI_LAYER_MATVEC_MANIFEST,
                 "matvec_fp32_" + std::to_string(select_block(intermediate)));
    bundles_.add("score", LRRT_TRITON_MINI_LAYER_ATTENTION_SCORE_MANIFEST,
                 "attention_score_fp32_" +
                     std::to_string(select_head_block(head_dim)));
    bundles_.add("softmax", LRRT_TRITON_MINI_LAYER_CAUSAL_SOFTMAX_MANIFEST,
                 "causal_softmax_fp32_" + std::to_string(select_block(keys)));
    bundles_.add(
        "aggregation", LRRT_TRITON_MINI_LAYER_VALUE_AGGREGATION_MANIFEST,
        "value_aggregation_fp32_" + std::to_string(select_block(keys)));
    bundles_.add("kv_update", LRRT_TRITON_MINI_LAYER_KV_CACHE_MANIFEST,
                 "kv_cache_update_fp32_" +
                     std::to_string(select_head_block(head_dim)));
    bundles_.add("rope", LRRT_TRITON_MINI_LAYER_ROPE_MANIFEST,
                 "rope_" + std::to_string(select_head_block(head_dim)));
    bundles_.add("silu_mul", LRRT_TRITON_MINI_LAYER_SILU_MUL_MANIFEST,
                 "silu_mul");
    bundles_.add("residual_add", LRRT_TRITON_MINI_LAYER_VECTOR_ADD_MANIFEST,
                 "vector_add");

    buffers_.allocate<float>("hidden_states", keys * hidden);
    buffers_.allocate<float>("attention_norm_weight", hidden);
    buffers_.allocate<float>("mlp_norm_weight", hidden);
    buffers_.allocate<float>("attention_norm_states", keys * hidden);
    buffers_.allocate<float>("mlp_norm_hidden", hidden);
    buffers_.allocate<float>("q_weight", head_dim * hidden);
    buffers_.allocate<float>("k_weight", head_dim * hidden);
    buffers_.allocate<float>("v_weight", head_dim * hidden);
    buffers_.allocate<float>("out_weight", hidden * head_dim);
    buffers_.allocate<float>("gate_weight", intermediate * hidden);
    buffers_.allocate<float>("up_weight", intermediate * hidden);
    buffers_.allocate<float>("down_weight", hidden * intermediate);
    buffers_.allocate<float>("q", head_dim);
    buffers_.allocate<float>("q_rope", head_dim);
    buffers_.allocate<float>("source_k", keys * head_dim);
    buffers_.allocate<float>("source_v", keys * head_dim);
    buffers_.allocate<float>("k_rope", head_dim);
    buffers_.allocate<float>("k_cache", keys * head_dim);
    buffers_.allocate<float>("v_cache", keys * head_dim);
    buffers_.allocate<float>("cos", keys * (head_dim / 2));
    buffers_.allocate<float>("sin", keys * (head_dim / 2));
    buffers_.allocate<float>("scores", keys);
    buffers_.allocate<float>("probs", keys);
    buffers_.allocate<float>("attention_out", head_dim);
    buffers_.allocate<float>("projected_attention", hidden);
    buffers_.allocate<float>("attention_residual", hidden);
    buffers_.allocate<float>("gate", intermediate);
    buffers_.allocate<float>("up", intermediate);
    buffers_.allocate<float>("activated", intermediate);
    buffers_.allocate<float>("projected_mlp", hidden);
    buffers_.allocate<float>("out", hidden);
  }

  void copy_inputs(const std::vector<float> &hidden_states,
                   const std::vector<float> &attention_norm_weight,
                   const std::vector<float> &mlp_norm_weight,
                   const std::vector<float> &q_weight,
                   const std::vector<float> &k_weight,
                   const std::vector<float> &v_weight,
                   const std::vector<float> &out_weight,
                   const std::vector<float> &gate_weight,
                   const std::vector<float> &up_weight,
                   const std::vector<float> &down_weight,
                   const std::vector<float> &cos,
                   const std::vector<float> &sin) {
    if (hidden_states.size() != keys_ * hidden_ ||
        attention_norm_weight.size() != hidden_ ||
        mlp_norm_weight.size() != hidden_ ||
        q_weight.size() != head_dim_ * hidden_ ||
        k_weight.size() != head_dim_ * hidden_ ||
        v_weight.size() != head_dim_ * hidden_ ||
        out_weight.size() != hidden_ * head_dim_ ||
        gate_weight.size() != intermediate_ * hidden_ ||
        up_weight.size() != intermediate_ * hidden_ ||
        down_weight.size() != hidden_ * intermediate_ ||
        cos.size() != keys_ * (head_dim_ / 2) ||
        sin.size() != keys_ * (head_dim_ / 2)) {
      throw std::runtime_error("mini decoder layer input shape mismatch");
    }

    std::vector<float> hidden_zero(hidden_, 0.0f);
    std::vector<float> hidden_cache_zero(keys_ * hidden_, 0.0f);
    std::vector<float> head_zero(head_dim_, 0.0f);
    std::vector<float> head_cache_zero(keys_ * head_dim_, 0.0f);
    std::vector<float> intermediate_zero(intermediate_, 0.0f);
    std::vector<float> scores_zero(keys_, 0.0f);

    buffers_.copy_to("hidden_states", hidden_states);
    buffers_.copy_to("attention_norm_weight", attention_norm_weight);
    buffers_.copy_to("mlp_norm_weight", mlp_norm_weight);
    buffers_.copy_to("attention_norm_states", hidden_cache_zero);
    buffers_.copy_to("mlp_norm_hidden", hidden_zero);
    buffers_.copy_to("q_weight", q_weight);
    buffers_.copy_to("k_weight", k_weight);
    buffers_.copy_to("v_weight", v_weight);
    buffers_.copy_to("out_weight", out_weight);
    buffers_.copy_to("gate_weight", gate_weight);
    buffers_.copy_to("up_weight", up_weight);
    buffers_.copy_to("down_weight", down_weight);
    buffers_.copy_to("q", head_zero);
    buffers_.copy_to("q_rope", head_zero);
    buffers_.copy_to("source_k", head_cache_zero);
    buffers_.copy_to("source_v", head_cache_zero);
    buffers_.copy_to("k_rope", head_zero);
    buffers_.copy_to("k_cache", head_cache_zero);
    buffers_.copy_to("v_cache", head_cache_zero);
    buffers_.copy_to("cos", cos);
    buffers_.copy_to("sin", sin);
    buffers_.copy_to("scores", scores_zero);
    buffers_.copy_to("probs", scores_zero);
    buffers_.copy_to("attention_out", head_zero);
    buffers_.copy_to("projected_attention", hidden_zero);
    buffers_.copy_to("attention_residual", hidden_zero);
    buffers_.copy_to("gate", intermediate_zero);
    buffers_.copy_to("up", intermediate_zero);
    buffers_.copy_to("activated", intermediate_zero);
    buffers_.copy_to("projected_mlp", hidden_zero);
    buffers_.copy_to("out", hidden_zero);
  }

  void run(uint32_t valid_keys) {
    if (valid_keys == 0 || valid_keys > keys_) {
      throw std::runtime_error("mini decoder layer valid_keys is out of range");
    }

    const uint32_t half = head_dim_ / 2;
    const uint32_t query_position = valid_keys - 1;
    tex::launch(
        queue_, bundles_.get("attention_norm"), keys_,
        {
            tex::arg("x", buffers_.ptr<float>("hidden_states")),
            tex::arg("weight", buffers_.ptr<float>("attention_norm_weight")),
            tex::arg("out", buffers_.ptr<float>("attention_norm_states")),
            tex::arg("eps", 1.0e-5f),
            tex::arg("rows", (int32_t)keys_),
            tex::arg("hidden", (int32_t)hidden_),
        });

    tex::launch(
        queue_, bundles_.get("q_projection"), head_dim_,
        {
            tex::arg("x", buffers_.ptr<float>("attention_norm_states",
                                              query_position * hidden_)),
            tex::arg("weight", buffers_.ptr<float>("q_weight")),
            tex::arg("out", buffers_.ptr<float>("q")),
            tex::arg("outputs", (int32_t)head_dim_),
            tex::arg("hidden", (int32_t)hidden_),
        });
    for (uint32_t position = 0; position < valid_keys; ++position) {
      tex::launch(queue_, bundles_.get("k_projection"), head_dim_,
                  {
                      tex::arg("x", buffers_.ptr<float>("attention_norm_states",
                                                        position * hidden_)),
                      tex::arg("weight", buffers_.ptr<float>("k_weight")),
                      tex::arg("out", buffers_.ptr<float>(
                                          "source_k", position * head_dim_)),
                      tex::arg("outputs", (int32_t)head_dim_),
                      tex::arg("hidden", (int32_t)hidden_),
                  });
      tex::launch(queue_, bundles_.get("v_projection"), head_dim_,
                  {
                      tex::arg("x", buffers_.ptr<float>("attention_norm_states",
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
    tex::launch(queue_, bundles_.get("out_projection"), hidden_,
                {
                    tex::arg("x", buffers_.ptr<float>("attention_out")),
                    tex::arg("weight", buffers_.ptr<float>("out_weight")),
                    tex::arg("out", buffers_.ptr<float>("projected_attention")),
                    tex::arg("outputs", (int32_t)hidden_),
                    tex::arg("hidden", (int32_t)head_dim_),
                });
    tex::launch(
        queue_, bundles_.get("residual_add"), hidden_,
        {
            tex::arg("x", buffers_.ptr<float>("hidden_states",
                                              query_position * hidden_)),
            tex::arg("y", buffers_.ptr<float>("projected_attention")),
            tex::arg("out", buffers_.ptr<float>("attention_residual")),
            tex::arg("n", (int32_t)hidden_),
        });

    tex::launch(queue_, bundles_.get("mlp_norm"), 1,
                {
                    tex::arg("x", buffers_.ptr<float>("attention_residual")),
                    tex::arg("weight", buffers_.ptr<float>("mlp_norm_weight")),
                    tex::arg("out", buffers_.ptr<float>("mlp_norm_hidden")),
                    tex::arg("eps", 1.0e-5f),
                    tex::arg("rows", (int32_t)1),
                    tex::arg("hidden", (int32_t)hidden_),
                });
    tex::launch(queue_, bundles_.get("gate_projection"), intermediate_,
                {
                    tex::arg("x", buffers_.ptr<float>("mlp_norm_hidden")),
                    tex::arg("weight", buffers_.ptr<float>("gate_weight")),
                    tex::arg("out", buffers_.ptr<float>("gate")),
                    tex::arg("outputs", (int32_t)intermediate_),
                    tex::arg("hidden", (int32_t)hidden_),
                });
    tex::launch(queue_, bundles_.get("up_projection"), intermediate_,
                {
                    tex::arg("x", buffers_.ptr<float>("mlp_norm_hidden")),
                    tex::arg("weight", buffers_.ptr<float>("up_weight")),
                    tex::arg("out", buffers_.ptr<float>("up")),
                    tex::arg("outputs", (int32_t)intermediate_),
                    tex::arg("hidden", (int32_t)hidden_),
                });
    tex::launch(queue_, bundles_.get("silu_mul"), intermediate_,
                {
                    tex::arg("gate", buffers_.ptr<float>("gate")),
                    tex::arg("up", buffers_.ptr<float>("up")),
                    tex::arg("out", buffers_.ptr<float>("activated")),
                    tex::arg("n", (int32_t)intermediate_),
                });
    tex::launch(queue_, bundles_.get("down_projection"), hidden_,
                {
                    tex::arg("x", buffers_.ptr<float>("activated")),
                    tex::arg("weight", buffers_.ptr<float>("down_weight")),
                    tex::arg("out", buffers_.ptr<float>("projected_mlp")),
                    tex::arg("outputs", (int32_t)hidden_),
                    tex::arg("hidden", (int32_t)intermediate_),
                });
    tex::launch(queue_, bundles_.get("residual_add"), hidden_,
                {
                    tex::arg("x", buffers_.ptr<float>("attention_residual")),
                    tex::arg("y", buffers_.ptr<float>("projected_mlp")),
                    tex::arg("out", buffers_.ptr<float>("out")),
                    tex::arg("n", (int32_t)hidden_),
                });
  }

  void synchronize() const { queue_.synchronize(); }

  void copy_output(std::vector<float> &out) const {
    if (out.size() != hidden_) {
      throw std::runtime_error("mini decoder layer output shape mismatch");
    }
    buffers_.copy_from(out, "out");
  }

private:
  lrrt::Queue queue_;
  tex::BundleSet bundles_;
  tex::BufferSet buffers_;
  uint32_t keys_;
  uint32_t hidden_;
  uint32_t head_dim_;
  uint32_t intermediate_;
  float scale_;
};

static void fill_projection_weight(std::vector<float> &weight, uint32_t input,
                                   uint32_t seed) {
  for (uint32_t row = 0; row < weight.size() / input; ++row) {
    for (uint32_t col = 0; col < input; ++col) {
      uint32_t index = row * input + col;
      int32_t lane = (int32_t)((index + row * 7 + seed * 11) % 23) - 11;
      weight[index] = 0.0078125f * (float)lane;
    }
  }
}

static void reference_rmsnorm(const std::vector<float> &x,
                              const std::vector<float> &weight, uint32_t rows,
                              uint32_t hidden, std::vector<float> &out) {
  for (uint32_t row = 0; row < rows; ++row) {
    float sum_square = 0.0f;
    for (uint32_t col = 0; col < hidden; ++col) {
      float value = x[row * hidden + col];
      sum_square += value * value;
    }
    float scale = 1.0f / sqrtf(sum_square / (float)hidden + 1.0e-5f);
    for (uint32_t col = 0; col < hidden; ++col) {
      uint32_t index = row * hidden + col;
      out[index] = x[index] * scale * weight[col];
    }
  }
}

static void reference_rmsnorm_vector(const std::vector<float> &x,
                                     const std::vector<float> &weight,
                                     std::vector<float> &out) {
  float sum_square = 0.0f;
  for (float value : x) {
    sum_square += value * value;
  }
  float scale = 1.0f / sqrtf(sum_square / (float)x.size() + 1.0e-5f);
  for (uint32_t i = 0; i < x.size(); ++i) {
    out[i] = x[i] * scale * weight[i];
  }
}

static void reference_projection(const std::vector<float> &x,
                                 const std::vector<float> &weight,
                                 uint32_t input, uint32_t row,
                                 std::vector<float> &out) {
  for (uint32_t projection = 0; projection < out.size(); ++projection) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < input; ++col) {
      sum += x[row * input + col] * weight[projection * input + col];
    }
    out[projection] = sum;
  }
}

static void reference_vector_projection(const std::vector<float> &x,
                                        const std::vector<float> &weight,
                                        uint32_t input,
                                        std::vector<float> &out) {
  for (uint32_t projection = 0; projection < out.size(); ++projection) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < input; ++col) {
      sum += x[col] * weight[projection * input + col];
    }
    out[projection] = sum;
  }
}

static void run_case(lrrt::Device &device, uint32_t keys, uint32_t hidden,
                     uint32_t head_dim, uint32_t intermediate,
                     uint32_t valid_keys) {
  std::vector<float> hidden_states(keys * hidden);
  std::vector<float> attention_norm_weight(hidden);
  std::vector<float> mlp_norm_weight(hidden);
  std::vector<float> attention_norm_states(keys * hidden, 0.0f);
  std::vector<float> q_weight(head_dim * hidden);
  std::vector<float> k_weight(head_dim * hidden);
  std::vector<float> v_weight(head_dim * hidden);
  std::vector<float> out_weight(hidden * head_dim);
  std::vector<float> gate_weight(intermediate * hidden);
  std::vector<float> up_weight(intermediate * hidden);
  std::vector<float> down_weight(hidden * intermediate);
  std::vector<float> cos(keys * (head_dim / 2));
  std::vector<float> sin(keys * (head_dim / 2));
  std::vector<float> out(hidden, 0.0f);
  const uint32_t half = head_dim / 2;
  const float scale = 1.0f / sqrtf((float)head_dim);

  for (uint32_t i = 0; i < hidden_states.size(); ++i) {
    hidden_states[i] =
        0.03125f * (float)((int32_t)((i * 5 + i / hidden) % 31) - 15);
  }
  for (uint32_t i = 0; i < hidden; ++i) {
    attention_norm_weight[i] = 1.0f + 0.001f * (float)(i % 29);
    mlp_norm_weight[i] = 1.0f - 0.001f * (float)(i % 17);
  }
  fill_projection_weight(q_weight, hidden, 1);
  fill_projection_weight(k_weight, hidden, 2);
  fill_projection_weight(v_weight, hidden, 3);
  fill_projection_weight(out_weight, head_dim, 4);
  fill_projection_weight(gate_weight, hidden, 5);
  fill_projection_weight(up_weight, hidden, 6);
  fill_projection_weight(down_weight, intermediate, 7);
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

  MiniDecoderLayerExecutor executor(device, keys, hidden, head_dim,
                                    intermediate);
  executor.copy_inputs(hidden_states, attention_norm_weight, mlp_norm_weight,
                       q_weight, k_weight, v_weight, out_weight, gate_weight,
                       up_weight, down_weight, cos, sin);
  executor.run(valid_keys);
  executor.synchronize();
  executor.copy_output(out);

  reference_rmsnorm(hidden_states, attention_norm_weight, keys, hidden,
                    attention_norm_states);

  const uint32_t query_position = valid_keys - 1;
  std::vector<float> q(head_dim);
  std::vector<float> k(keys * head_dim, 0.0f);
  std::vector<float> v(keys * head_dim, 0.0f);
  reference_projection(attention_norm_states, q_weight, hidden, query_position,
                       q);
  for (uint32_t row = 0; row < valid_keys; ++row) {
    std::vector<float> k_row(head_dim);
    std::vector<float> v_row(head_dim);
    reference_projection(attention_norm_states, k_weight, hidden, row, k_row);
    reference_projection(attention_norm_states, v_weight, hidden, row, v_row);
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

  std::vector<float> attention_out(head_dim, 0.0f);
  for (uint32_t col = 0; col < head_dim; ++col) {
    for (uint32_t row = 0; row < valid_keys; ++row) {
      attention_out[col] += reference_probs[row] * v[row * head_dim + col];
    }
  }

  std::vector<float> projected_attention(hidden, 0.0f);
  reference_vector_projection(attention_out, out_weight, head_dim,
                              projected_attention);

  std::vector<float> attention_residual(hidden, 0.0f);
  for (uint32_t col = 0; col < hidden; ++col) {
    attention_residual[col] =
        hidden_states[query_position * hidden + col] + projected_attention[col];
  }

  std::vector<float> mlp_norm_hidden(hidden, 0.0f);
  std::vector<float> gate(intermediate, 0.0f);
  std::vector<float> up(intermediate, 0.0f);
  std::vector<float> activated(intermediate, 0.0f);
  std::vector<float> projected_mlp(hidden, 0.0f);
  reference_rmsnorm_vector(attention_residual, mlp_norm_weight,
                           mlp_norm_hidden);
  reference_vector_projection(mlp_norm_hidden, gate_weight, hidden, gate);
  reference_vector_projection(mlp_norm_hidden, up_weight, hidden, up);
  for (uint32_t i = 0; i < intermediate; ++i) {
    activated[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
  }
  reference_vector_projection(activated, down_weight, intermediate,
                              projected_mlp);

  float max_diff = 0.0f;
  uint32_t max_index = 0;
  float max_expected = 0.0f;
  float max_actual = 0.0f;
  for (uint32_t col = 0; col < hidden; ++col) {
    float expected = attention_residual[col] + projected_mlp[col];
    float diff = fabsf(out[col] - expected);
    if (diff > max_diff) {
      max_diff = diff;
      max_index = col;
      max_expected = expected;
      max_actual = out[col];
    }
  }
  if (max_diff > 0.02f) {
    fprintf(stderr,
            "triton_mini_decoder_layer keys=%u hidden=%u head_dim=%u "
            "intermediate=%u valid_keys=%u mismatch at %u: actual=%f "
            "expected=%f diff=%f\n",
            keys, hidden, head_dim, intermediate, valid_keys, max_index,
            max_actual, max_expected, max_diff);
    throw std::runtime_error("triton_mini_decoder_layer result mismatch");
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
    run_case(device, 16, 768, 64, 2048, 7);
    run_case(device, 64, 1024, 128, 3072, 33);

    printf("triton_mini_decoder_layer: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
