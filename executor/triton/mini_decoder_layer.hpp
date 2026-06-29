#pragma once

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

namespace lrrt::executor::triton::mini {

inline uint32_t select_head_block(uint32_t head_dim) {
  if (head_dim <= 64) {
    return 64;
  }
  if (head_dim <= 128) {
    return 128;
  }
  return 256;
}

inline uint32_t select_block(uint32_t size) {
  if (size <= 1024) {
    return 1024;
  }
  if (size <= 2048) {
    return 2048;
  }
  return 4096;
}

class DecoderLayer {
public:
  DecoderLayer(lrrt::Device &device, uint32_t keys, uint32_t hidden,
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
    launch(queue_, bundles_.get("attention_norm"), keys_,
           {
               arg("x", buffers_.ptr<float>("hidden_states")),
               arg("weight", buffers_.ptr<float>("attention_norm_weight")),
               arg("out", buffers_.ptr<float>("attention_norm_states")),
               arg("eps", 1.0e-5f),
               arg("rows", (int32_t)keys_),
               arg("hidden", (int32_t)hidden_),
           });

    launch(queue_, bundles_.get("q_projection"), head_dim_,
           {
               arg("x", buffers_.ptr<float>("attention_norm_states",
                                            query_position * hidden_)),
               arg("weight", buffers_.ptr<float>("q_weight")),
               arg("out", buffers_.ptr<float>("q")),
               arg("outputs", (int32_t)head_dim_),
               arg("hidden", (int32_t)hidden_),
           });
    for (uint32_t position = 0; position < valid_keys; ++position) {
      launch(
          queue_, bundles_.get("k_projection"), head_dim_,
          {
              arg("x", buffers_.ptr<float>("attention_norm_states",
                                           position * hidden_)),
              arg("weight", buffers_.ptr<float>("k_weight")),
              arg("out", buffers_.ptr<float>("source_k", position * head_dim_)),
              arg("outputs", (int32_t)head_dim_),
              arg("hidden", (int32_t)hidden_),
          });
      launch(
          queue_, bundles_.get("v_projection"), head_dim_,
          {
              arg("x", buffers_.ptr<float>("attention_norm_states",
                                           position * hidden_)),
              arg("weight", buffers_.ptr<float>("v_weight")),
              arg("out", buffers_.ptr<float>("source_v", position * head_dim_)),
              arg("outputs", (int32_t)head_dim_),
              arg("hidden", (int32_t)hidden_),
          });
    }

    launch(queue_, bundles_.get("rope"), 1,
           {
               arg("x", buffers_.ptr<float>("q")),
               arg("cos", buffers_.ptr<float>("cos", query_position * half)),
               arg("sin", buffers_.ptr<float>("sin", query_position * half)),
               arg("out", buffers_.ptr<float>("q_rope")),
               arg("rows", (int32_t)1),
               arg("heads", (int32_t)1),
               arg("head_dim", (int32_t)head_dim_),
           });

    for (uint32_t position = 0; position < valid_keys; ++position) {
      launch(
          queue_, bundles_.get("rope"), 1,
          {
              arg("x", buffers_.ptr<float>("source_k", position * head_dim_)),
              arg("cos", buffers_.ptr<float>("cos", position * half)),
              arg("sin", buffers_.ptr<float>("sin", position * half)),
              arg("out", buffers_.ptr<float>("k_rope")),
              arg("rows", (int32_t)1),
              arg("heads", (int32_t)1),
              arg("head_dim", (int32_t)head_dim_),
          });
      launch(
          queue_, bundles_.get("kv_update"), 1,
          {
              arg("k", buffers_.ptr<float>("k_rope")),
              arg("v", buffers_.ptr<float>("source_v", position * head_dim_)),
              arg("k_cache", buffers_.ptr<float>("k_cache")),
              arg("v_cache", buffers_.ptr<float>("v_cache")),
              arg("position", (int32_t)position),
              arg("max_tokens", (int32_t)keys_),
              arg("head_dim", (int32_t)head_dim_),
          });
    }

    launch(queue_, bundles_.get("score"), keys_,
           {
               arg("q", buffers_.ptr<float>("q_rope")),
               arg("k", buffers_.ptr<float>("k_cache")),
               arg("out", buffers_.ptr<float>("scores")),
               arg("keys", (int32_t)keys_),
               arg("head_dim", (int32_t)head_dim_),
               arg("scale", scale_),
           });
    launch(queue_, bundles_.get("softmax"), 1,
           {
               arg("x", buffers_.ptr<float>("scores")),
               arg("out", buffers_.ptr<float>("probs")),
               arg("rows", (int32_t)1),
               arg("hidden", (int32_t)keys_),
               arg("query_start", (int32_t)(valid_keys - 1)),
           });
    launch(queue_, bundles_.get("aggregation"), head_dim_,
           {
               arg("probs", buffers_.ptr<float>("probs")),
               arg("v", buffers_.ptr<float>("v_cache")),
               arg("out", buffers_.ptr<float>("attention_out")),
               arg("keys", (int32_t)keys_),
               arg("head_dim", (int32_t)head_dim_),
           });
    launch(queue_, bundles_.get("out_projection"), hidden_,
           {
               arg("x", buffers_.ptr<float>("attention_out")),
               arg("weight", buffers_.ptr<float>("out_weight")),
               arg("out", buffers_.ptr<float>("projected_attention")),
               arg("outputs", (int32_t)hidden_),
               arg("hidden", (int32_t)head_dim_),
           });
    launch(queue_, bundles_.get("residual_add"), hidden_,
           {
               arg("x", buffers_.ptr<float>("hidden_states",
                                            query_position * hidden_)),
               arg("y", buffers_.ptr<float>("projected_attention")),
               arg("out", buffers_.ptr<float>("attention_residual")),
               arg("n", (int32_t)hidden_),
           });

    launch(queue_, bundles_.get("mlp_norm"), 1,
           {
               arg("x", buffers_.ptr<float>("attention_residual")),
               arg("weight", buffers_.ptr<float>("mlp_norm_weight")),
               arg("out", buffers_.ptr<float>("mlp_norm_hidden")),
               arg("eps", 1.0e-5f),
               arg("rows", (int32_t)1),
               arg("hidden", (int32_t)hidden_),
           });
    launch(queue_, bundles_.get("gate_projection"), intermediate_,
           {
               arg("x", buffers_.ptr<float>("mlp_norm_hidden")),
               arg("weight", buffers_.ptr<float>("gate_weight")),
               arg("out", buffers_.ptr<float>("gate")),
               arg("outputs", (int32_t)intermediate_),
               arg("hidden", (int32_t)hidden_),
           });
    launch(queue_, bundles_.get("up_projection"), intermediate_,
           {
               arg("x", buffers_.ptr<float>("mlp_norm_hidden")),
               arg("weight", buffers_.ptr<float>("up_weight")),
               arg("out", buffers_.ptr<float>("up")),
               arg("outputs", (int32_t)intermediate_),
               arg("hidden", (int32_t)hidden_),
           });
    launch(queue_, bundles_.get("silu_mul"), intermediate_,
           {
               arg("gate", buffers_.ptr<float>("gate")),
               arg("up", buffers_.ptr<float>("up")),
               arg("out", buffers_.ptr<float>("activated")),
               arg("n", (int32_t)intermediate_),
           });
    launch(queue_, bundles_.get("down_projection"), hidden_,
           {
               arg("x", buffers_.ptr<float>("activated")),
               arg("weight", buffers_.ptr<float>("down_weight")),
               arg("out", buffers_.ptr<float>("projected_mlp")),
               arg("outputs", (int32_t)hidden_),
               arg("hidden", (int32_t)intermediate_),
           });
    launch(queue_, bundles_.get("residual_add"), hidden_,
           {
               arg("x", buffers_.ptr<float>("attention_residual")),
               arg("y", buffers_.ptr<float>("projected_mlp")),
               arg("out", buffers_.ptr<float>("out")),
               arg("n", (int32_t)hidden_),
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
  BundleSet bundles_;
  BufferSet buffers_;
  uint32_t keys_;
  uint32_t hidden_;
  uint32_t head_dim_;
  uint32_t intermediate_;
  float scale_;
};

inline void fill_projection_weight(std::vector<float> &weight, uint32_t input,
                                   uint32_t seed) {
  for (uint32_t row = 0; row < weight.size() / input; ++row) {
    for (uint32_t col = 0; col < input; ++col) {
      uint32_t index = row * input + col;
      int32_t lane = (int32_t)((index + row * 7 + seed * 11) % 23) - 11;
      weight[index] = 0.0078125f * (float)lane;
    }
  }
}

} // namespace lrrt::executor::triton::mini
