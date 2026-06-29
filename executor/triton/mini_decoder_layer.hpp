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
      : DecoderLayer(device, keys, hidden, 1, head_dim, intermediate) {}

  DecoderLayer(lrrt::Device &device, uint32_t keys, uint32_t hidden,
               uint32_t heads, uint32_t head_dim, uint32_t intermediate)
      : DecoderLayer(device, keys, hidden, heads, heads, head_dim,
                     intermediate) {}

  DecoderLayer(lrrt::Device &device, uint32_t keys, uint32_t hidden,
               uint32_t heads, uint32_t kv_heads, uint32_t head_dim,
               uint32_t intermediate)
      : queue_(device), bundles_(device), buffers_(device), keys_(keys),
        hidden_(hidden), heads_(heads), kv_heads_(kv_heads),
        head_dim_(head_dim), q_dim_(heads * head_dim),
        kv_dim_(kv_heads * head_dim), intermediate_(intermediate),
        scale_(1.0f / sqrtf((float)head_dim)) {
    if (heads == 0 || kv_heads == 0 || kv_heads > heads ||
        heads % kv_heads != 0 || head_dim == 0 || head_dim % 2 != 0) {
      throw std::runtime_error("mini decoder layer invalid head shape");
    }
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
                 "matvec_fp32_" + std::to_string(select_block(q_dim_)));
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
    buffers_.allocate<float>("q_weight", q_dim_ * hidden);
    buffers_.allocate<float>("k_weight", kv_dim_ * hidden);
    buffers_.allocate<float>("v_weight", kv_dim_ * hidden);
    buffers_.allocate<float>("out_weight", hidden * q_dim_);
    buffers_.allocate<float>("gate_weight", intermediate * hidden);
    buffers_.allocate<float>("up_weight", intermediate * hidden);
    buffers_.allocate<float>("down_weight", hidden * intermediate);
    buffers_.allocate<float>("q", q_dim_);
    buffers_.allocate<float>("q_rope", q_dim_);
    buffers_.allocate<float>("source_k", keys * kv_dim_);
    buffers_.allocate<float>("source_v", keys * kv_dim_);
    buffers_.allocate<float>("k_rope", head_dim);
    buffers_.allocate<float>("k_cache", kv_heads * keys * head_dim);
    buffers_.allocate<float>("v_cache", kv_heads * keys * head_dim);
    buffers_.allocate<float>("cos", keys * (head_dim / 2));
    buffers_.allocate<float>("sin", keys * (head_dim / 2));
    buffers_.allocate<float>("scores", heads * keys);
    buffers_.allocate<float>("probs", heads * keys);
    buffers_.allocate<float>("attention_out", q_dim_);
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
        q_weight.size() != q_dim_ * hidden_ ||
        k_weight.size() != kv_dim_ * hidden_ ||
        v_weight.size() != kv_dim_ * hidden_ ||
        out_weight.size() != hidden_ * q_dim_ ||
        gate_weight.size() != intermediate_ * hidden_ ||
        up_weight.size() != intermediate_ * hidden_ ||
        down_weight.size() != hidden_ * intermediate_ ||
        cos.size() != keys_ * (head_dim_ / 2) ||
        sin.size() != keys_ * (head_dim_ / 2)) {
      throw std::runtime_error("mini decoder layer input shape mismatch");
    }

    std::vector<float> hidden_zero(hidden_, 0.0f);
    std::vector<float> hidden_cache_zero(keys_ * hidden_, 0.0f);
    std::vector<float> q_zero(q_dim_, 0.0f);
    std::vector<float> kv_zero(kv_dim_, 0.0f);
    std::vector<float> head_zero(head_dim_, 0.0f);
    std::vector<float> kv_cache_zero(keys_ * kv_dim_, 0.0f);
    std::vector<float> head_cache_zero(kv_heads_ * keys_ * head_dim_, 0.0f);
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
    buffers_.copy_to("q", q_zero);
    buffers_.copy_to("q_rope", q_zero);
    buffers_.copy_to("source_k", kv_cache_zero);
    buffers_.copy_to("source_v", kv_cache_zero);
    buffers_.copy_to("k_rope", head_zero);
    buffers_.copy_to("k_cache", head_cache_zero);
    buffers_.copy_to("v_cache", head_cache_zero);
    buffers_.copy_to("cos", cos);
    buffers_.copy_to("sin", sin);
    buffers_.copy_to("scores", scores_zero);
    buffers_.copy_to("probs", scores_zero);
    buffers_.copy_to("attention_out", q_zero);
    buffers_.copy_to("projected_attention", hidden_zero);
    buffers_.copy_to("attention_residual", hidden_zero);
    buffers_.copy_to("gate", intermediate_zero);
    buffers_.copy_to("up", intermediate_zero);
    buffers_.copy_to("activated", intermediate_zero);
    buffers_.copy_to("projected_mlp", hidden_zero);
    buffers_.copy_to("out", hidden_zero);
  }

  void copy_hidden_states(const std::vector<float> &hidden_states) {
    if (hidden_states.size() != keys_ * hidden_) {
      throw std::runtime_error(
          "mini decoder layer hidden state shape mismatch");
    }
    buffers_.copy_to("hidden_states", hidden_states);
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

    launch(queue_, bundles_.get("q_projection"), q_dim_,
           {
               arg("x", buffers_.ptr<float>("attention_norm_states",
                                            query_position * hidden_)),
               arg("weight", buffers_.ptr<float>("q_weight")),
               arg("out", buffers_.ptr<float>("q")),
               arg("outputs", (int32_t)q_dim_),
               arg("hidden", (int32_t)hidden_),
           });
    for (uint32_t position = 0; position < valid_keys; ++position) {
      launch(
          queue_, bundles_.get("k_projection"), kv_dim_,
          {
              arg("x", buffers_.ptr<float>("attention_norm_states",
                                           position * hidden_)),
              arg("weight", buffers_.ptr<float>("k_weight")),
              arg("out", buffers_.ptr<float>("source_k", position * kv_dim_)),
              arg("outputs", (int32_t)kv_dim_),
              arg("hidden", (int32_t)hidden_),
          });
      launch(
          queue_, bundles_.get("v_projection"), kv_dim_,
          {
              arg("x", buffers_.ptr<float>("attention_norm_states",
                                           position * hidden_)),
              arg("weight", buffers_.ptr<float>("v_weight")),
              arg("out", buffers_.ptr<float>("source_v", position * kv_dim_)),
              arg("outputs", (int32_t)kv_dim_),
              arg("hidden", (int32_t)hidden_),
          });
    }

    for (uint32_t kv_head = 0; kv_head < kv_heads_; ++kv_head) {
      const uint32_t kv_head_offset = kv_head * head_dim_;
      const uint32_t cache_offset = kv_head * keys_ * head_dim_;
      for (uint32_t position = 0; position < valid_keys; ++position) {
        launch(
            queue_, bundles_.get("rope"), 1,
            {
                arg("x", buffers_.ptr<float>("source_k", position * kv_dim_ +
                                                             kv_head_offset)),
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
                arg("v", buffers_.ptr<float>("source_v", position * kv_dim_ +
                                                             kv_head_offset)),
                arg("k_cache", buffers_.ptr<float>("k_cache", cache_offset)),
                arg("v_cache", buffers_.ptr<float>("v_cache", cache_offset)),
                arg("position", (int32_t)position),
                arg("max_tokens", (int32_t)keys_),
                arg("head_dim", (int32_t)head_dim_),
            });
      }
    }

    for (uint32_t head = 0; head < heads_; ++head) {
      const uint32_t head_offset = head * head_dim_;
      const uint32_t kv_head = head * kv_heads_ / heads_;
      const uint32_t cache_offset = kv_head * keys_ * head_dim_;
      const uint32_t scores_offset = head * keys_;

      launch(queue_, bundles_.get("rope"), 1,
             {
                 arg("x", buffers_.ptr<float>("q", head_offset)),
                 arg("cos", buffers_.ptr<float>("cos", query_position * half)),
                 arg("sin", buffers_.ptr<float>("sin", query_position * half)),
                 arg("out", buffers_.ptr<float>("q_rope", head_offset)),
                 arg("rows", (int32_t)1),
                 arg("heads", (int32_t)1),
                 arg("head_dim", (int32_t)head_dim_),
             });

      launch(queue_, bundles_.get("score"), keys_,
             {
                 arg("q", buffers_.ptr<float>("q_rope", head_offset)),
                 arg("k", buffers_.ptr<float>("k_cache", cache_offset)),
                 arg("out", buffers_.ptr<float>("scores", scores_offset)),
                 arg("keys", (int32_t)keys_),
                 arg("head_dim", (int32_t)head_dim_),
                 arg("scale", scale_),
             });
      launch(queue_, bundles_.get("softmax"), 1,
             {
                 arg("x", buffers_.ptr<float>("scores", scores_offset)),
                 arg("out", buffers_.ptr<float>("probs", scores_offset)),
                 arg("rows", (int32_t)1),
                 arg("hidden", (int32_t)keys_),
                 arg("query_start", (int32_t)(valid_keys - 1)),
             });
      launch(queue_, bundles_.get("aggregation"), head_dim_,
             {
                 arg("probs", buffers_.ptr<float>("probs", scores_offset)),
                 arg("v", buffers_.ptr<float>("v_cache", cache_offset)),
                 arg("out", buffers_.ptr<float>("attention_out", head_offset)),
                 arg("keys", (int32_t)keys_),
                 arg("head_dim", (int32_t)head_dim_),
             });
    }
    launch(queue_, bundles_.get("out_projection"), hidden_,
           {
               arg("x", buffers_.ptr<float>("attention_out")),
               arg("weight", buffers_.ptr<float>("out_weight")),
               arg("out", buffers_.ptr<float>("projected_attention")),
               arg("outputs", (int32_t)hidden_),
               arg("hidden", (int32_t)q_dim_),
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

  const lrrt::Queue &queue() const { return queue_; }

  void copy_output(std::vector<float> &out) const {
    if (out.size() != hidden_) {
      throw std::runtime_error("mini decoder layer output shape mismatch");
    }
    buffers_.copy_from(out, "out");
  }

  void copy_output_to_hidden_state(DecoderLayer &dst,
                                   uint32_t dst_position) const {
    if (hidden_ != dst.hidden_) {
      throw std::runtime_error(
          "mini decoder layer handoff hidden size mismatch");
    }
    if (dst_position >= dst.keys_) {
      throw std::runtime_error(
          "mini decoder layer handoff position is out of range");
    }
    lrrt::copy_device_to_device(
        dst.buffers_.get("hidden_states"),
        static_cast<size_t>(dst_position) * dst.hidden_ * sizeof(float),
        buffers_.get("out"), 0, static_cast<size_t>(hidden_) * sizeof(float));
  }

private:
  lrrt::Queue queue_;
  BundleSet bundles_;
  BufferSet buffers_;
  uint32_t keys_;
  uint32_t hidden_;
  uint32_t heads_;
  uint32_t kv_heads_;
  uint32_t head_dim_;
  uint32_t q_dim_;
  uint32_t kv_dim_;
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
