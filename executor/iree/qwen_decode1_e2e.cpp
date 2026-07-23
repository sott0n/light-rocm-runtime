#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "executor/iree/vmfb_runner.hpp"
#include "executor/qwen/weight_bundle.hpp"
#include "iree/base/api.h"
#include "iree/hal/api.h"

namespace {

using lrrt::executor::qwen::DecoderLayerWeights;
using lrrt::executor::qwen::load_decoder_layer_weights;
using lrrt::executor::qwen::load_model_tail_weights;
using lrrt::executor::qwen::ModelTailWeights;
using lrrt::iree_executor::BufferViewPtr;
using lrrt::iree_executor::VmfbRunner;

constexpr uint32_t kHidden = 896;
constexpr uint32_t kKvDim = 128;
constexpr uint32_t kIntermediate = 4864;
constexpr uint32_t kDefaultLayers = 24;

struct Args {
  const char *layer_vmfb = nullptr;
  const char *decode2_layer_vmfb = nullptr;
  const char *decode3_layer_vmfb = nullptr;
  const char *variable_layer_vmfb = nullptr;
  const char *tail_vmfb = nullptr;
  std::filesystem::path weights_dir;
  uint32_t layers = kDefaultLayers;
  uint32_t decode_steps = 1;
  uint32_t max_cache_tokens = 0;
};

struct LayerOutput {
  BufferViewPtr key_cache;
  BufferViewPtr value_cache;
  BufferViewPtr hidden;
};

struct LayerKvCache {
  BufferViewPtr key_cache;
  BufferViewPtr value_cache;
};

struct QwenKvCacheAbi {
  uint32_t visible_tokens;
  uint32_t kv_dim;
};

constexpr QwenKvCacheAbi kDecode1CacheAbi = {
    /*visible_tokens=*/1,
    /*kv_dim=*/kKvDim,
};

constexpr QwenKvCacheAbi kDecode2CacheAbi = {
    /*visible_tokens=*/2,
    /*kv_dim=*/kKvDim,
};

constexpr QwenKvCacheAbi kDecode3CacheAbi = {
    /*visible_tokens=*/3,
    /*kv_dim=*/kKvDim,
};

struct LayerWeightViews {
  BufferViewPtr attention_norm_weight;
  BufferViewPtr q_weight;
  BufferViewPtr k_weight;
  BufferViewPtr v_weight;
  BufferViewPtr out_weight;
  BufferViewPtr mlp_norm_weight;
  BufferViewPtr gate_weight;
  BufferViewPtr up_weight;
  BufferViewPtr down_weight;
};

std::vector<float> transpose(const std::vector<float> &source, uint32_t rows,
                             uint32_t cols, const char *label) {
  if (source.size() != static_cast<size_t>(rows) * cols) {
    throw std::runtime_error(std::string(label) + " shape mismatch");
  }
  std::vector<float> result(static_cast<size_t>(rows) * cols);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t col = 0; col < cols; ++col) {
      result[static_cast<size_t>(col) * rows + row] =
          source[static_cast<size_t>(row) * cols + col];
    }
  }
  return result;
}

iree_status_t make_view(VmfbRunner *runner, const std::vector<float> &data,
                        std::vector<iree_hal_dim_t> shape,
                        BufferViewPtr *out_view) {
  return runner->make_f32_buffer_view(data, shape, out_view);
}

iree_status_t make_i32_view(VmfbRunner *runner,
                            const std::vector<int32_t> &data,
                            std::vector<iree_hal_dim_t> shape,
                            BufferViewPtr *out_view) {
  return runner->make_i32_buffer_view(data, shape, out_view);
}

iree_status_t make_layer_weight_views(VmfbRunner *runner,
                                      const DecoderLayerWeights &weights,
                                      LayerWeightViews *views) {
  const std::vector<float> q_weight_t =
      transpose(weights.q_weight, kHidden, kHidden, "q_weight");
  const std::vector<float> k_weight_t =
      transpose(weights.k_weight, kKvDim, kHidden, "k_weight");
  const std::vector<float> v_weight_t =
      transpose(weights.v_weight, kKvDim, kHidden, "v_weight");
  const std::vector<float> out_weight_t =
      transpose(weights.out_weight, kHidden, kHidden, "out_weight");
  const std::vector<float> gate_weight_t =
      transpose(weights.gate_weight, kIntermediate, kHidden, "gate_weight");
  const std::vector<float> up_weight_t =
      transpose(weights.up_weight, kIntermediate, kHidden, "up_weight");
  const std::vector<float> down_weight_t =
      transpose(weights.down_weight, kHidden, kIntermediate, "down_weight");

  IREE_RETURN_IF_ERROR(make_view(runner, weights.attention_norm_weight,
                                 {kHidden}, &views->attention_norm_weight));
  IREE_RETURN_IF_ERROR(
      make_view(runner, q_weight_t, {kHidden, kHidden}, &views->q_weight));
  IREE_RETURN_IF_ERROR(
      make_view(runner, k_weight_t, {kHidden, kKvDim}, &views->k_weight));
  IREE_RETURN_IF_ERROR(
      make_view(runner, v_weight_t, {kHidden, kKvDim}, &views->v_weight));
  IREE_RETURN_IF_ERROR(
      make_view(runner, out_weight_t, {kHidden, kHidden}, &views->out_weight));
  IREE_RETURN_IF_ERROR(make_view(runner, weights.mlp_norm_weight, {kHidden},
                                 &views->mlp_norm_weight));
  IREE_RETURN_IF_ERROR(make_view(
      runner, gate_weight_t, {kHidden, kIntermediate}, &views->gate_weight));
  IREE_RETURN_IF_ERROR(make_view(runner, up_weight_t, {kHidden, kIntermediate},
                                 &views->up_weight));
  IREE_RETURN_IF_ERROR(make_view(
      runner, down_weight_t, {kIntermediate, kHidden}, &views->down_weight));
  return iree_ok_status();
}

iree_status_t run_layer(VmfbRunner *runner, const iree_vm_function_t &function,
                        iree_hal_buffer_view_t *hidden,
                        const LayerWeightViews &weight_views,
                        LayerOutput *out) {
  std::vector<iree_hal_buffer_view_t *> inputs = {
      hidden,
      weight_views.attention_norm_weight.get(),
      weight_views.q_weight.get(),
      weight_views.k_weight.get(),
      weight_views.v_weight.get(),
      weight_views.out_weight.get(),
      weight_views.mlp_norm_weight.get(),
      weight_views.gate_weight.get(),
      weight_views.up_weight.get(),
      weight_views.down_weight.get(),
  };

  std::vector<BufferViewPtr> outputs;
  IREE_RETURN_IF_ERROR(runner->invoke_views(function, inputs, 3, &outputs));
  out->key_cache.reset(outputs[0].release());
  out->value_cache.reset(outputs[1].release());
  out->hidden.reset(outputs[2].release());
  return iree_ok_status();
}

iree_status_t run_layer_with_cache(VmfbRunner *runner,
                                   const iree_vm_function_t &function,
                                   iree_hal_buffer_view_t *hidden,
                                   iree_hal_buffer_view_t *old_key_cache,
                                   iree_hal_buffer_view_t *old_value_cache,
                                   const LayerWeightViews &weight_views,
                                   LayerOutput *out) {
  std::vector<iree_hal_buffer_view_t *> inputs = {
      hidden,
      old_key_cache,
      old_value_cache,
      weight_views.attention_norm_weight.get(),
      weight_views.q_weight.get(),
      weight_views.k_weight.get(),
      weight_views.v_weight.get(),
      weight_views.out_weight.get(),
      weight_views.mlp_norm_weight.get(),
      weight_views.gate_weight.get(),
      weight_views.up_weight.get(),
      weight_views.down_weight.get(),
  };

  std::vector<BufferViewPtr> outputs;
  IREE_RETURN_IF_ERROR(runner->invoke_views(function, inputs, 3, &outputs));
  out->key_cache.reset(outputs[0].release());
  out->value_cache.reset(outputs[1].release());
  out->hidden.reset(outputs[2].release());
  return iree_ok_status();
}

iree_status_t run_layer_with_position_cache(
    VmfbRunner *runner, const iree_vm_function_t &function,
    iree_hal_buffer_view_t *hidden, iree_hal_buffer_view_t *old_key_cache,
    iree_hal_buffer_view_t *old_value_cache, iree_hal_buffer_view_t *position,
    const LayerWeightViews &weight_views, LayerOutput *out) {
  std::vector<iree_hal_buffer_view_t *> inputs = {
      hidden,
      old_key_cache,
      old_value_cache,
      position,
      weight_views.attention_norm_weight.get(),
      weight_views.q_weight.get(),
      weight_views.k_weight.get(),
      weight_views.v_weight.get(),
      weight_views.out_weight.get(),
      weight_views.mlp_norm_weight.get(),
      weight_views.gate_weight.get(),
      weight_views.up_weight.get(),
      weight_views.down_weight.get(),
  };

  std::vector<BufferViewPtr> outputs;
  IREE_RETURN_IF_ERROR(runner->invoke_views(function, inputs, 3, &outputs));
  out->key_cache.reset(outputs[0].release());
  out->value_cache.reset(outputs[1].release());
  out->hidden.reset(outputs[2].release());
  return iree_ok_status();
}

bool has_layer_cache(const LayerKvCache &cache) {
  return cache.key_cache.get() != nullptr && cache.value_cache.get() != nullptr;
}

iree_status_t validate_layer_cache(iree_hal_buffer_view_t *key_cache,
                                   iree_hal_buffer_view_t *value_cache,
                                   const QwenKvCacheAbi &abi,
                                   const char *label) {
  if (!key_cache || !value_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s requires both key and value cache views",
                            label);
  }
  if (iree_hal_buffer_view_shape_rank(key_cache) != 2 ||
      iree_hal_buffer_view_shape_dim(key_cache, 0) != abi.visible_tokens ||
      iree_hal_buffer_view_shape_dim(key_cache, 1) != abi.kv_dim) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s key cache must be %ux%uxf32", label,
                            abi.visible_tokens, abi.kv_dim);
  }
  if (iree_hal_buffer_view_shape_rank(value_cache) != 2 ||
      iree_hal_buffer_view_shape_dim(value_cache, 0) != abi.visible_tokens ||
      iree_hal_buffer_view_shape_dim(value_cache, 1) != abi.kv_dim) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s value cache must be %ux%uxf32", label,
                            abi.visible_tokens, abi.kv_dim);
  }
  return iree_ok_status();
}

iree_status_t run_decode_token_step(
    VmfbRunner *runner, const iree_vm_function_t &decode1_function,
    const iree_vm_function_t &decode2_function,
    const iree_vm_function_t &decode3_function, uint32_t step_index,
    iree_hal_buffer_view_t *input_hidden,
    const std::vector<LayerWeightViews> &weights,
    std::vector<LayerKvCache> *layer_caches, BufferViewPtr *out_hidden) {
  if (step_index > 2) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Qwen IREE E2E currently supports decode steps 1 through 3 only; "
        "step %u needs the variable-length KV cache ABI VMFB",
        step_index + 1);
  }

  BufferViewPtr hidden;
  iree_hal_buffer_view_retain(input_hidden);
  hidden.reset(input_hidden);

  for (uint32_t layer = 0; layer < weights.size(); ++layer) {
    LayerOutput output;
    if (step_index == 0) {
      IREE_RETURN_IF_ERROR(run_layer(runner, decode1_function, hidden.get(),
                                     weights[layer], &output));
      IREE_RETURN_IF_ERROR(
          validate_layer_cache(output.key_cache.get(), output.value_cache.get(),
                               kDecode1CacheAbi, "decode step 1"));
    } else if (step_index == 1) {
      if (!has_layer_cache((*layer_caches)[layer])) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "decode step %u requires existing layer_%u KV cache",
            step_index + 1, layer);
      }
      IREE_RETURN_IF_ERROR(run_layer_with_cache(
          runner, decode2_function, hidden.get(),
          (*layer_caches)[layer].key_cache.get(),
          (*layer_caches)[layer].value_cache.get(), weights[layer], &output));
      IREE_RETURN_IF_ERROR(
          validate_layer_cache(output.key_cache.get(), output.value_cache.get(),
                               kDecode2CacheAbi, "decode step 2"));
    } else {
      if (!has_layer_cache((*layer_caches)[layer])) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "decode step %u requires existing layer_%u KV cache",
            step_index + 1, layer);
      }
      IREE_RETURN_IF_ERROR(run_layer_with_cache(
          runner, decode3_function, hidden.get(),
          (*layer_caches)[layer].key_cache.get(),
          (*layer_caches)[layer].value_cache.get(), weights[layer], &output));
      IREE_RETURN_IF_ERROR(
          validate_layer_cache(output.key_cache.get(), output.value_cache.get(),
                               kDecode3CacheAbi, "decode step 3"));
    }

    hidden = std::move(output.hidden);
    (*layer_caches)[layer].key_cache = std::move(output.key_cache);
    (*layer_caches)[layer].value_cache = std::move(output.value_cache);
    std::printf("step %u layer %u/%zu complete\n", step_index + 1, layer + 1,
                weights.size());
    std::fflush(stdout);
  }

  *out_hidden = std::move(hidden);
  return iree_ok_status();
}

iree_status_t
initialize_variable_layer_caches(VmfbRunner *runner, uint32_t layers,
                                 uint32_t max_cache_tokens,
                                 std::vector<LayerKvCache> *layer_caches) {
  const std::vector<float> zero_cache(static_cast<size_t>(max_cache_tokens) *
                                      kKvDim);
  for (uint32_t layer = 0; layer < layers; ++layer) {
    IREE_RETURN_IF_ERROR(make_view(runner, zero_cache,
                                   {max_cache_tokens, kKvDim},
                                   &(*layer_caches)[layer].key_cache));
    IREE_RETURN_IF_ERROR(make_view(runner, zero_cache,
                                   {max_cache_tokens, kKvDim},
                                   &(*layer_caches)[layer].value_cache));
  }
  return iree_ok_status();
}

iree_status_t run_decode_token_step_with_variable_cache(
    VmfbRunner *runner, const iree_vm_function_t &function, uint32_t step_index,
    uint32_t max_cache_tokens, iree_hal_buffer_view_t *input_hidden,
    const std::vector<LayerWeightViews> &weights,
    std::vector<LayerKvCache> *layer_caches, BufferViewPtr *out_hidden) {
  if (step_index >= max_cache_tokens) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "decode step %u exceeds KV cache capacity %u",
                            step_index + 1, max_cache_tokens);
  }

  BufferViewPtr position;
  IREE_RETURN_IF_ERROR(make_i32_view(runner, {static_cast<int32_t>(step_index)},
                                     {1}, &position));

  BufferViewPtr hidden;
  iree_hal_buffer_view_retain(input_hidden);
  hidden.reset(input_hidden);

  for (uint32_t layer = 0; layer < weights.size(); ++layer) {
    if (!has_layer_cache((*layer_caches)[layer])) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "decode step %u requires initialized layer_%u KV cache",
          step_index + 1, layer);
    }

    LayerOutput output;
    IREE_RETURN_IF_ERROR(run_layer_with_position_cache(
        runner, function, hidden.get(), (*layer_caches)[layer].key_cache.get(),
        (*layer_caches)[layer].value_cache.get(), position.get(),
        weights[layer], &output));
    IREE_RETURN_IF_ERROR(validate_layer_cache(
        output.key_cache.get(), output.value_cache.get(),
        QwenKvCacheAbi{/*visible_tokens=*/max_cache_tokens, /*kv_dim=*/kKvDim},
        "variable KV cache decode step"));

    hidden = std::move(output.hidden);
    (*layer_caches)[layer].key_cache = std::move(output.key_cache);
    (*layer_caches)[layer].value_cache = std::move(output.value_cache);
    std::printf("step %u layer %u/%zu complete\n", step_index + 1, layer + 1,
                weights.size());
    std::fflush(stdout);
  }

  *out_hidden = std::move(hidden);
  return iree_ok_status();
}

iree_status_t run_tail(VmfbRunner *runner, const iree_vm_function_t &function,
                       iree_hal_buffer_view_t *hidden,
                       const ModelTailWeights &weights,
                       BufferViewPtr *out_logits) {
  BufferViewPtr final_norm_weight;
  BufferViewPtr lm_head_weight;
  const std::vector<float> lm_head_weight_t = transpose(
      weights.lm_head_weight, weights.vocab, weights.hidden, "lm_head_weight");

  IREE_RETURN_IF_ERROR(make_view(runner, weights.final_norm_weight,
                                 {weights.hidden}, &final_norm_weight));
  IREE_RETURN_IF_ERROR(make_view(runner, lm_head_weight_t,
                                 {weights.hidden, weights.vocab},
                                 &lm_head_weight));

  std::vector<iree_hal_buffer_view_t *> inputs = {
      hidden,
      final_norm_weight.get(),
      lm_head_weight.get(),
  };

  std::vector<BufferViewPtr> outputs;
  IREE_RETURN_IF_ERROR(runner->invoke_views(function, inputs, 1, &outputs));
  out_logits->reset(outputs[0].release());
  return iree_ok_status();
}

iree_status_t inspect_logits(iree_hal_buffer_view_t *view, uint32_t vocab,
                             uint32_t step, uint32_t *out_top_token) {
  if (iree_hal_buffer_view_shape_rank(view) != 2 ||
      iree_hal_buffer_view_shape_dim(view, 0) != 1 ||
      iree_hal_buffer_view_shape_dim(view, 1) != vocab) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected logits shape");
  }

  std::vector<float> logits(vocab);
  iree_hal_buffer_t *buffer = iree_hal_buffer_view_buffer(view);
  iree_status_t status = iree_hal_buffer_map_read(
      buffer, 0, logits.data(), logits.size() * sizeof(float));
  IREE_RETURN_IF_ERROR(status);

  uint32_t top_index = 0;
  float top_value = -std::numeric_limits<float>::infinity();
  for (uint32_t i = 0; i < vocab; ++i) {
    const float value = logits[i];
    if (!std::isfinite(value)) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "logits[%u] is non-finite: %f", i, value);
    }
    if (value > top_value) {
      top_value = value;
      top_index = i;
    }
  }
  std::printf("step %u top_token=%u logit=%g vocab=%u\n", step, top_index,
              top_value, vocab);
  *out_top_token = top_index;
  return iree_ok_status();
}

iree_status_t make_token_hidden(VmfbRunner *runner,
                                const ModelTailWeights &tail_weights,
                                uint32_t token_id, BufferViewPtr *out_hidden) {
  const auto iter = std::find(tail_weights.token_ids.begin(),
                              tail_weights.token_ids.end(), token_id);
  if (iter == tail_weights.token_ids.end()) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "token embedding for logits-selected token %u is not available in the "
        "model tail bundle",
        token_id);
  }
  const size_t token_index =
      static_cast<size_t>(std::distance(tail_weights.token_ids.begin(), iter));
  const auto begin = tail_weights.token_embeddings.begin() +
                     static_cast<ptrdiff_t>(token_index * kHidden);
  std::vector<float> initial_hidden(begin, begin + kHidden);
  return make_view(runner, initial_hidden, {1, kHidden}, out_hidden);
}

Args parse_args(int argc, char **argv) {
  Args args;
  if (argc >= 2 && std::string(argv[1]) == "--steps") {
    if (argc < 3) {
      throw std::runtime_error(
          "usage: lrrt_iree_qwen_decode1_e2e --steps <N> "
          "<decode1-layer.vmfb> "
          "[decode2-layer.vmfb] [decode3-layer.vmfb] <tail.vmfb> "
          "<weights-dir> [layers]\n"
          "   or: lrrt_iree_qwen_decode1_e2e --steps <N> --max-cache-tokens 8 "
          "<kv-cache-layer.vmfb> <tail.vmfb> <weights-dir> [layers]");
    }
    const int parsed_steps = std::stoi(argv[2]);
    if (parsed_steps <= 0) {
      throw std::runtime_error("--steps must be positive");
    }
    args.decode_steps = static_cast<uint32_t>(parsed_steps);
    if (argc >= 5 && std::string(argv[3]) == "--max-cache-tokens") {
      const int parsed_max_cache_tokens = std::stoi(argv[4]);
      if (parsed_max_cache_tokens <= 0) {
        throw std::runtime_error("--max-cache-tokens must be positive");
      }
      args.max_cache_tokens = static_cast<uint32_t>(parsed_max_cache_tokens);
      if (args.decode_steps > args.max_cache_tokens) {
        throw std::runtime_error("--steps must not exceed --max-cache-tokens");
      }
      if (args.max_cache_tokens != 8) {
        throw std::runtime_error(
            "only --max-cache-tokens 8 is currently built");
      }
      if (argc != 8 && argc != 9) {
        throw std::runtime_error(
            "usage: lrrt_iree_qwen_decode1_e2e --steps <N> "
            "--max-cache-tokens 8 <kv-cache-layer.vmfb> <tail.vmfb> "
            "<weights-dir> [layers]");
      }
      args.variable_layer_vmfb = argv[5];
      args.tail_vmfb = argv[6];
      args.weights_dir = argv[7];
      if (argc == 9) {
        const int parsed = std::stoi(argv[8]);
        if (parsed <= 0) {
          throw std::runtime_error("layers must be positive");
        }
        args.layers = static_cast<uint32_t>(parsed);
      }
      return args;
    }
    if (args.decode_steps == 1) {
      if (argc != 6 && argc != 7) {
        throw std::runtime_error(
            "usage: lrrt_iree_qwen_decode1_e2e --steps 1 <decode1-layer.vmfb> "
            "<tail.vmfb> <weights-dir> [layers]");
      }
      args.layer_vmfb = argv[3];
      args.tail_vmfb = argv[4];
      args.weights_dir = argv[5];
      if (argc == 7) {
        const int parsed = std::stoi(argv[6]);
        if (parsed <= 0) {
          throw std::runtime_error("layers must be positive");
        }
        args.layers = static_cast<uint32_t>(parsed);
      }
      return args;
    }
    if (args.decode_steps == 2) {
      if (argc != 7 && argc != 8) {
        throw std::runtime_error(
            "usage: lrrt_iree_qwen_decode1_e2e --steps 2 <decode1-layer.vmfb> "
            "<decode2-layer.vmfb> <tail.vmfb> <weights-dir> [layers]");
      }
      args.layer_vmfb = argv[3];
      args.decode2_layer_vmfb = argv[4];
      args.tail_vmfb = argv[5];
      args.weights_dir = argv[6];
      if (argc == 8) {
        const int parsed = std::stoi(argv[7]);
        if (parsed <= 0) {
          throw std::runtime_error("layers must be positive");
        }
        args.layers = static_cast<uint32_t>(parsed);
      }
      return args;
    }
    if (args.decode_steps == 3) {
      if (argc != 8 && argc != 9) {
        throw std::runtime_error(
            "usage: lrrt_iree_qwen_decode1_e2e --steps 3 <decode1-layer.vmfb> "
            "<decode2-layer.vmfb> <decode3-layer.vmfb> <tail.vmfb> "
            "<weights-dir> [layers]");
      }
      args.layer_vmfb = argv[3];
      args.decode2_layer_vmfb = argv[4];
      args.decode3_layer_vmfb = argv[5];
      args.tail_vmfb = argv[6];
      args.weights_dir = argv[7];
      if (argc == 9) {
        const int parsed = std::stoi(argv[8]);
        if (parsed <= 0) {
          throw std::runtime_error("layers must be positive");
        }
        args.layers = static_cast<uint32_t>(parsed);
      }
      return args;
    }
    throw std::runtime_error(
        "--steps greater than 3 requires --max-cache-tokens 8 "
        "<kv-cache-layer.vmfb>");
  }

  if (argc != 4 && argc != 5) {
    throw std::runtime_error(
        "usage: lrrt_iree_qwen_decode1_e2e <layer.vmfb> <tail.vmfb> "
        "<weights-dir> [layers]");
  }
  args.layer_vmfb = argv[1];
  args.tail_vmfb = argv[2];
  args.weights_dir = argv[3];
  if (argc == 5) {
    const int parsed = std::stoi(argv[4]);
    if (parsed <= 0) {
      throw std::runtime_error("layers must be positive");
    }
    args.layers = static_cast<uint32_t>(parsed);
  }
  return args;
}

iree_status_t run(const Args &args) {
  VmfbRunner runner;
  std::vector<const char *> modules = {args.layer_vmfb, args.tail_vmfb};
  if (args.max_cache_tokens != 0) {
    modules = {args.variable_layer_vmfb, args.tail_vmfb};
  } else if (args.decode_steps == 2) {
    modules = {args.layer_vmfb, args.decode2_layer_vmfb, args.tail_vmfb};
  } else if (args.decode_steps == 3) {
    modules = {args.layer_vmfb, args.decode2_layer_vmfb,
               args.decode3_layer_vmfb, args.tail_vmfb};
  }
  IREE_RETURN_IF_ERROR(runner.initialize(modules));

  iree_vm_function_t layer_function = {};
  iree_vm_function_t decode2_layer_function = {};
  iree_vm_function_t decode3_layer_function = {};
  iree_vm_function_t variable_layer_function = {};
  iree_vm_function_t tail_function = {};
  if (args.max_cache_tokens != 0) {
    IREE_RETURN_IF_ERROR(runner.lookup_function(
        "qwen_decode_layer_kv_cache_max8", &variable_layer_function));
  } else {
    IREE_RETURN_IF_ERROR(
        runner.lookup_function("qwen_decode1_layer", &layer_function));
  }
  if (args.max_cache_tokens == 0 && args.decode_steps > 1) {
    IREE_RETURN_IF_ERROR(runner.lookup_function("qwen_decode2_layer_kv_cache",
                                                &decode2_layer_function));
  }
  if (args.max_cache_tokens == 0 && args.decode_steps > 2) {
    IREE_RETURN_IF_ERROR(runner.lookup_function("qwen_decode3_layer_kv_cache",
                                                &decode3_layer_function));
  }
  IREE_RETURN_IF_ERROR(
      runner.lookup_function("qwen_decode1_tail", &tail_function));

  const std::filesystem::path tail_manifest =
      args.weights_dir / "model_tail" / "weights.json";
  const ModelTailWeights tail_weights =
      load_model_tail_weights(tail_manifest.c_str());
  if (tail_weights.hidden != kHidden || tail_weights.token_embeddings.empty()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model tail bundle has unexpected Qwen shape");
  }

  if (tail_weights.token_embeddings.size() < kHidden) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model tail bundle does not contain enough token "
                            "embedding rows");
  }

  std::vector<DecoderLayerWeights> layer_weights;
  layer_weights.reserve(args.layers);
  std::vector<LayerWeightViews> layer_weight_views(args.layers);
  std::vector<LayerKvCache> layer_caches(args.layers);

  for (uint32_t layer = 0; layer < args.layers; ++layer) {
    const std::filesystem::path layer_manifest =
        args.weights_dir / ("layer_" + std::to_string(layer)) / "weights.json";
    layer_weights.push_back(load_decoder_layer_weights(layer_manifest.c_str()));
    if (layer_weights.back().shape.hidden != kHidden ||
        layer_weights.back().shape.kv_dim() != kKvDim ||
        layer_weights.back().shape.intermediate != kIntermediate) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "layer_%u bundle has unexpected Qwen shape",
                              layer);
    }
    IREE_RETURN_IF_ERROR(make_layer_weight_views(&runner, layer_weights.back(),
                                                 &layer_weight_views[layer]));
  }
  if (args.max_cache_tokens != 0) {
    IREE_RETURN_IF_ERROR(initialize_variable_layer_caches(
        &runner, args.layers, args.max_cache_tokens, &layer_caches));
  }

  BufferViewPtr hidden;
  uint32_t current_token = tail_weights.token_ids.front();
  for (uint32_t step = 0; step < args.decode_steps; ++step) {
    BufferViewPtr token_hidden;
    IREE_RETURN_IF_ERROR(
        make_token_hidden(&runner, tail_weights, current_token, &token_hidden));
    if (args.max_cache_tokens != 0) {
      IREE_RETURN_IF_ERROR(run_decode_token_step_with_variable_cache(
          &runner, variable_layer_function, step, args.max_cache_tokens,
          token_hidden.get(), layer_weight_views, &layer_caches, &hidden));
    } else {
      IREE_RETURN_IF_ERROR(run_decode_token_step(
          &runner, layer_function, decode2_layer_function,
          decode3_layer_function, step, token_hidden.get(), layer_weight_views,
          &layer_caches, &hidden));
    }
    BufferViewPtr step_logits;
    IREE_RETURN_IF_ERROR(run_tail(&runner, tail_function, hidden.get(),
                                  tail_weights, &step_logits));
    IREE_RETURN_IF_ERROR(inspect_logits(step_logits.get(), tail_weights.vocab,
                                        step + 1, &current_token));
  }

  std::printf("iree_qwen_decode%u_e2e: ok layers=%u\n", args.decode_steps,
              args.layers);
  return iree_ok_status();
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Args args = parse_args(argc, argv);
    iree_status_t status = run(args);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      return 1;
    }
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "lrrt_iree_qwen_decode1_e2e: %s\n", error.what());
    return 2;
  }
}
