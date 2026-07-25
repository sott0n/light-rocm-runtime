#include <algorithm>
#include <chrono>
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

#include "executor/iree/qwen_decode_bundle.hpp"
#include "executor/iree/vmfb_runner.hpp"
#include "executor/qwen/weight_bundle.hpp"
#include "iree/base/api.h"
#include "iree/hal/api.h"

namespace {

using lrrt::executor::iree::load_qwen_decode_bundle_manifest;
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
constexpr uint32_t kSupportedMaxCacheTokens[] = {8, 16, 32, 64};
using SteadyClock = std::chrono::steady_clock;

double elapsed_ms(SteadyClock::time_point start) {
  return std::chrono::duration<double, std::milli>(SteadyClock::now() - start)
      .count();
}

struct Args {
  std::string layer_vmfb;
  std::string decode2_layer_vmfb;
  std::string decode3_layer_vmfb;
  std::string variable_layer_vmfb;
  std::string tail_vmfb;
  std::string layer_export = "qwen_decode1_layer";
  std::string decode2_layer_export = "qwen_decode2_layer_kv_cache";
  std::string decode3_layer_export = "qwen_decode3_layer_kv_cache";
  std::string variable_layer_export = "qwen_decode_layer_kv_cache_max8";
  std::string tail_export = "qwen_decode1_tail";
  std::filesystem::path weights_dir;
  uint32_t layers = kDefaultLayers;
  uint32_t decode_steps = 1;
  uint32_t max_seq_len = 1;
  uint32_t max_cache_tokens = 0;
  std::vector<uint32_t> prompt_token_ids;
  bool eos_token_id_set = false;
  uint32_t eos_token_id = 0;
  bool verbose_layers = false;
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

bool is_supported_max_cache_tokens(uint32_t max_cache_tokens) {
  for (const uint32_t supported : kSupportedMaxCacheTokens) {
    if (max_cache_tokens == supported) {
      return true;
    }
  }
  return false;
}

std::vector<uint32_t> parse_token_ids(const std::string &text,
                                      const char *flag) {
  if (text.empty()) {
    throw std::runtime_error(std::string(flag) + " must not be empty");
  }
  std::vector<uint32_t> values;
  size_t begin = 0;
  while (begin <= text.size()) {
    const size_t end = text.find(',', begin);
    const std::string item = text.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin);
    if (item.empty()) {
      throw std::runtime_error(std::string(flag) + " contains an empty entry");
    }
    size_t parsed = 0;
    const int value = std::stoi(item, &parsed);
    if (parsed != item.size()) {
      throw std::runtime_error(std::string(flag) +
                               " contains a non-integer entry");
    }
    if (value < 0) {
      throw std::runtime_error(std::string(flag) +
                               " must contain non-negative integers");
    }
    values.push_back(static_cast<uint32_t>(value));
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return values;
}

void validate_generation_window(const Args &args) {
  const uint32_t prompt_len =
      args.prompt_token_ids.empty()
          ? 1
          : static_cast<uint32_t>(args.prompt_token_ids.size());
  if (prompt_len + args.decode_steps > args.max_seq_len) {
    throw std::runtime_error(
        "--prompt-token-ids length plus --max-new-tokens must not exceed "
        "--max-seq-len");
  }
}

struct LayerWeightViews {
  BufferViewPtr attention_norm_weight;
  BufferViewPtr q_weight;
  BufferViewPtr k_weight;
  BufferViewPtr v_weight;
  BufferViewPtr q_bias;
  BufferViewPtr k_bias;
  BufferViewPtr v_bias;
  BufferViewPtr rope_theta;
  BufferViewPtr out_weight;
  BufferViewPtr mlp_norm_weight;
  BufferViewPtr gate_weight;
  BufferViewPtr up_weight;
  BufferViewPtr down_weight;
};

struct ModuleFunctions {
  iree_vm_function_t layer = {};
  iree_vm_function_t decode2_layer = {};
  iree_vm_function_t decode3_layer = {};
  iree_vm_function_t variable_layer = {};
  iree_vm_function_t tail = {};
};

struct LoadedQwenWeights {
  ModelTailWeights tail;
  std::vector<DecoderLayerWeights> layers;
  std::vector<LayerWeightViews> layer_views;
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
      make_view(runner, weights.q_bias, {kHidden}, &views->q_bias));
  IREE_RETURN_IF_ERROR(
      make_view(runner, weights.k_bias, {kKvDim}, &views->k_bias));
  IREE_RETURN_IF_ERROR(
      make_view(runner, weights.v_bias, {kKvDim}, &views->v_bias));
  IREE_RETURN_IF_ERROR(
      make_view(runner, {weights.rope_theta}, {1}, &views->rope_theta));
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
      weight_views.q_bias.get(),
      weight_views.k_bias.get(),
      weight_views.v_bias.get(),
      weight_views.rope_theta.get(),
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

iree_status_t
run_decode_token_step(VmfbRunner *runner,
                      const iree_vm_function_t &decode1_function,
                      const iree_vm_function_t &decode2_function,
                      const iree_vm_function_t &decode3_function,
                      uint32_t step_index, iree_hal_buffer_view_t *input_hidden,
                      const std::vector<LayerWeightViews> &weights,
                      std::vector<LayerKvCache> *layer_caches,
                      bool verbose_layers, BufferViewPtr *out_hidden) {
  if (step_index > 2) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Qwen IREE E2E currently supports decode tokens 1 through 3 only; "
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
    if (verbose_layers) {
      std::printf("step %u layer %u/%zu complete\n", step_index + 1, layer + 1,
                  weights.size());
      std::fflush(stdout);
    }
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
    std::vector<LayerKvCache> *layer_caches, bool verbose_layers,
    BufferViewPtr *out_hidden) {
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
    if (verbose_layers) {
      std::printf("step %u layer %u/%zu complete\n", step_index + 1, layer + 1,
                  weights.size());
      std::fflush(stdout);
    }
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
  if (argc >= 2 && std::string(argv[1]) == "--max-new-tokens") {
    if (argc < 3) {
      throw std::runtime_error(
          "usage: lrrt_iree_qwen_decode1_e2e --max-new-tokens <N> "
          "[--max-seq-len <N>] [--prompt-token-ids <ids>] "
          "[--eos-token-id <id>] [--verbose-layers] "
          "<decode1-layer.vmfb> "
          "[decode2-layer.vmfb] [decode3-layer.vmfb] <tail.vmfb> "
          "<weights-dir> [layers]\n"
          "   or: lrrt_iree_qwen_decode1_e2e --max-new-tokens <N> --bundle "
          "<bundle-dir> <weights-dir> [layers]\n"
          "   or: lrrt_iree_qwen_decode1_e2e --max-new-tokens <N> "
          "--max-cache-tokens <8|16|32|64> "
          "<kv-cache-layer.vmfb> <tail.vmfb> <weights-dir> [layers]");
    }
    const int parsed_max_new_tokens = std::stoi(argv[2]);
    if (parsed_max_new_tokens <= 0) {
      throw std::runtime_error("--max-new-tokens must be positive");
    }
    args.decode_steps = static_cast<uint32_t>(parsed_max_new_tokens);
    args.max_seq_len = args.decode_steps;

    int option_index = 3;
    bool max_seq_len_explicit = false;
    if (argc >= 5 && std::string(argv[option_index]) == "--max-seq-len") {
      const int parsed_max_seq_len = std::stoi(argv[option_index + 1]);
      if (parsed_max_seq_len <= 0) {
        throw std::runtime_error("--max-seq-len must be positive");
      }
      args.max_seq_len = static_cast<uint32_t>(parsed_max_seq_len);
      max_seq_len_explicit = true;
      if (args.decode_steps > args.max_seq_len) {
        throw std::runtime_error(
            "--max-new-tokens must not exceed --max-seq-len");
      }
      option_index += 2;
    }
    if (argc >= option_index + 2 &&
        std::string(argv[option_index]) == "--prompt-token-ids") {
      args.prompt_token_ids =
          parse_token_ids(argv[option_index + 1], "--prompt-token-ids");
      option_index += 2;
    }
    if (argc >= option_index + 2 &&
        std::string(argv[option_index]) == "--eos-token-id") {
      const int parsed_eos_token_id = std::stoi(argv[option_index + 1]);
      if (parsed_eos_token_id < 0) {
        throw std::runtime_error("--eos-token-id must be non-negative");
      }
      args.eos_token_id_set = true;
      args.eos_token_id = static_cast<uint32_t>(parsed_eos_token_id);
      option_index += 2;
    }
    if (argc > option_index &&
        std::string(argv[option_index]) == "--verbose-layers") {
      args.verbose_layers = true;
      option_index += 1;
    }
    if (!max_seq_len_explicit) {
      const uint32_t prompt_len =
          args.prompt_token_ids.empty()
              ? 1
              : static_cast<uint32_t>(args.prompt_token_ids.size());
      args.max_seq_len = prompt_len + args.decode_steps;
    }
    validate_generation_window(args);

    if (argc > option_index && std::string(argv[option_index]) == "--bundle") {
      if (argc != option_index + 3 && argc != option_index + 4) {
        throw std::runtime_error(
            "usage: lrrt_iree_qwen_decode1_e2e --max-new-tokens <N> "
            "[--max-seq-len <N>] [--prompt-token-ids <ids>] "
            "[--eos-token-id <id>] [--verbose-layers] --bundle "
            "<bundle-dir> <weights-dir> [layers]");
      }
      const std::filesystem::path bundle_dir = argv[option_index + 1];
      const auto manifest = load_qwen_decode_bundle_manifest(bundle_dir);
      args.max_cache_tokens = manifest.max_cache_tokens;
      if (args.max_seq_len > manifest.sequence_capacity) {
        throw std::runtime_error(
            "--max-seq-len must not exceed bundle sequence_capacity");
      }
      if (args.decode_steps > args.max_seq_len) {
        throw std::runtime_error(
            "--max-new-tokens must not exceed --max-seq-len");
      }
      validate_generation_window(args);
      if (manifest.kv_cache_dim != kKvDim) {
        throw std::runtime_error(
            "bundle kv_cache_shape[1] does not match Qwen kv dim");
      }
      args.variable_layer_vmfb = manifest.layer_vmfb.string();
      args.tail_vmfb = manifest.tail_vmfb.string();
      args.variable_layer_export = manifest.layer_export;
      args.tail_export = manifest.tail_export;
      args.weights_dir = argv[option_index + 2];
      if (argc == option_index + 4) {
        const int parsed = std::stoi(argv[option_index + 3]);
        if (parsed <= 0) {
          throw std::runtime_error("layers must be positive");
        }
        args.layers = static_cast<uint32_t>(parsed);
      }
      return args;
    }
    if (argc > option_index &&
        std::string(argv[option_index]) == "--max-cache-tokens") {
      const int parsed_max_cache_tokens = std::stoi(argv[option_index + 1]);
      if (parsed_max_cache_tokens <= 0) {
        throw std::runtime_error("--max-cache-tokens must be positive");
      }
      args.max_cache_tokens = static_cast<uint32_t>(parsed_max_cache_tokens);
      if (args.max_seq_len > args.max_cache_tokens) {
        throw std::runtime_error(
            "--max-seq-len must not exceed --max-cache-tokens");
      }
      if (!is_supported_max_cache_tokens(args.max_cache_tokens)) {
        throw std::runtime_error(
            "only --max-cache-tokens 8, 16, 32, or 64 are currently built");
      }
      if (argc != option_index + 5 && argc != option_index + 6) {
        throw std::runtime_error(
            "usage: lrrt_iree_qwen_decode1_e2e --max-new-tokens <N> "
            "[--max-seq-len <N>] --max-cache-tokens <8|16|32|64> "
            "<kv-cache-layer.vmfb> "
            "<tail.vmfb> <weights-dir> [layers]");
      }
      args.variable_layer_vmfb = argv[option_index + 2];
      args.tail_vmfb = argv[option_index + 3];
      args.weights_dir = argv[option_index + 4];
      if (argc == option_index + 6) {
        const int parsed = std::stoi(argv[option_index + 5]);
        if (parsed <= 0) {
          throw std::runtime_error("layers must be positive");
        }
        args.layers = static_cast<uint32_t>(parsed);
      }
      return args;
    }
    if (!args.prompt_token_ids.empty()) {
      throw std::runtime_error(
          "--prompt-token-ids requires --bundle or --max-cache-tokens");
    }
    if (args.decode_steps == 1) {
      if (argc != option_index + 3 && argc != option_index + 4) {
        throw std::runtime_error(
            "usage: lrrt_iree_qwen_decode1_e2e --max-new-tokens 1 "
            "[--max-seq-len <N>] "
            "<decode1-layer.vmfb> <tail.vmfb> <weights-dir> [layers]");
      }
      args.layer_vmfb = argv[option_index];
      args.tail_vmfb = argv[option_index + 1];
      args.weights_dir = argv[option_index + 2];
      if (argc == option_index + 4) {
        const int parsed = std::stoi(argv[option_index + 3]);
        if (parsed <= 0) {
          throw std::runtime_error("layers must be positive");
        }
        args.layers = static_cast<uint32_t>(parsed);
      }
      return args;
    }
    if (args.decode_steps == 2) {
      if (argc != option_index + 4 && argc != option_index + 5) {
        throw std::runtime_error(
            "usage: lrrt_iree_qwen_decode1_e2e --max-new-tokens 2 "
            "[--max-seq-len <N>] "
            "<decode1-layer.vmfb> <decode2-layer.vmfb> <tail.vmfb> "
            "<weights-dir> [layers]");
      }
      args.layer_vmfb = argv[option_index];
      args.decode2_layer_vmfb = argv[option_index + 1];
      args.tail_vmfb = argv[option_index + 2];
      args.weights_dir = argv[option_index + 3];
      if (argc == option_index + 5) {
        const int parsed = std::stoi(argv[option_index + 4]);
        if (parsed <= 0) {
          throw std::runtime_error("layers must be positive");
        }
        args.layers = static_cast<uint32_t>(parsed);
      }
      return args;
    }
    if (args.decode_steps == 3) {
      if (argc != option_index + 5 && argc != option_index + 6) {
        throw std::runtime_error(
            "usage: lrrt_iree_qwen_decode1_e2e --max-new-tokens 3 "
            "[--max-seq-len <N>] "
            "<decode1-layer.vmfb> <decode2-layer.vmfb> "
            "<decode3-layer.vmfb> <tail.vmfb> <weights-dir> [layers]");
      }
      args.layer_vmfb = argv[option_index];
      args.decode2_layer_vmfb = argv[option_index + 1];
      args.decode3_layer_vmfb = argv[option_index + 2];
      args.tail_vmfb = argv[option_index + 3];
      args.weights_dir = argv[option_index + 4];
      if (argc == option_index + 6) {
        const int parsed = std::stoi(argv[option_index + 5]);
        if (parsed <= 0) {
          throw std::runtime_error("layers must be positive");
        }
        args.layers = static_cast<uint32_t>(parsed);
      }
      return args;
    }
    throw std::runtime_error("--max-new-tokens greater than 3 requires "
                             "--max-cache-tokens <8|16|32|64> "
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

std::vector<const char *> module_paths_for_args(const Args &args) {
  std::vector<const char *> modules = {args.layer_vmfb.c_str(),
                                       args.tail_vmfb.c_str()};
  if (args.max_cache_tokens != 0) {
    modules = {args.variable_layer_vmfb.c_str(), args.tail_vmfb.c_str()};
  } else if (args.decode_steps == 2) {
    modules = {args.layer_vmfb.c_str(), args.decode2_layer_vmfb.c_str(),
               args.tail_vmfb.c_str()};
  } else if (args.decode_steps == 3) {
    modules = {args.layer_vmfb.c_str(), args.decode2_layer_vmfb.c_str(),
               args.decode3_layer_vmfb.c_str(), args.tail_vmfb.c_str()};
  }
  return modules;
}

iree_status_t lookup_module_functions(VmfbRunner *runner, const Args &args,
                                      ModuleFunctions *functions) {
  if (args.max_cache_tokens != 0) {
    IREE_RETURN_IF_ERROR(runner->lookup_function(
        args.variable_layer_export.c_str(), &functions->variable_layer));
  } else {
    IREE_RETURN_IF_ERROR(
        runner->lookup_function(args.layer_export.c_str(), &functions->layer));
  }
  if (args.max_cache_tokens == 0 && args.decode_steps > 1) {
    IREE_RETURN_IF_ERROR(runner->lookup_function(
        args.decode2_layer_export.c_str(), &functions->decode2_layer));
  }
  if (args.max_cache_tokens == 0 && args.decode_steps > 2) {
    IREE_RETURN_IF_ERROR(runner->lookup_function(
        args.decode3_layer_export.c_str(), &functions->decode3_layer));
  }
  IREE_RETURN_IF_ERROR(
      runner->lookup_function(args.tail_export.c_str(), &functions->tail));
  return iree_ok_status();
}

iree_status_t load_qwen_weights(VmfbRunner *runner, const Args &args,
                                LoadedQwenWeights *weights) {
  const std::filesystem::path tail_manifest =
      args.weights_dir / "model_tail" / "weights.json";
  weights->tail = load_model_tail_weights(tail_manifest.c_str());
  if (weights->tail.hidden != kHidden ||
      weights->tail.token_embeddings.empty()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model tail bundle has unexpected Qwen shape");
  }

  if (weights->tail.token_embeddings.size() < kHidden) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model tail bundle does not contain enough token "
                            "embedding rows");
  }

  weights->layers.clear();
  weights->layers.reserve(args.layers);
  weights->layer_views.clear();
  weights->layer_views.resize(args.layers);

  for (uint32_t layer = 0; layer < args.layers; ++layer) {
    const std::filesystem::path layer_manifest =
        args.weights_dir / ("layer_" + std::to_string(layer)) / "weights.json";
    weights->layers.push_back(
        load_decoder_layer_weights(layer_manifest.c_str()));
    if (weights->layers.back().shape.hidden != kHidden ||
        weights->layers.back().shape.kv_dim() != kKvDim ||
        weights->layers.back().shape.intermediate != kIntermediate) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "layer_%u bundle has unexpected Qwen shape",
                              layer);
    }
    IREE_RETURN_IF_ERROR(make_layer_weight_views(runner, weights->layers.back(),
                                                 &weights->layer_views[layer]));
    if (args.verbose_layers) {
      std::printf(
          "iree_qwen startup phase=weights layer=%u/%u status=complete\n",
          layer + 1, args.layers);
      std::fflush(stdout);
    }
  }
  return iree_ok_status();
}

iree_status_t run_decode_loop(VmfbRunner *runner, const Args &args,
                              const ModuleFunctions &functions,
                              const LoadedQwenWeights &weights) {
  const auto decode_start = SteadyClock::now();
  std::vector<LayerKvCache> layer_caches(args.layers);
  if (args.max_cache_tokens != 0) {
    IREE_RETURN_IF_ERROR(initialize_variable_layer_caches(
        runner, args.layers, args.max_cache_tokens, &layer_caches));
  }

  BufferViewPtr hidden;
  const std::vector<uint32_t> prompt_token_ids =
      args.prompt_token_ids.empty()
          ? std::vector<uint32_t>{weights.tail.token_ids.front()}
          : args.prompt_token_ids;
  const uint32_t prompt_len = static_cast<uint32_t>(prompt_token_ids.size());
  const uint32_t total_decode_steps = prompt_len + args.decode_steps - 1;
  std::vector<uint32_t> generated_token_ids;
  generated_token_ids.reserve(args.decode_steps);
  bool stopped_on_eos = false;
  uint32_t current_token = prompt_token_ids.front();
  for (uint32_t step = 0; step < total_decode_steps; ++step) {
    const auto step_start = SteadyClock::now();
    const uint32_t input_token =
        step < prompt_len ? prompt_token_ids[step] : current_token;
    if (step < prompt_len) {
      std::printf(
          "iree_qwen progress phase=prefill token=%u/%u input_token=%u\n",
          step + 1, prompt_len, input_token);
    } else {
      const uint32_t generation_step = step + 2 - prompt_len;
      std::printf("iree_qwen progress phase=decode step=%u/%u input_token=%u\n",
                  generation_step, args.decode_steps, input_token);
    }
    std::fflush(stdout);
    BufferViewPtr token_hidden;
    IREE_RETURN_IF_ERROR(
        make_token_hidden(runner, weights.tail, input_token, &token_hidden));
    if (args.max_cache_tokens != 0) {
      IREE_RETURN_IF_ERROR(run_decode_token_step_with_variable_cache(
          runner, functions.variable_layer, step, args.max_cache_tokens,
          token_hidden.get(), weights.layer_views, &layer_caches,
          args.verbose_layers, &hidden));
    } else {
      IREE_RETURN_IF_ERROR(run_decode_token_step(
          runner, functions.layer, functions.decode2_layer,
          functions.decode3_layer, step, token_hidden.get(),
          weights.layer_views, &layer_caches, args.verbose_layers, &hidden));
    }
    if (step + 1 == prompt_len) {
      std::printf("iree_qwen progress phase=prefill status=submitted tokens=%u "
                  "host_submit_ms=%.3f\n",
                  prompt_len, elapsed_ms(decode_start));
      std::fflush(stdout);
    }
    if (step + 1 >= prompt_len) {
      const uint32_t generation_step = step + 2 - prompt_len;
      BufferViewPtr step_logits;
      IREE_RETURN_IF_ERROR(run_tail(runner, functions.tail, hidden.get(),
                                    weights.tail, &step_logits));
      IREE_RETURN_IF_ERROR(inspect_logits(step_logits.get(), weights.tail.vocab,
                                          generation_step, &current_token));
      std::printf("iree_qwen progress phase=decode step=%u/%u status=complete "
                  "elapsed_ms=%.3f\n",
                  generation_step, args.decode_steps, elapsed_ms(step_start));
      std::fflush(stdout);
      generated_token_ids.push_back(current_token);
      if (args.eos_token_id_set && current_token == args.eos_token_id) {
        stopped_on_eos = true;
        break;
      }
    }
  }
  std::printf("generated_token_ids=[");
  for (size_t i = 0; i < generated_token_ids.size(); ++i) {
    std::printf("%s%u", i == 0 ? "" : ",", generated_token_ids[i]);
  }
  std::printf("]\n");
  std::printf("stop_reason=%s\n",
              stopped_on_eos ? "eos_token" : "max_new_tokens");
  std::printf("iree_qwen summary generated_tokens=%zu elapsed_ms=%.3f\n",
              generated_token_ids.size(), elapsed_ms(decode_start));
  std::fflush(stdout);
  return iree_ok_status();
}

iree_status_t run(const Args &args) {
  const uint32_t prompt_len =
      args.prompt_token_ids.empty()
          ? 1
          : static_cast<uint32_t>(args.prompt_token_ids.size());
  std::printf("iree_qwen config layers=%u prompt_tokens=%u max_new_tokens=%u "
              "max_seq_len=%u cache_capacity=%u\n",
              args.layers, prompt_len, args.decode_steps, args.max_seq_len,
              args.max_cache_tokens);
  std::printf("iree_qwen startup phase=modules status=start\n");
  std::fflush(stdout);
  const auto modules_start = SteadyClock::now();
  VmfbRunner runner;
  IREE_RETURN_IF_ERROR(runner.initialize(module_paths_for_args(args)));

  ModuleFunctions functions;
  IREE_RETURN_IF_ERROR(lookup_module_functions(&runner, args, &functions));
  std::printf(
      "iree_qwen startup phase=modules status=complete elapsed_ms=%.3f\n",
      elapsed_ms(modules_start));
  std::printf("iree_qwen startup phase=weights status=start layers=%u\n",
              args.layers);
  std::fflush(stdout);

  const auto weights_start = SteadyClock::now();
  LoadedQwenWeights weights;
  IREE_RETURN_IF_ERROR(load_qwen_weights(&runner, args, &weights));
  std::printf("iree_qwen startup phase=weights status=complete layers=%u "
              "elapsed_ms=%.3f\n",
              args.layers, elapsed_ms(weights_start));
  std::fflush(stdout);
  IREE_RETURN_IF_ERROR(run_decode_loop(&runner, args, functions, weights));

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
