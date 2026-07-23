#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "executor/iree/vmfb_runner.hpp"
#include "executor/triton/mini_decoder_weights.hpp"
#include "iree/base/api.h"
#include "iree/hal/api.h"

namespace {

using lrrt::executor::triton::mini::DecoderLayerWeights;
using lrrt::executor::triton::mini::load_decoder_layer_weights;
using lrrt::executor::triton::mini::load_model_tail_weights;
using lrrt::executor::triton::mini::ModelTailWeights;
using lrrt::iree_executor::BufferViewPtr;
using lrrt::iree_executor::VmfbRunner;

constexpr uint32_t kHidden = 896;
constexpr uint32_t kKvDim = 128;
constexpr uint32_t kIntermediate = 4864;
constexpr uint32_t kDefaultLayers = 24;

struct Args {
  const char *layer_vmfb = nullptr;
  const char *tail_vmfb = nullptr;
  std::filesystem::path weights_dir;
  uint32_t layers = kDefaultLayers;
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

iree_status_t run_layer(VmfbRunner *runner, const iree_vm_function_t &function,
                        iree_hal_buffer_view_t *hidden,
                        const DecoderLayerWeights &weights,
                        BufferViewPtr *out_hidden) {
  BufferViewPtr attention_norm_weight;
  BufferViewPtr q_weight;
  BufferViewPtr k_weight;
  BufferViewPtr v_weight;
  BufferViewPtr out_weight;
  BufferViewPtr mlp_norm_weight;
  BufferViewPtr gate_weight;
  BufferViewPtr up_weight;
  BufferViewPtr down_weight;

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
                                 {kHidden}, &attention_norm_weight));
  IREE_RETURN_IF_ERROR(
      make_view(runner, q_weight_t, {kHidden, kHidden}, &q_weight));
  IREE_RETURN_IF_ERROR(
      make_view(runner, k_weight_t, {kHidden, kKvDim}, &k_weight));
  IREE_RETURN_IF_ERROR(
      make_view(runner, v_weight_t, {kHidden, kKvDim}, &v_weight));
  IREE_RETURN_IF_ERROR(
      make_view(runner, out_weight_t, {kHidden, kHidden}, &out_weight));
  IREE_RETURN_IF_ERROR(
      make_view(runner, weights.mlp_norm_weight, {kHidden}, &mlp_norm_weight));
  IREE_RETURN_IF_ERROR(
      make_view(runner, gate_weight_t, {kHidden, kIntermediate}, &gate_weight));
  IREE_RETURN_IF_ERROR(
      make_view(runner, up_weight_t, {kHidden, kIntermediate}, &up_weight));
  IREE_RETURN_IF_ERROR(
      make_view(runner, down_weight_t, {kIntermediate, kHidden}, &down_weight));

  std::vector<iree_hal_buffer_view_t *> inputs = {
      hidden,
      attention_norm_weight.get(),
      q_weight.get(),
      k_weight.get(),
      v_weight.get(),
      out_weight.get(),
      mlp_norm_weight.get(),
      gate_weight.get(),
      up_weight.get(),
      down_weight.get(),
  };

  std::vector<BufferViewPtr> outputs;
  IREE_RETURN_IF_ERROR(runner->invoke_views(function, inputs, 3, &outputs));
  out_hidden->reset(outputs[2].release());
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

bool inspect_logits(iree_hal_buffer_view_t *view, uint32_t vocab) {
  if (iree_hal_buffer_view_shape_rank(view) != 2 ||
      iree_hal_buffer_view_shape_dim(view, 0) != 1 ||
      iree_hal_buffer_view_shape_dim(view, 1) != vocab) {
    std::fprintf(stderr, "unexpected logits shape\n");
    return false;
  }

  std::vector<float> logits(vocab);
  iree_hal_buffer_t *buffer = iree_hal_buffer_view_buffer(view);
  iree_status_t status = iree_hal_buffer_map_read(
      buffer, 0, logits.data(), logits.size() * sizeof(float));
  if (!iree_status_is_ok(status)) {
    std::fprintf(stderr, "logits readback failed\n");
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return false;
  }

  uint32_t top_index = 0;
  float top_value = -std::numeric_limits<float>::infinity();
  for (uint32_t i = 0; i < vocab; ++i) {
    const float value = logits[i];
    if (!std::isfinite(value)) {
      std::fprintf(stderr, "logits[%u] is non-finite: %f\n", i, value);
      return false;
    }
    if (value > top_value) {
      top_value = value;
      top_index = i;
    }
  }
  std::printf("iree_qwen_decode1_e2e: ok top_token=%u logit=%g vocab=%u\n",
              top_index, top_value, vocab);
  return true;
}

Args parse_args(int argc, char **argv) {
  if (argc != 4 && argc != 5) {
    throw std::runtime_error(
        "usage: lrrt_iree_qwen_decode1_e2e <layer.vmfb> <tail.vmfb> "
        "<weights-dir> [layers]");
  }
  Args args;
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
  IREE_RETURN_IF_ERROR(runner.initialize(modules));

  iree_vm_function_t layer_function = {};
  iree_vm_function_t tail_function = {};
  IREE_RETURN_IF_ERROR(
      runner.lookup_function("qwen_decode1_layer", &layer_function));
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

  BufferViewPtr hidden;
  std::vector<float> initial_hidden(tail_weights.token_embeddings.begin(),
                                    tail_weights.token_embeddings.begin() +
                                        kHidden);
  IREE_RETURN_IF_ERROR(
      make_view(&runner, initial_hidden, {1, kHidden}, &hidden));

  for (uint32_t layer = 0; layer < args.layers; ++layer) {
    const std::filesystem::path layer_manifest =
        args.weights_dir / ("layer_" + std::to_string(layer)) / "weights.json";
    DecoderLayerWeights layer_weights =
        load_decoder_layer_weights(layer_manifest.c_str());
    if (layer_weights.shape.hidden != kHidden ||
        layer_weights.shape.kv_dim() != kKvDim ||
        layer_weights.shape.intermediate != kIntermediate) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "layer_%u bundle has unexpected Qwen shape",
                              layer);
    }
    BufferViewPtr next_hidden;
    IREE_RETURN_IF_ERROR(run_layer(&runner, layer_function, hidden.get(),
                                   layer_weights, &next_hidden));
    hidden = std::move(next_hidden);
    std::printf("layer %u/%u complete\n", layer + 1, args.layers);
  }

  BufferViewPtr logits;
  IREE_RETURN_IF_ERROR(
      run_tail(&runner, tail_function, hidden.get(), tail_weights, &logits));
  if (!inspect_logits(logits.get(), tail_weights.vocab)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "Qwen decode1 logits check failed");
  }
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
