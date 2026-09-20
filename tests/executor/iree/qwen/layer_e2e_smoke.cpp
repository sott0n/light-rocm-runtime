#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

#include "executor/iree/vmfb_runner.hpp"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"

namespace {

using lrrt::iree_executor::BufferViewPtr;
using lrrt::iree_executor::VmfbRunner;

constexpr iree_hal_dim_t kHidden = 896;
constexpr iree_hal_dim_t kKvDim = 128;
constexpr iree_hal_dim_t kIntermediate = 4864;

iree_status_t append_view(VmfbRunner *runner, std::vector<float> data,
                          std::vector<iree_hal_dim_t> shape,
                          std::vector<BufferViewPtr> *views) {
  BufferViewPtr view;
  IREE_RETURN_IF_ERROR(runner->make_f32_buffer_view(data, shape, &view));
  views->push_back(std::move(view));
  return iree_ok_status();
}

std::vector<float> diagonal_matrix(iree_hal_dim_t rows,
                                   iree_hal_dim_t columns) {
  std::vector<float> values(static_cast<size_t>(rows * columns), 0.0f);
  const iree_hal_dim_t diagonal = std::min(rows, columns);
  for (iree_hal_dim_t i = 0; i < diagonal; ++i) {
    values[static_cast<size_t>(i * columns + i)] = 1.0f;
  }
  return values;
}

iree_status_t expect_values(iree_hal_buffer_view_t *view,
                            const std::vector<iree_hal_dim_t> &shape,
                            const std::vector<float> &expected, float tolerance,
                            const char *label) {
  if (iree_hal_buffer_view_shape_rank(view) != shape.size()) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "%s rank does not match",
                            label);
  }
  size_t count = 1;
  for (iree_host_size_t i = 0; i < shape.size(); ++i) {
    if (iree_hal_buffer_view_shape_dim(view, i) != shape[i]) {
      return iree_make_status(IREE_STATUS_DATA_LOSS, "%s shape does not match",
                              label);
    }
    count *= static_cast<size_t>(shape[i]);
  }
  if (count != expected.size()) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "%s expected value count does not match", label);
  }

  std::vector<float> values(count);
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_map_read(iree_hal_buffer_view_buffer(view), 0,
                               values.data(), values.size() * sizeof(float)));
  for (size_t i = 0; i < values.size(); ++i) {
    if (std::fabs(values[i] - expected[i]) > tolerance) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "%s[%zu] mismatch: expected %.7g, got %.7g",
                              label, i, expected[i], values[i]);
    }
  }
  return iree_ok_status();
}

iree_status_t run_smoke(const char *module_path) {
  VmfbRunner runner;
  IREE_RETURN_IF_ERROR(runner.initialize(module_path));

  iree_vm_function_t function = {};
  IREE_RETURN_IF_ERROR(runner.lookup_function("qwen_decode1_layer", &function));

  std::vector<BufferViewPtr> input_views;
  input_views.reserve(10);
  std::vector<float> input(kHidden);
  for (iree_hal_dim_t i = 0; i < kHidden; ++i) {
    input[static_cast<size_t>(i)] = 0.5f + 0.001f * static_cast<float>(i % 31);
  }
  IREE_RETURN_IF_ERROR(append_view(&runner, input, {1, kHidden}, &input_views));
  IREE_RETURN_IF_ERROR(append_view(&runner, std::vector<float>(kHidden, 1.0f),
                                   {kHidden}, &input_views));
  IREE_RETURN_IF_ERROR(append_view(
      &runner, std::vector<float>(static_cast<size_t>(kHidden * kHidden), 0.0f),
      {kHidden, kHidden}, &input_views));
  IREE_RETURN_IF_ERROR(append_view(&runner, diagonal_matrix(kHidden, kKvDim),
                                   {kHidden, kKvDim}, &input_views));
  IREE_RETURN_IF_ERROR(append_view(&runner, diagonal_matrix(kHidden, kKvDim),
                                   {kHidden, kKvDim}, &input_views));
  IREE_RETURN_IF_ERROR(append_view(&runner, diagonal_matrix(kHidden, kHidden),
                                   {kHidden, kHidden}, &input_views));
  IREE_RETURN_IF_ERROR(append_view(&runner, std::vector<float>(kHidden, 1.0f),
                                   {kHidden}, &input_views));
  IREE_RETURN_IF_ERROR(append_view(&runner,
                                   diagonal_matrix(kHidden, kIntermediate),
                                   {kHidden, kIntermediate}, &input_views));
  IREE_RETURN_IF_ERROR(append_view(&runner,
                                   diagonal_matrix(kHidden, kIntermediate),
                                   {kHidden, kIntermediate}, &input_views));
  IREE_RETURN_IF_ERROR(append_view(&runner,
                                   diagonal_matrix(kIntermediate, kHidden),
                                   {kIntermediate, kHidden}, &input_views));

  std::vector<iree_hal_buffer_view_t *> inputs;
  inputs.reserve(input_views.size());
  for (const BufferViewPtr &view : input_views) {
    inputs.push_back(view.get());
  }

  std::vector<BufferViewPtr> outputs;
  IREE_RETURN_IF_ERROR(runner.invoke_views(function, inputs, 3, &outputs));

  float attention_sum_square = 0.0f;
  for (float value : input) {
    attention_sum_square += value * value;
  }
  const float attention_scale =
      1.0f / std::sqrt(attention_sum_square / kHidden + 1.0e-6f);
  std::vector<float> normalized(kHidden);
  for (iree_hal_dim_t i = 0; i < kHidden; ++i) {
    normalized[static_cast<size_t>(i)] =
        input[static_cast<size_t>(i)] * attention_scale;
  }

  std::vector<float> expected_cache(normalized.begin(),
                                    normalized.begin() + kKvDim);
  std::vector<float> residual(kHidden);
  float mlp_sum_square = 0.0f;
  for (iree_hal_dim_t i = 0; i < kHidden; ++i) {
    const iree_hal_dim_t context_index =
        (i / 448) * 64 + static_cast<iree_hal_dim_t>(i % 64);
    residual[static_cast<size_t>(i)] =
        input[static_cast<size_t>(i)] +
        normalized[static_cast<size_t>(context_index)];
    mlp_sum_square +=
        residual[static_cast<size_t>(i)] * residual[static_cast<size_t>(i)];
  }

  const float mlp_scale = 1.0f / std::sqrt(mlp_sum_square / kHidden + 1.0e-6f);
  std::vector<float> expected_hidden(kHidden);
  for (iree_hal_dim_t i = 0; i < kHidden; ++i) {
    const float normalized_value = residual[static_cast<size_t>(i)] * mlp_scale;
    const float sigmoid = 1.0f / (1.0f + std::exp(-normalized_value));
    expected_hidden[static_cast<size_t>(i)] =
        residual[static_cast<size_t>(i)] +
        normalized_value * sigmoid * normalized_value;
  }

  IREE_RETURN_IF_ERROR(expect_values(outputs[0].get(), {1, kKvDim},
                                     expected_cache, 2.0e-3f, "key cache"));
  IREE_RETURN_IF_ERROR(expect_values(outputs[1].get(), {1, kKvDim},
                                     expected_cache, 2.0e-3f, "value cache"));
  IREE_RETURN_IF_ERROR(expect_values(outputs[2].get(), {1, kHidden},
                                     expected_hidden, 4.0e-3f, "hidden state"));
  return iree_ok_status();
}

} // namespace

int main(int argc, char **argv) {
  iree_flags_set_usage("lrrt_iree_qwen_layer_e2e_smoke",
                       "Runs one Qwen-shaped decoder layer through lrrt.");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <qwen_decode1_layer.vmfb>\n", argv[0]);
    return 2;
  }

  iree_status_t status = run_smoke(argv[1]);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }
  std::puts("iree_qwen_layer_e2e: ok");
  return 0;
}
