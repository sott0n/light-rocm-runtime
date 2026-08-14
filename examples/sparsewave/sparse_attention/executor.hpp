#ifndef LRRT_EXAMPLES_SPARSEWAVE_SPARSE_ATTENTION_EXECUTOR_HPP_
#define LRRT_EXAMPLES_SPARSEWAVE_SPARSE_ATTENTION_EXECUTOR_HPP_

#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include <stdint.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace lrrt::examples::sparsewave::sparse_attention {

struct Problem {
  uint32_t output_rows;
  uint32_t key_value_rows;
  uint32_t head_dimension;
  uint32_t value_columns;
};

struct Inputs {
  std::vector<int32_t> row_offsets;
  std::vector<int32_t> column_indices;
  std::vector<float> mask_values;
  std::vector<float> query;
  std::vector<float> transposed_key;
  std::vector<float> value;
};

class Executor {
public:
  Executor(lrrt::Device device, const char *manifest_path, Problem problem)
      : device_(device),
        scores_(device, manifest_path, "sparse_attention_scores"),
        row_max_(device, manifest_path, "sparse_attention_row_max"),
        exp_(device, manifest_path, "sparse_attention_exp"),
        row_sum_(device, manifest_path, "sparse_attention_row_sum"),
        normalize_(device, manifest_path, "sparse_attention_normalize"),
        output_(device, manifest_path, "sparse_attention_output"),
        problem_(problem) {
    validate_manifests();
  }

  std::vector<float> execute(const Inputs &inputs) const {
    validate_inputs(inputs);

    const size_t nonzeros = inputs.mask_values.size();
    std::vector<float> result(output_elements(), 0.0f);
    std::vector<float> scores(nonzeros, 0.0f);
    std::vector<float> row_maximum(problem_.output_rows, 0.0f);
    std::vector<float> row_sum(problem_.output_rows, 0.0f);

    lrrt::DeviceBuffer row_offsets(device_, bytes(inputs.row_offsets));
    lrrt::DeviceBuffer column_indices(device_,
                                      allocated_bytes(inputs.column_indices));
    lrrt::DeviceBuffer mask_values(device_,
                                   allocated_bytes(inputs.mask_values));
    lrrt::DeviceBuffer query(device_, bytes(inputs.query));
    lrrt::DeviceBuffer transposed_key(device_, bytes(inputs.transposed_key));
    lrrt::DeviceBuffer value(device_, bytes(inputs.value));
    lrrt::DeviceBuffer scores_buffer(device_, allocated_bytes(scores));
    lrrt::DeviceBuffer row_maximum_buffer(device_, bytes(row_maximum));
    lrrt::DeviceBuffer row_sum_buffer(device_, bytes(row_sum));
    lrrt::DeviceBuffer output_buffer(device_, bytes(result));

    lrrt::copy_to_device(row_offsets, inputs.row_offsets);
    lrrt::copy_to_device(column_indices, inputs.column_indices);
    lrrt::copy_to_device(mask_values, inputs.mask_values);
    lrrt::copy_to_device(query, inputs.query);
    lrrt::copy_to_device(transposed_key, inputs.transposed_key);
    lrrt::copy_to_device(value, inputs.value);

    {
      lrrt::KernargBuffer args = scores_.make_args();
      args.set("rowOffsets", static_cast<int32_t *>(row_offsets.data()));
      args.set("columnIndices", static_cast<int32_t *>(column_indices.data()));
      args.set("maskValues", static_cast<float *>(mask_values.data()));
      args.set("query", static_cast<float *>(query.data()));
      args.set("transposedKey", static_cast<float *>(transposed_key.data()));
      args.set("scores", static_cast<float *>(scores_buffer.data()));
      scores_.launch(problem_.output_rows, args);
    }
    {
      lrrt::KernargBuffer args = row_max_.make_args();
      args.set("rowOffsets", static_cast<int32_t *>(row_offsets.data()));
      args.set("scores", static_cast<float *>(scores_buffer.data()));
      args.set("rowMaximum", static_cast<float *>(row_maximum_buffer.data()));
      row_max_.launch(problem_.output_rows, args);
    }
    {
      lrrt::KernargBuffer args = exp_.make_args();
      args.set("rowOffsets", static_cast<int32_t *>(row_offsets.data()));
      args.set("rowMaximum", static_cast<float *>(row_maximum_buffer.data()));
      args.set("scores", static_cast<float *>(scores_buffer.data()));
      exp_.launch(problem_.output_rows, args);
    }
    {
      lrrt::KernargBuffer args = row_sum_.make_args();
      args.set("rowOffsets", static_cast<int32_t *>(row_offsets.data()));
      args.set("scores", static_cast<float *>(scores_buffer.data()));
      args.set("rowSum", static_cast<float *>(row_sum_buffer.data()));
      row_sum_.launch(problem_.output_rows, args);
    }
    {
      lrrt::KernargBuffer args = normalize_.make_args();
      args.set("rowOffsets", static_cast<int32_t *>(row_offsets.data()));
      args.set("rowSum", static_cast<float *>(row_sum_buffer.data()));
      args.set("scores", static_cast<float *>(scores_buffer.data()));
      normalize_.launch(problem_.output_rows, args);
    }
    {
      lrrt::KernargBuffer args = output_.make_args();
      args.set("rowOffsets", static_cast<int32_t *>(row_offsets.data()));
      args.set("columnIndices", static_cast<int32_t *>(column_indices.data()));
      args.set("scores", static_cast<float *>(scores_buffer.data()));
      args.set("value", static_cast<float *>(value.data()));
      args.set("output", static_cast<float *>(output_buffer.data()));
      output_.launch(static_cast<uint32_t>(output_elements()), args);
    }

    device_.synchronize();
    lrrt::copy_to_host(result, output_buffer);
    return result;
  }

  const Problem &problem() const { return problem_; }

private:
  template <typename T> static size_t bytes(const std::vector<T> &values) {
    return values.size() * sizeof(T);
  }

  template <typename T>
  static size_t allocated_bytes(const std::vector<T> &values) {
    return values.empty() ? sizeof(T) : bytes(values);
  }

  size_t output_elements() const {
    return static_cast<size_t>(problem_.output_rows) * problem_.value_columns;
  }

  static void require_manifest(const lrrt::Bundle &bundle,
                               const std::vector<std::string> &expected) {
    const lrrt::KernelManifest &manifest = bundle.manifest();
    if (manifest.args.size() != expected.size()) {
      throw std::runtime_error(
          "SparseWave SparseAttention bundle has an incompatible argument ABI");
    }
    for (size_t i = 0; i < expected.size(); ++i) {
      if (manifest.args[i].name != expected[i] ||
          manifest.args[i].type != "ptr") {
        throw std::runtime_error("SparseWave SparseAttention bundle has an "
                                 "incompatible argument ABI");
      }
    }
    if (manifest.workspace_bytes != 0) {
      throw std::runtime_error(
          "SparseWave SparseAttention bundle unexpectedly requires workspace");
    }
  }

  void validate_manifests() const {
    if (problem_.output_rows == 0 || problem_.key_value_rows == 0 ||
        problem_.head_dimension == 0 || problem_.value_columns == 0) {
      throw std::invalid_argument(
          "SparseWave SparseAttention dimensions must be non-zero");
    }
    require_manifest(scores_, {"rowOffsets", "columnIndices", "maskValues",
                               "query", "transposedKey", "scores"});
    require_manifest(row_max_, {"rowOffsets", "scores", "rowMaximum"});
    require_manifest(exp_, {"rowOffsets", "rowMaximum", "scores"});
    require_manifest(row_sum_, {"rowOffsets", "scores", "rowSum"});
    require_manifest(normalize_, {"rowOffsets", "rowSum", "scores"});
    require_manifest(
        output_, {"rowOffsets", "columnIndices", "scores", "value", "output"});
  }

  void validate_inputs(const Inputs &inputs) const {
    if (inputs.row_offsets.size() !=
        static_cast<size_t>(problem_.output_rows) + 1) {
      throw std::invalid_argument(
          "SparseWave SparseAttention row-offset size mismatch");
    }
    if (inputs.column_indices.size() != inputs.mask_values.size()) {
      throw std::invalid_argument(
          "SparseWave SparseAttention nonzero size mismatch");
    }
    if (inputs.mask_values.size() > static_cast<size_t>(INT32_MAX) ||
        inputs.row_offsets.front() != 0 ||
        inputs.row_offsets.back() !=
            static_cast<int32_t>(inputs.mask_values.size())) {
      throw std::invalid_argument(
          "SparseWave SparseAttention CSR bounds are invalid");
    }
    for (size_t i = 1; i < inputs.row_offsets.size(); ++i) {
      if (inputs.row_offsets[i] < inputs.row_offsets[i - 1]) {
        throw std::invalid_argument(
            "SparseWave SparseAttention row offsets must be nondecreasing");
      }
    }
    for (int32_t column : inputs.column_indices) {
      if (column < 0 ||
          static_cast<uint32_t>(column) >= problem_.key_value_rows) {
        throw std::invalid_argument(
            "SparseWave SparseAttention column index is out of range");
      }
    }
    const size_t query_elements =
        static_cast<size_t>(problem_.output_rows) * problem_.head_dimension;
    const size_t key_elements =
        static_cast<size_t>(problem_.head_dimension) * problem_.key_value_rows;
    const size_t value_elements =
        static_cast<size_t>(problem_.key_value_rows) * problem_.value_columns;
    if (inputs.query.size() != query_elements ||
        inputs.transposed_key.size() != key_elements ||
        inputs.value.size() != value_elements) {
      throw std::invalid_argument(
          "SparseWave SparseAttention dense input size mismatch");
    }
  }

  lrrt::Device device_;
  lrrt::Bundle scores_;
  lrrt::Bundle row_max_;
  lrrt::Bundle exp_;
  lrrt::Bundle row_sum_;
  lrrt::Bundle normalize_;
  lrrt::Bundle output_;
  Problem problem_;
};

} // namespace lrrt::examples::sparsewave::sparse_attention

#endif // LRRT_EXAMPLES_SPARSEWAVE_SPARSE_ATTENTION_EXECUTOR_HPP_
