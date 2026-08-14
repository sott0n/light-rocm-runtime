#ifndef LRRT_EXAMPLES_SPARSEWAVE_SPMM_EXECUTOR_HPP_
#define LRRT_EXAMPLES_SPARSEWAVE_SPMM_EXECUTOR_HPP_

#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include <stdint.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace lrrt::examples::sparsewave::spmm {

struct Problem {
  uint32_t rows;
  uint32_t columns;
  uint32_t rhs_columns;
};

struct Inputs {
  std::vector<int32_t> row_offsets;
  std::vector<int32_t> column_indices;
  std::vector<float> values;
  std::vector<float> rhs;
};

class Executor {
public:
  Executor(lrrt::Device device, const char *manifest_path, Problem problem)
      : device_(device), bundle_(device, manifest_path, "spmm"),
        problem_(problem) {
    validate_manifest();
  }

  std::vector<float> execute(const Inputs &inputs) const {
    validate_inputs(inputs);

    std::vector<float> output(output_elements(), 0.0f);
    lrrt::DeviceBuffer row_offsets(device_, bytes(inputs.row_offsets));
    lrrt::DeviceBuffer column_indices(device_, bytes(inputs.column_indices));
    lrrt::DeviceBuffer values(device_, bytes(inputs.values));
    lrrt::DeviceBuffer rhs(device_, bytes(inputs.rhs));
    lrrt::DeviceBuffer output_buffer(device_, bytes(output));

    lrrt::copy_to_device(row_offsets, inputs.row_offsets);
    lrrt::copy_to_device(column_indices, inputs.column_indices);
    lrrt::copy_to_device(values, inputs.values);
    lrrt::copy_to_device(rhs, inputs.rhs);
    lrrt::copy_to_device(output_buffer, output);

    lrrt::KernargBuffer args = bundle_.make_args();
    args.set("rowOffsets", static_cast<int32_t *>(row_offsets.data()));
    args.set("columnIndices", static_cast<int32_t *>(column_indices.data()));
    args.set("values", static_cast<float *>(values.data()));
    args.set("rhs", static_cast<float *>(rhs.data()));
    args.set("output", static_cast<float *>(output_buffer.data()));
    args.bind_optional_nulls();

    bundle_.launch(problem_.rows, args);
    device_.synchronize();
    lrrt::copy_to_host(output, output_buffer);
    return output;
  }

  const Problem &problem() const { return problem_; }

private:
  template <typename T> static size_t bytes(const std::vector<T> &values) {
    return values.size() * sizeof(T);
  }

  size_t output_elements() const {
    return static_cast<size_t>(problem_.rows) * problem_.rhs_columns;
  }

  void validate_manifest() const {
    static const char *const expected[] = {"rowOffsets", "columnIndices",
                                           "values", "rhs", "output"};
    const lrrt::KernelManifest &manifest = bundle_.manifest();
    if (manifest.args.size() != 5) {
      throw std::runtime_error(
          "SparseWave SpMM bundle must have exactly five arguments");
    }
    for (size_t i = 0; i < manifest.args.size(); ++i) {
      if (manifest.args[i].name != expected[i] ||
          manifest.args[i].type != "ptr") {
        throw std::runtime_error(
            "SparseWave SpMM bundle has an incompatible argument ABI");
      }
    }
    if (manifest.workspace_bytes != 0) {
      throw std::runtime_error(
          "SparseWave SpMM bundle unexpectedly requires workspace");
    }
    if (problem_.rows == 0 || problem_.columns == 0 ||
        problem_.rhs_columns == 0) {
      throw std::invalid_argument(
          "SparseWave SpMM dimensions must be non-zero");
    }
  }

  void validate_inputs(const Inputs &inputs) const {
    if (inputs.row_offsets.size() != static_cast<size_t>(problem_.rows) + 1) {
      throw std::invalid_argument("SparseWave SpMM row-offset size mismatch");
    }
    if (inputs.column_indices.size() != inputs.values.size()) {
      throw std::invalid_argument("SparseWave SpMM nonzero size mismatch");
    }
    const size_t rhs_elements =
        static_cast<size_t>(problem_.columns) * problem_.rhs_columns;
    if (inputs.rhs.size() != rhs_elements) {
      throw std::invalid_argument("SparseWave SpMM RHS size mismatch");
    }
    if (inputs.values.size() > static_cast<size_t>(INT32_MAX) ||
        inputs.row_offsets.front() != 0 ||
        inputs.row_offsets.back() !=
            static_cast<int32_t>(inputs.values.size())) {
      throw std::invalid_argument("SparseWave SpMM CSR bounds are invalid");
    }
    for (size_t i = 1; i < inputs.row_offsets.size(); ++i) {
      if (inputs.row_offsets[i] < inputs.row_offsets[i - 1]) {
        throw std::invalid_argument(
            "SparseWave SpMM row offsets must be nondecreasing");
      }
    }
    for (int32_t column : inputs.column_indices) {
      if (column < 0 || static_cast<uint32_t>(column) >= problem_.columns) {
        throw std::invalid_argument(
            "SparseWave SpMM column index is out of range");
      }
    }
  }

  lrrt::Device device_;
  lrrt::Bundle bundle_;
  Problem problem_;
};

} // namespace lrrt::examples::sparsewave::spmm

#endif // LRRT_EXAMPLES_SPARSEWAVE_SPMM_EXECUTOR_HPP_
