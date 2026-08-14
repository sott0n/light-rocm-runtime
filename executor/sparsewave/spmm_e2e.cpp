#include "lrrt/executor/sparsewave/spmm.hpp"

#include <math.h>
#include <stdio.h>

#include <stdexcept>
#include <vector>

#ifndef LRRT_SPARSEWAVE_SPMM_MANIFEST
#define LRRT_SPARSEWAVE_SPMM_MANIFEST "manifest.json"
#endif

namespace {

std::vector<float>
reference_spmm(const lrrt::executor::sparsewave::SpmmProblem &problem,
               const lrrt::executor::sparsewave::CsrMatrix &matrix,
               const std::vector<float> &rhs) {
  std::vector<float> output(
      static_cast<size_t>(problem.rows) * problem.rhs_columns, 0.0f);
  for (uint32_t row = 0; row < problem.rows; ++row) {
    for (int32_t index = matrix.row_offsets[row];
         index < matrix.row_offsets[row + 1]; ++index) {
      const uint32_t column = matrix.column_indices[index];
      for (uint32_t lane = 0; lane < problem.rhs_columns; ++lane) {
        output[row * problem.rhs_columns + lane] +=
            matrix.values[index] * rhs[column * problem.rhs_columns + lane];
      }
    }
  }
  return output;
}

} // namespace

int main() {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      printf("sparsewave_spmm: skipped, no GPU devices\n");
      return 0;
    }

    lrrt::Device device = runtime.open_device(0);
    const lrrt::executor::sparsewave::SpmmProblem problem{4, 8, 4};
    const lrrt::executor::sparsewave::CsrMatrix matrix{
        {0, 2, 4, 6, 8},
        {0, 3, 1, 7, 2, 4, 5, 6},
        {1.0f, 2.0f, -1.0f, 0.5f, 3.0f, -2.0f, 4.0f, 1.5f}};

    std::vector<float> rhs(problem.columns * problem.rhs_columns);
    for (uint32_t row = 0; row < problem.columns; ++row) {
      for (uint32_t column = 0; column < problem.rhs_columns; ++column) {
        rhs[row * problem.rhs_columns + column] =
            static_cast<float>(row * 4 + column + 1) * 0.25f;
      }
    }

    lrrt::executor::sparsewave::SpmmExecutor executor(
        device, LRRT_SPARSEWAVE_SPMM_MANIFEST, problem);
    const std::vector<float> actual = executor.execute(matrix, rhs);
    const std::vector<float> expected = reference_spmm(problem, matrix, rhs);
    for (size_t i = 0; i < actual.size(); ++i) {
      if (fabsf(actual[i] - expected[i]) > 1.0e-5f) {
        throw std::runtime_error("SparseWave SpMM result mismatch at element " +
                                 std::to_string(i));
      }
    }

    printf("sparsewave_spmm: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
