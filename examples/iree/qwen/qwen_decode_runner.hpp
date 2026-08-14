#ifndef LRRT_EXAMPLES_IREE_QWEN_DECODE_RUNNER_HPP_
#define LRRT_EXAMPLES_IREE_QWEN_DECODE_RUNNER_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "executor/iree/vmfb_runner.hpp"
#include "iree/base/api.h"
#include "iree/hal/api.h"

namespace lrrt::examples::iree::qwen {

using lrrt::iree_executor::BufferViewPtr;
using lrrt::iree_executor::BufferViewReplacement;
using lrrt::iree_executor::VmfbRunner;

enum class QwenDecodeInputIndex : iree_host_size_t {
  kInput = 0,
  kQuery = 1,
  kOldKeyCacheTransposed = 2,
  kNewKey = 3,
  kOldValueCache = 4,
  kNewValue = 5,
  kCos = 6,
  kSin = 7,
  kGateWeight = 8,
  kUpWeight = 9,
  kDownWeight = 10,
};

enum class QwenDecodeOutputIndex : iree_host_size_t {
  kKeyCacheTransposed = 0,
  kValueCache = 1,
  kHidden = 2,
};

struct QwenDecodeStepSpecs {
  std::string input = "2x2xf32=1 2 3 4";
  std::string query = "2x2xf32=1 2 3 4";
  std::string old_key_cache_transposed = "2x3xf32=0 0 0 0 0 0";
  std::string new_key = "2xf32=0 0";
  std::string old_value_cache = "3x2xf32=2 4 4 6 0 0";
  std::string new_value;
  std::string cos = "2xf32=1 0";
  std::string sin = "2xf32=0 1";
  std::string gate_weight = "2x2xf32=1 0 0 1";
  std::string up_weight = "2x2xf32=1 0 0 1";
  std::string down_weight = "2x2xf32=1 0 0 1";
};

inline std::vector<std::string>
qwen_decode_input_specs(const QwenDecodeStepSpecs &specs) {
  return {
      specs.input,
      specs.query,
      specs.old_key_cache_transposed,
      specs.new_key,
      specs.old_value_cache,
      specs.new_value,
      specs.cos,
      specs.sin,
      specs.gate_weight,
      specs.up_weight,
      specs.down_weight,
  };
}

inline QwenDecodeStepSpecs qwen_decode_stub_step(const char *new_value_spec) {
  QwenDecodeStepSpecs specs;
  specs.new_value = new_value_spec;
  return specs;
}

inline iree_status_t qwen_decode_step(VmfbRunner *runner,
                                      const iree_vm_function_t &function,
                                      const QwenDecodeStepSpecs &specs,
                                      iree_hal_buffer_view_t *key_cache,
                                      iree_hal_buffer_view_t *value_cache,
                                      std::vector<BufferViewPtr> *outputs) {
  std::vector<BufferViewReplacement> replacements;
  if (key_cache) {
    replacements.push_back({static_cast<iree_host_size_t>(
                                QwenDecodeInputIndex::kOldKeyCacheTransposed),
                            key_cache});
  }
  if (value_cache) {
    replacements.push_back(
        {static_cast<iree_host_size_t>(QwenDecodeInputIndex::kOldValueCache),
         value_cache});
  }
  return runner->invoke(function, qwen_decode_input_specs(specs), replacements,
                        /*output_count=*/3, outputs);
}

inline bool qwen_decode_expect_hidden(iree_hal_buffer_view_t *view,
                                      const std::array<float, 4> &expected,
                                      const char *label) {
  if (iree_hal_buffer_view_shape_rank(view) != 2 ||
      iree_hal_buffer_view_shape_dim(view, 0) != 2 ||
      iree_hal_buffer_view_shape_dim(view, 1) != 2) {
    std::fprintf(stderr, "%s: unexpected shape\n", label);
    return false;
  }

  float values[4] = {};
  iree_hal_buffer_t *buffer = iree_hal_buffer_view_buffer(view);
  iree_status_t status =
      iree_hal_buffer_map_read(buffer, 0, values, sizeof(values));
  if (!iree_status_is_ok(status)) {
    std::fprintf(stderr, "%s readback failed\n", label);
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return false;
  }

  for (int i = 0; i < 4; ++i) {
    if (std::fabs(values[i] - expected[static_cast<size_t>(i)]) > 1e-3f) {
      std::fprintf(stderr, "%s[%d]: expected %.6g, got %.6g\n", label, i,
                   expected[static_cast<size_t>(i)], values[i]);
      return false;
    }
  }
  return true;
}

} // namespace lrrt::examples::iree::qwen

#endif // LRRT_EXAMPLES_IREE_QWEN_DECODE_RUNNER_HPP_
