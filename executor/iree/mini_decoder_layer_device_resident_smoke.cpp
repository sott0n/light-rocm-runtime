#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "executor/iree/vmfb_runner.hpp"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"

namespace {

using lrrt::iree_executor::BufferViewPtr;
using lrrt::iree_executor::BufferViewReplacement;
using lrrt::iree_executor::VmfbRunner;

bool report_status(iree_status_t status, const char *label) {
  if (iree_status_is_ok(status)) {
    return true;
  }
  std::fprintf(stderr, "%s failed\n", label);
  iree_status_fprint(stderr, status);
  iree_status_free(status);
  return false;
}

iree_status_t invoke_step(VmfbRunner *runner, iree_hal_buffer_view_t *key_cache,
                          iree_hal_buffer_view_t *value_cache,
                          const char *new_value_spec,
                          std::vector<BufferViewPtr> *outputs) {
  const std::vector<std::string> specs = {
      "2x2xf32=1 2 3 4", "2x2xf32=1 2 3 4",     "2x3xf32=0 0 0 0 0 0",
      "2xf32=0 0",       "3x2xf32=2 4 4 6 0 0", new_value_spec,
      "2xf32=1 0",       "2xf32=0 1",           "2x2xf32=1 0 0 1",
      "2x2xf32=1 0 0 1", "2x2xf32=1 0 0 1",
  };

  std::vector<BufferViewReplacement> replacements;
  if (key_cache) {
    replacements.push_back({2, key_cache});
  }
  if (value_cache) {
    replacements.push_back({4, value_cache});
  }
  return runner->invoke(specs, replacements, 3, outputs);
}

bool expect_matrix(iree_hal_buffer_view_t *view, const float (&expected)[4],
                   const char *label) {
  if (iree_hal_buffer_view_shape_rank(view) != 2 ||
      iree_hal_buffer_view_shape_dim(view, 0) != 2 ||
      iree_hal_buffer_view_shape_dim(view, 1) != 2) {
    std::fprintf(stderr, "%s: unexpected shape\n", label);
    return false;
  }

  float values[4] = {};
  iree_hal_buffer_t *buffer = iree_hal_buffer_view_buffer(view);
  if (!report_status(
          iree_hal_buffer_map_read(buffer, 0, values, sizeof(values)), label)) {
    return false;
  }

  for (int i = 0; i < 4; ++i) {
    if (std::fabs(values[i] - expected[i]) > 1e-3f) {
      std::fprintf(stderr, "%s[%d]: expected %.6g, got %.6g\n", label, i,
                   expected[i], values[i]);
      return false;
    }
  }
  return true;
}

iree_status_t run_smoke(const char *module_path) {
  VmfbRunner runner;
  IREE_RETURN_IF_ERROR(runner.initialize(
      module_path, "mini_decoder_layer_rope_kv_cache_outputs"));

  std::vector<BufferViewPtr> first_outputs;
  IREE_RETURN_IF_ERROR(invoke_step(&runner, /*key_cache=*/nullptr,
                                   /*value_cache=*/nullptr, "2xf32=6 8",
                                   &first_outputs));

  const float first_expected[4] = {30.0f, 72.0f, 56.0f, 110.0f};
  if (!expect_matrix(first_outputs[2].get(), first_expected,
                     "first decoder result")) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "first decoder result did not match");
  }

  std::vector<BufferViewPtr> second_outputs;
  IREE_RETURN_IF_ERROR(invoke_step(&runner, first_outputs[0].get(),
                                   first_outputs[1].get(), "2xf32=8 10",
                                   &second_outputs));

  const float second_expected[4] = {
      340.0f / 9.0f,
      754.0f / 9.0f,
      598.0f / 9.0f,
      1120.0f / 9.0f,
  };
  if (!expect_matrix(second_outputs[2].get(), second_expected,
                     "second decoder result")) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "second decoder result did not match");
  }

  return iree_ok_status();
}

} // namespace

int main(int argc, char **argv) {
  iree_flags_set_usage("lrrt_iree_mini_decoder_layer_device_resident_smoke",
                       "Runs a mini decoder layer twice with device-resident "
                       "KV cache handoff.");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  if (argc != 2) {
    std::fprintf(stderr,
                 "usage: %s <mini_decoder_layer_rope_kv_cache_outputs.vmfb>\n",
                 argv[0]);
    return 2;
  }

  iree_status_t status = run_smoke(argv[1]);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }

  std::puts("iree_mini_decoder_layer_device_resident: ok");
  return 0;
}
