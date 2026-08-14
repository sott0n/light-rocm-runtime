#include <array>
#include <cstdio>
#include <vector>

#include "examples/iree/qwen/qwen_decode_runner.hpp"
#include "executor/iree/vmfb_runner.hpp"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"

namespace {

using lrrt::examples::iree::qwen::qwen_decode_expect_hidden;
using lrrt::examples::iree::qwen::qwen_decode_step;
using lrrt::examples::iree::qwen::qwen_decode_stub_step;
using lrrt::examples::iree::qwen::QwenDecodeOutputIndex;
using lrrt::iree_executor::BufferViewPtr;
using lrrt::iree_executor::VmfbRunner;

iree_status_t run_smoke(const char *module_path) {
  VmfbRunner runner;
  IREE_RETURN_IF_ERROR(runner.initialize(module_path));

  iree_vm_function_t decode_function = {};
  IREE_RETURN_IF_ERROR(
      runner.lookup_function("qwen_decode_step", &decode_function));

  std::vector<BufferViewPtr> first_outputs;
  IREE_RETURN_IF_ERROR(qwen_decode_step(
      &runner, decode_function, qwen_decode_stub_step("2xf32=6 8"),
      /*key_cache=*/nullptr, /*value_cache=*/nullptr, &first_outputs));

  const std::array<float, 4> first_expected = {30.0f, 72.0f, 56.0f, 110.0f};
  if (!qwen_decode_expect_hidden(
          first_outputs[static_cast<size_t>(QwenDecodeOutputIndex::kHidden)]
              .get(),
          first_expected, "first qwen decode result")) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "first qwen decode result did not match");
  }

  std::vector<BufferViewPtr> second_outputs;
  IREE_RETURN_IF_ERROR(qwen_decode_step(
      &runner, decode_function, qwen_decode_stub_step("2xf32=8 10"),
      first_outputs[static_cast<size_t>(
                        QwenDecodeOutputIndex::kKeyCacheTransposed)]
          .get(),
      first_outputs[static_cast<size_t>(QwenDecodeOutputIndex::kValueCache)]
          .get(),
      &second_outputs));

  const std::array<float, 4> second_expected = {
      340.0f / 9.0f,
      754.0f / 9.0f,
      598.0f / 9.0f,
      1120.0f / 9.0f,
  };
  if (!qwen_decode_expect_hidden(
          second_outputs[static_cast<size_t>(QwenDecodeOutputIndex::kHidden)]
              .get(),
          second_expected, "second qwen decode result")) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "second qwen decode result did not match");
  }

  return iree_ok_status();
}

} // namespace

int main(int argc, char **argv) {
  iree_flags_set_usage("lrrt_iree_qwen_decode_e2e_smoke",
                       "Runs a two-token Qwen-like decode step through lrrt's "
                       "IREE HAL adapter.");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <qwen_decode_step.vmfb>\n", argv[0]);
    return 2;
  }

  iree_status_t status = run_smoke(argv[1]);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }

  std::puts("iree_qwen_decode_e2e: ok");
  return 0;
}
