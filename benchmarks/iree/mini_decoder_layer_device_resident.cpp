#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
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

struct Colors {
  const char *title = "";
  const char *label = "";
  const char *time = "";
  const char *speedup = "";
  const char *reset = "";
};

Colors output_colors() {
  const char *term = std::getenv("TERM");
  if (!term || std::string(term) == "dumb") {
    return {};
  }
  return {
      "\033[1;32m", "\033[32m", "\033[1;33m", "\033[1;36m", "\033[0m",
  };
}

uint32_t parse_iterations(const char *text) {
  char *end = nullptr;
  unsigned long value = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0' || value == 0 ||
      value > static_cast<unsigned long>(UINT32_MAX)) {
    throw std::invalid_argument("benchmark count must be a positive uint32");
  }
  return static_cast<uint32_t>(value);
}

std::string tensor_2x3(const float values[6]) {
  char text[160] = {};
  std::snprintf(text, sizeof(text), "2x3xf32=%.9g %.9g %.9g %.9g %.9g %.9g",
                values[0], values[1], values[2], values[3], values[4],
                values[5]);
  return text;
}

std::string tensor_3x2(const float values[6]) {
  char text[160] = {};
  std::snprintf(text, sizeof(text), "3x2xf32=%.9g %.9g %.9g %.9g %.9g %.9g",
                values[0], values[1], values[2], values[3], values[4],
                values[5]);
  return text;
}

void read_cache(iree_hal_buffer_view_t *view, float *values,
                iree_host_size_t byte_length, const char *label) {
  iree_hal_buffer_t *buffer = iree_hal_buffer_view_buffer(view);
  iree_status_t status =
      iree_hal_buffer_map_read(buffer, 0, values, byte_length);
  if (!iree_status_is_ok(status)) {
    std::fprintf(stderr, "%s failed\n", label);
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    throw std::runtime_error("failed to read benchmark cache output");
  }
}

bool expect_matrix(iree_hal_buffer_view_t *view, const float (&expected)[4]) {
  float values[4] = {};
  read_cache(view, values, sizeof(values), "iree_hal_buffer_map_read(result)");
  for (int i = 0; i < 4; ++i) {
    if (std::fabs(values[i] - expected[i]) > 1e-3f) {
      std::fprintf(stderr, "result[%d]: expected %.6g, got %.6g\n", i,
                   expected[i], values[i]);
      return false;
    }
  }
  return true;
}

std::vector<std::string> base_specs(const std::string &key_cache,
                                    const std::string &value_cache,
                                    const char *new_value_spec) {
  return {
      "2x2xf32=1 2 3 4", "2x2xf32=1 2 3 4", key_cache,         "2xf32=0 0",
      value_cache,       new_value_spec,    "2xf32=1 0",       "2xf32=0 1",
      "2x2xf32=1 0 0 1", "2x2xf32=1 0 0 1", "2x2xf32=1 0 0 1",
  };
}

iree_status_t invoke_host_step(VmfbRunner *runner,
                               const iree_vm_function_t &function,
                               const std::string &key_cache,
                               const std::string &value_cache,
                               const char *new_value_spec,
                               std::vector<BufferViewPtr> *outputs) {
  return runner->invoke(function,
                        base_specs(key_cache, value_cache, new_value_spec), {},
                        3, outputs);
}

iree_status_t invoke_device_step(VmfbRunner *runner,
                                 const iree_vm_function_t &function,
                                 iree_hal_buffer_view_t *key_cache,
                                 iree_hal_buffer_view_t *value_cache,
                                 const char *new_value_spec,
                                 std::vector<BufferViewPtr> *outputs) {
  std::vector<BufferViewReplacement> replacements;
  if (key_cache) {
    replacements.push_back({2, key_cache});
  }
  if (value_cache) {
    replacements.push_back({4, value_cache});
  }
  return runner->invoke(
      function,
      base_specs("2x3xf32=0 0 0 0 0 0", "3x2xf32=2 4 4 6 0 0", new_value_spec),
      replacements, 3, outputs);
}

void check_status(iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    throw std::runtime_error("IREE benchmark invocation failed");
  }
}

void run_host_handoff(VmfbRunner *runner, const iree_vm_function_t &function) {
  std::vector<BufferViewPtr> first_outputs;
  check_status(invoke_host_step(runner, function, "2x3xf32=0 0 0 0 0 0",
                                "3x2xf32=2 4 4 6 0 0", "2xf32=6 8",
                                &first_outputs));

  float key_cache[6] = {};
  float value_cache[6] = {};
  read_cache(first_outputs[0].get(), key_cache, sizeof(key_cache),
             "iree_hal_buffer_map_read(key_cache)");
  read_cache(first_outputs[1].get(), value_cache, sizeof(value_cache),
             "iree_hal_buffer_map_read(value_cache)");

  std::vector<BufferViewPtr> second_outputs;
  check_status(invoke_host_step(runner, function, tensor_2x3(key_cache),
                                tensor_3x2(value_cache), "2xf32=8 10",
                                &second_outputs));
  const float expected[4] = {
      340.0f / 9.0f,
      754.0f / 9.0f,
      598.0f / 9.0f,
      1120.0f / 9.0f,
  };
  if (!expect_matrix(second_outputs[2].get(), expected)) {
    throw std::runtime_error("host handoff benchmark result mismatch");
  }
}

void run_device_handoff(VmfbRunner *runner,
                        const iree_vm_function_t &function) {
  std::vector<BufferViewPtr> first_outputs;
  check_status(invoke_device_step(runner, function, /*key_cache=*/nullptr,
                                  /*value_cache=*/nullptr, "2xf32=6 8",
                                  &first_outputs));

  std::vector<BufferViewPtr> second_outputs;
  check_status(invoke_device_step(runner, function, first_outputs[0].get(),
                                  first_outputs[1].get(), "2xf32=8 10",
                                  &second_outputs));
  const float expected[4] = {
      340.0f / 9.0f,
      754.0f / 9.0f,
      598.0f / 9.0f,
      1120.0f / 9.0f,
  };
  if (!expect_matrix(second_outputs[2].get(), expected)) {
    throw std::runtime_error("device handoff benchmark result mismatch");
  }
}

template <typename Fn>
double measure_ns(uint32_t iterations, uint32_t warmup_iterations, Fn fn) {
  for (uint32_t i = 0; i < warmup_iterations; ++i) {
    fn();
  }

  auto begin = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < iterations; ++i) {
    fn();
  }
  auto end = std::chrono::steady_clock::now();
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                 .count()) /
         static_cast<double>(iterations);
}

void print_results(uint32_t iterations, uint32_t warmup_iterations,
                   double host_ns, double device_ns) {
  const Colors colors = output_colors();
  const double speedup = host_ns / device_ns;
  std::printf("\n%sLRRT IREE Device-Resident Benchmark%s\n", colors.title,
              colors.reset);
  std::printf("%s====================================%s\n", colors.title,
              colors.reset);
  std::printf("%sVMFB%s mini_decoder_layer_rope_kv_cache_outputs\n",
              colors.label, colors.reset);
  std::printf("%sIterations%s measured=%u warmup=%u\n\n", colors.label,
              colors.reset, iterations, warmup_iterations);

  std::printf("%s%-26s %14s  %s%s\n", colors.label, "Path", "CPU round us",
              "Measurement", colors.reset);
  std::printf("%-26s %s%11.3f us%s  %s\n", "Host handoff", colors.time,
              host_ns / 1.0e3, colors.reset, "readback + parse + upload");
  std::printf("%-26s %s%11.3f us%s  %s\n", "Device-resident handoff",
              colors.time, device_ns / 1.0e3, colors.reset,
              "buffer_view replacement");
  std::printf("\n%sDevice-resident speedup: %.2fx%s\n\n", colors.speedup,
              speedup, colors.reset);
}

iree_status_t run_benchmark(const char *module_path, uint32_t iterations,
                            uint32_t warmup_iterations) {
  VmfbRunner runner;
  IREE_RETURN_IF_ERROR(runner.initialize(module_path));

  iree_vm_function_t function;
  IREE_RETURN_IF_ERROR(runner.lookup_function(
      "mini_decoder_layer_rope_kv_cache_outputs", &function));

  const double host_ns = measure_ns(iterations, warmup_iterations, [&]() {
    run_host_handoff(&runner, function);
  });
  const double device_ns = measure_ns(iterations, warmup_iterations, [&]() {
    run_device_handoff(&runner, function);
  });
  print_results(iterations, warmup_iterations, host_ns, device_ns);
  return iree_ok_status();
}

} // namespace

int main(int argc, char **argv) {
  iree_flags_set_usage("lrrt_iree_mini_decoder_layer_benchmark",
                       "Benchmarks IREE mini decoder host handoff vs "
                       "device-resident KV handoff.");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  if (argc < 2 || argc > 4) {
    std::fprintf(stderr,
                 "usage: %s <mini_decoder_layer_rope_kv_cache_outputs.vmfb> "
                 "[iterations] [warmup]\n",
                 argv[0]);
    return 2;
  }

  try {
    const uint32_t iterations = argc >= 3 ? parse_iterations(argv[2]) : 100;
    const uint32_t warmup_iterations =
        argc >= 4 ? parse_iterations(argv[3]) : 10;
    iree_status_t status =
        run_benchmark(argv[1], iterations, warmup_iterations);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      return 1;
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "benchmark failed: %s\n", error.what());
    return 1;
  }

  return 0;
}
