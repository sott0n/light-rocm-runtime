#include "light_rocr/runtime/topology.hpp"

#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct TestContext {
  int failures = 0;

  void expect(bool condition, const std::string &message) {
    if (!condition) {
      ++failures;
      std::cerr << "  FAIL: " << message << '\n';
    }
  }
};

using TestFunction = std::function<void(TestContext *)>;

light_rocr::runtime::Node gpu_node(uint32_t node_id, uint32_t stepping = 1) {
  light_rocr::runtime::Node node;
  node.node_id = node_id;
  node.gpu_id = 100 + node_id;
  node.simd_count = 120;
  node.simd_per_compute_unit = 2;
  node.architecture = {11, 0, stepping};
  return node;
}

void node_classification_and_compute_units(TestContext *context) {
  light_rocr::runtime::Node cpu;
  cpu.cpu_core_count = 64;
  context->expect(!cpu.is_gpu(), "CPU-only node was classified as a GPU");
  context->expect(cpu.compute_unit_count() == 0,
                  "CPU-only node has compute units");

  const light_rocr::runtime::Node gpu = gpu_node(1);
  context->expect(gpu.is_gpu(), "GPU node was not classified as a GPU");
  context->expect(gpu.compute_unit_count() == 60,
                  "unexpected compute-unit count");

  light_rocr::runtime::Node malformed = gpu;
  malformed.simd_per_compute_unit = 0;
  context->expect(malformed.compute_unit_count() == 0,
                  "zero SIMD-per-CU must not divide");
  malformed.simd_per_compute_unit = 7;
  context->expect(malformed.compute_unit_count() == 0,
                  "non-integral compute-unit count was truncated");
}

void target_names(TestContext *context) {
  context->expect(light_rocr::runtime::gfx_target_name({11, 0, 1}) == "gfx1101",
                  "gfx1101 architecture name mismatch");
  context->expect(light_rocr::runtime::gfx_target_name({9, 0, 10}) == "gfx90a",
                  "hexadecimal stepping name mismatch");
  context->expect(light_rocr::runtime::gfx_target_name({}) == "",
                  "unknown architecture should have no target name");
}

void target_canonicalization(TestContext *context) {
  context->expect(light_rocr::runtime::canonical_gfx_target("gfx1101") ==
                      "gfx1101",
                  "short target was not preserved");
  context->expect(light_rocr::runtime::canonical_gfx_target(
                      "amdgcn-amd-amdhsa--gfx1101") == "gfx1101",
                  "canonical target was not reduced");
  context->expect(light_rocr::runtime::canonical_gfx_target(
                      "amdgcn-amd-amdhsa--gfx1101:sramecc-:xnack-") ==
                      "gfx1101",
                  "target features were not removed");
  for (const std::string invalid : {"", "gfx", "gfx11G1", "cpu1101"}) {
    context->expect(light_rocr::runtime::canonical_gfx_target(invalid).empty(),
                    "invalid target was accepted: " + invalid);
  }
}

void select_one_matching_gpu(TestContext *context) {
  light_rocr::runtime::Topology topology;
  light_rocr::runtime::Node cpu;
  cpu.node_id = 0;
  cpu.cpu_core_count = 64;
  topology.nodes = {cpu, gpu_node(1), gpu_node(2, 2)};

  const auto result = light_rocr::runtime::select_unique_gpu(
      topology, "amdgcn-amd-amdhsa--gfx1101");
  context->expect(static_cast<bool>(result), result.message);
  context->expect(result.node_index == 1, "wrong GPU node selected");
}

void reject_invalid_target(TestContext *context) {
  light_rocr::runtime::Topology topology;
  topology.nodes = {gpu_node(1)};
  const auto result =
      light_rocr::runtime::select_unique_gpu(topology, "not-a-gpu");
  context->expect(!result, "invalid target unexpectedly selected a GPU");
  context->expect(result.error ==
                      light_rocr::runtime::GpuSelectionError::InvalidTarget,
                  "wrong invalid-target error");
}

void reject_no_matching_gpu(TestContext *context) {
  light_rocr::runtime::Topology topology;
  topology.nodes = {gpu_node(1, 2)};
  const auto result =
      light_rocr::runtime::select_unique_gpu(topology, "gfx1101");
  context->expect(!result, "missing target unexpectedly selected a GPU");
  context->expect(result.error ==
                      light_rocr::runtime::GpuSelectionError::NoMatchingGpu,
                  "wrong no-match error");
}

void reject_ambiguous_gpu(TestContext *context) {
  light_rocr::runtime::Topology topology;
  topology.nodes = {gpu_node(1), gpu_node(2)};
  const auto result =
      light_rocr::runtime::select_unique_gpu(topology, "gfx1101");
  context->expect(!result, "ambiguous target unexpectedly selected a GPU");
  context->expect(
      result.error ==
          light_rocr::runtime::GpuSelectionError::MultipleMatchingGpus,
      "wrong ambiguous-target error");
}

void enum_names(TestContext *context) {
  context->expect(std::string(light_rocr::runtime::memory_heap_type_name(
                      light_rocr::runtime::MemoryHeapType::DeviceSvm)) ==
                      "device_svm",
                  "memory heap name mismatch");
  context->expect(
      std::string(light_rocr::runtime::gpu_selection_error_name(
          light_rocr::runtime::GpuSelectionError::MultipleMatchingGpus)) ==
          "multiple_matching_gpus",
      "selection error name mismatch");
}

} // namespace

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"node_classification_and_compute_units",
       node_classification_and_compute_units},
      {"target_names", target_names},
      {"target_canonicalization", target_canonicalization},
      {"select_one_matching_gpu", select_one_matching_gpu},
      {"reject_invalid_target", reject_invalid_target},
      {"reject_no_matching_gpu", reject_no_matching_gpu},
      {"reject_ambiguous_gpu", reject_ambiguous_gpu},
      {"enum_names", enum_names},
  };

  TestContext context;
  for (const auto &test : tests) {
    std::cout << "[ RUN      ] " << test.first << '\n';
    const int failures_before = context.failures;
    test.second(&context);
    if (context.failures == failures_before) {
      std::cout << "[       OK ] " << test.first << '\n';
    } else {
      std::cout << "[  FAILED  ] " << test.first << '\n';
    }
  }

  if (context.failures != 0) {
    std::cerr << context.failures << " assertion(s) failed\n";
    return 1;
  }
  std::cout << tests.size() << " test(s) passed\n";
  return 0;
}
