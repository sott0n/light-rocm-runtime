#include "light_rocr/arch/gfx11/scratch.hpp"

#include <functional>
#include <iostream>
#include <limits>
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

light_rocr::runtime::Node gfx1101_node() {
  light_rocr::runtime::Node node;
  node.node_id = 1;
  node.gpu_id = 43288;
  node.simd_count = 120;
  node.simd_per_compute_unit = 2;
  node.wavefront_size = 32;
  node.maximum_waves_per_simd = 16;
  node.shader_engine_count = 3;
  node.maximum_scratch_waves_per_compute_unit = 32;
  node.xcc_count = 1;
  node.architecture = {11, 0, 1};
  node.memory_banks.push_back({light_rocr::runtime::MemoryHeapType::Scratch,
                               uint64_t{4} * 1024 * 1024 * 1024,
                               0x2000000000000ULL});
  return node;
}

void computes_observed_vector_add_backing(TestContext *context) {
  const auto result =
      light_rocr::arch::gfx11::queue_scratch_requirements(gfx1101_node(), 272);
  context->expect(static_cast<bool>(result), result.status.message);
  context->expect(result.requirements.private_segment_size == 272 &&
                      result.requirements.allocation_size == 33423360 &&
                      result.requirements.backing_size_per_xcc == 33423360 &&
                      result.requirements.wave_size_units == 68 &&
                      result.requirements.waves_per_shader_engine == 640,
                  "unexpected gfx1101 scratch requirements");
}

void builds_observed_vector_add_registers(TestContext *context) {
  constexpr uint64_t kBackingAddress = 0x00007abc12000000ULL;
  const auto result = light_rocr::arch::gfx11::make_queue_scratch_control(
      gfx1101_node(), 272, kBackingAddress);
  context->expect(static_cast<bool>(result), result.status.message);
  context->expect(result.control.resource_descriptor[0] == 0x12000000 &&
                      result.control.resource_descriptor[1] == 0x40007abc &&
                      result.control.resource_descriptor[2] == 33423360 &&
                      result.control.resource_descriptor[3] == 0x20814fac,
                  "unexpected gfx1101 scratch SRD");
  context->expect(result.control.compute_tmpring_size == 0x44280,
                  "unexpected gfx1101 COMPUTE_TMPRING_SIZE");
  context->expect(result.control.backing_memory_location == kBackingAddress &&
                      result.control.backing_memory_byte_size == 33423360 &&
                      result.control.wave64_lane_byte_size == 272,
                  "unexpected AMD queue scratch fields");
}

void supports_canonical_scratch_free_queue(TestContext *context) {
  const auto result =
      light_rocr::arch::gfx11::make_queue_scratch_control(gfx1101_node(), 0, 0);
  context->expect(static_cast<bool>(result), result.status.message);
  context->expect(result.requirements.allocation_size == 0 &&
                      result.control.resource_descriptor[0] == 0 &&
                      result.control.resource_descriptor[1] == 0x40000000 &&
                      result.control.resource_descriptor[2] == 0 &&
                      result.control.resource_descriptor[3] == 0x20814fac &&
                      result.control.compute_tmpring_size == 0,
                  "scratch-free control state is not canonical");
}

void rejects_invalid_topology(TestContext *context) {
  auto node = gfx1101_node();
  node.architecture.stepping = 2;
  auto result = light_rocr::arch::gfx11::queue_scratch_requirements(node, 272);
  context->expect(
      result.status.error ==
          light_rocr::arch::gfx11::QueueScratchError::UnsupportedArchitecture,
      "unsupported architecture was accepted");

  node = gfx1101_node();
  node.xcc_count = 2;
  result = light_rocr::arch::gfx11::queue_scratch_requirements(node, 272);
  context->expect(
      result.status.error ==
          light_rocr::arch::gfx11::QueueScratchError::InvalidTopology,
      "multi-XCC topology was accepted");

  node = gfx1101_node();
  node.memory_banks.clear();
  result = light_rocr::arch::gfx11::queue_scratch_requirements(node, 272);
  context->expect(
      result.status.error ==
          light_rocr::arch::gfx11::QueueScratchError::MissingScratchAperture,
      "missing scratch aperture was accepted");
}

void rejects_unrepresentable_requests(TestContext *context) {
  auto node = gfx1101_node();
  auto result = light_rocr::arch::gfx11::queue_scratch_requirements(
      node, std::numeric_limits<uint32_t>::max());
  context->expect(
      result.status.error ==
          light_rocr::arch::gfx11::QueueScratchError::PrivateSegmentTooLarge,
      "oversized private segment was accepted");

  node.memory_banks.front().size = 4096;
  result = light_rocr::arch::gfx11::queue_scratch_requirements(node, 272);
  context->expect(
      result.status.error ==
          light_rocr::arch::gfx11::QueueScratchError::ScratchApertureTooSmall,
      "undersized scratch aperture was accepted");
}

void rejects_invalid_backing_addresses(TestContext *context) {
  auto result = light_rocr::arch::gfx11::make_queue_scratch_control(
      gfx1101_node(), 272, 0x12345000);
  context->expect(
      result.status.error ==
          light_rocr::arch::gfx11::QueueScratchError::InvalidBackingAddress,
      "misaligned backing was accepted");

  result = light_rocr::arch::gfx11::make_queue_scratch_control(
      gfx1101_node(), 272, uint64_t{1} << 48U);
  context->expect(
      result.status.error ==
          light_rocr::arch::gfx11::QueueScratchError::InvalidBackingAddress,
      "non-48-bit backing was accepted");

  result = light_rocr::arch::gfx11::make_queue_scratch_control(gfx1101_node(),
                                                               0, 0x10000);
  context->expect(
      result.status.error ==
          light_rocr::arch::gfx11::QueueScratchError::InvalidBackingAddress,
      "scratch-free queue accepted a backing address");
}

void enum_names(TestContext *context) {
  context->expect(
      std::string(light_rocr::arch::gfx11::queue_scratch_error_name(
          light_rocr::arch::gfx11::QueueScratchError::RegisterOverflow)) ==
          "register_overflow",
      "unexpected scratch error name");
}

} // namespace

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"computes_observed_vector_add_backing",
       computes_observed_vector_add_backing},
      {"builds_observed_vector_add_registers",
       builds_observed_vector_add_registers},
      {"supports_canonical_scratch_free_queue",
       supports_canonical_scratch_free_queue},
      {"rejects_invalid_topology", rejects_invalid_topology},
      {"rejects_unrepresentable_requests", rejects_unrepresentable_requests},
      {"rejects_invalid_backing_addresses", rejects_invalid_backing_addresses},
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
