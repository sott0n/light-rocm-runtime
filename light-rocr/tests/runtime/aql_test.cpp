#include "light_rocr/runtime/aql.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using light_rocr::runtime::AqlPacketError;
using light_rocr::runtime::KernelDispatchSpec;

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

KernelDispatchSpec valid_spec() {
  KernelDispatchSpec spec;
  spec.dimensions = 1;
  spec.workgroup_size_x = 64;
  spec.grid_size_x = 256;
  spec.private_segment_size = 32;
  spec.group_segment_size = 128;
  spec.kernel_object = 0x100000;
  spec.kernarg_address = 0x200000;
  spec.completion_signal = 0x300000;
  return spec;
}

void abi_layout(TestContext *context) {
  using light_rocr::runtime::AqlKernelDispatchPacket;
  context->expect(sizeof(AqlKernelDispatchPacket) == 64,
                  "AQL packet size is not 64 bytes");
  context->expect(alignof(AqlKernelDispatchPacket) == 64,
                  "AQL packet alignment is not 64 bytes");
  context->expect(
      offsetof(AqlKernelDispatchPacket, kernel_object) == 32 &&
          offsetof(AqlKernelDispatchPacket, kernarg_address) == 40 &&
          offsetof(AqlKernelDispatchPacket, completion_signal) == 56,
      "AQL packet pointer fields have incorrect offsets");
}

void builds_valid_packet(TestContext *context) {
  const KernelDispatchSpec spec = valid_spec();
  const auto result = light_rocr::runtime::make_kernel_dispatch_packet(spec);
  context->expect(static_cast<bool>(result), result.status.message);
  context->expect(result.packet.header ==
                          light_rocr::runtime::kAqlKernelDispatchHeader &&
                      result.packet.setup == 1,
                  "packet header or setup is incorrect");
  context->expect(result.packet.workgroup_size_x == 64 &&
                      result.packet.grid_size_x == 256 &&
                      result.packet.private_segment_size == 32 &&
                      result.packet.group_segment_size == 128,
                  "packet dispatch geometry or segment sizes are incorrect");
  context->expect(result.packet.kernel_object == spec.kernel_object &&
                      result.packet.kernarg_address == spec.kernarg_address &&
                      result.packet.completion_signal == spec.completion_signal,
                  "packet handles are incorrect");
}

void rejects_invalid_geometry(TestContext *context) {
  KernelDispatchSpec spec = valid_spec();
  spec.dimensions = 0;
  auto result = light_rocr::runtime::make_kernel_dispatch_packet(spec);
  context->expect(result.status.error == AqlPacketError::InvalidDimensions,
                  "zero dimensions were accepted");

  spec = valid_spec();
  spec.workgroup_size_x = 0;
  result = light_rocr::runtime::make_kernel_dispatch_packet(spec);
  context->expect(result.status.error == AqlPacketError::InvalidWorkgroupSize,
                  "zero workgroup size was accepted");

  spec = valid_spec();
  spec.grid_size_x = 32;
  result = light_rocr::runtime::make_kernel_dispatch_packet(spec);
  context->expect(result.status.error == AqlPacketError::InvalidGridSize,
                  "grid smaller than a workgroup was accepted");

  spec = valid_spec();
  spec.workgroup_size_y = 2;
  result = light_rocr::runtime::make_kernel_dispatch_packet(spec);
  context->expect(result.status.error == AqlPacketError::InvalidWorkgroupSize,
                  "unused workgroup dimension was accepted");
}

void enforces_gfx1101_workgroup_limits(TestContext *context) {
  KernelDispatchSpec spec = valid_spec();
  spec.workgroup_size_x = static_cast<uint16_t>(
      light_rocr::runtime::kGfx1101WorkgroupMaximumDimension + 1);
  spec.grid_size_x = spec.workgroup_size_x;
  auto result = light_rocr::runtime::make_kernel_dispatch_packet(spec);
  context->expect(result.status.error == AqlPacketError::InvalidWorkgroupSize,
                  "oversized gfx1101 workgroup dimension was accepted");

  spec = valid_spec();
  spec.dimensions = 2;
  spec.workgroup_size_x =
      light_rocr::runtime::kGfx1101WorkgroupMaximumDimension;
  spec.workgroup_size_y = 2;
  spec.grid_size_x = spec.workgroup_size_x;
  spec.grid_size_y = spec.workgroup_size_y;
  result = light_rocr::runtime::make_kernel_dispatch_packet(spec);
  context->expect(result.status.error == AqlPacketError::InvalidWorkgroupSize,
                  "oversized total gfx1101 workgroup was accepted");

  spec = valid_spec();
  spec.workgroup_size_x =
      light_rocr::runtime::kGfx1101WorkgroupMaximumDimension;
  spec.grid_size_x = spec.workgroup_size_x;
  result = light_rocr::runtime::make_kernel_dispatch_packet(spec);
  context->expect(static_cast<bool>(result),
                  "maximum valid gfx1101 workgroup was rejected");
}

void enforces_gfx1101_group_segment_limit(TestContext *context) {
  KernelDispatchSpec spec = valid_spec();
  spec.group_segment_size =
      light_rocr::runtime::kGfx1101GroupSegmentMaximumSize;
  auto result = light_rocr::runtime::make_kernel_dispatch_packet(spec);
  context->expect(static_cast<bool>(result),
                  "maximum valid gfx1101 group segment was rejected");

  spec.group_segment_size =
      light_rocr::runtime::kGfx1101GroupSegmentMaximumSize + 1U;
  result = light_rocr::runtime::make_kernel_dispatch_packet(spec);
  context->expect(result.status.error ==
                      AqlPacketError::InvalidGroupSegmentSize,
                  "oversized gfx1101 group segment was accepted");
}

void rejects_misaligned_handles(TestContext *context) {
  KernelDispatchSpec spec = valid_spec();
  spec.kernel_object += 1;
  auto result = light_rocr::runtime::make_kernel_dispatch_packet(spec);
  context->expect(result.status.error == AqlPacketError::InvalidKernelObject,
                  "misaligned kernel descriptor was accepted");

  spec = valid_spec();
  spec.kernarg_address += 1;
  result = light_rocr::runtime::make_kernel_dispatch_packet(spec);
  context->expect(result.status.error == AqlPacketError::InvalidKernargAddress,
                  "misaligned kernarg was accepted");

  spec = valid_spec();
  spec.completion_signal += 1;
  result = light_rocr::runtime::make_kernel_dispatch_packet(spec);
  context->expect(result.status.error ==
                      AqlPacketError::InvalidCompletionSignal,
                  "misaligned completion signal was accepted");
}

void permits_optional_zero_handles(TestContext *context) {
  KernelDispatchSpec spec = valid_spec();
  spec.kernarg_address = 0;
  spec.completion_signal = 0;
  const auto result = light_rocr::runtime::make_kernel_dispatch_packet(spec);
  context->expect(static_cast<bool>(result), result.status.message);
}

void validates_packet_before_publication(TestContext *context) {
  auto result = light_rocr::runtime::make_kernel_dispatch_packet(valid_spec());
  result.packet.header = light_rocr::runtime::kAqlPacketTypeInvalid;
  auto status =
      light_rocr::runtime::validate_kernel_dispatch_packet(result.packet);
  context->expect(status.error == AqlPacketError::InvalidHeader,
                  "invalid packet header was accepted");

  result = light_rocr::runtime::make_kernel_dispatch_packet(valid_spec());
  result.packet.reserved2 = 1;
  status = light_rocr::runtime::validate_kernel_dispatch_packet(result.packet);
  context->expect(status.error == AqlPacketError::NonzeroReservedField,
                  "non-zero reserved field was accepted");
}

void enum_names(TestContext *context) {
  context->expect(std::string(light_rocr::runtime::aql_packet_error_name(
                      AqlPacketError::InvalidCompletionSignal)) ==
                      "invalid_completion_signal",
                  "unexpected AQL packet error name");
}

} // namespace

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"abi_layout", abi_layout},
      {"builds_valid_packet", builds_valid_packet},
      {"rejects_invalid_geometry", rejects_invalid_geometry},
      {"enforces_gfx1101_workgroup_limits", enforces_gfx1101_workgroup_limits},
      {"enforces_gfx1101_group_segment_limit",
       enforces_gfx1101_group_segment_limit},
      {"rejects_misaligned_handles", rejects_misaligned_handles},
      {"permits_optional_zero_handles", permits_optional_zero_handles},
      {"validates_packet_before_publication",
       validates_packet_before_publication},
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
