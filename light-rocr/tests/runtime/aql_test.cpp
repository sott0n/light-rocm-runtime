#include "light_rocr/runtime/aql.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

using light_rocr::runtime::AqlKernelDispatchPacket;
using light_rocr::runtime::AqlPacketError;

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

AqlKernelDispatchPacket valid_packet() {
  AqlKernelDispatchPacket packet;
  packet.header = light_rocr::runtime::kAqlKernelDispatchHeader;
  packet.setup = 1;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;
  packet.private_segment_size = 32;
  packet.group_segment_size = 128;
  packet.kernel_object = 0x100000;
  packet.kernarg_address = 0x200000;
  packet.completion_signal = 0x300000;
  return packet;
}

void abi_layout(TestContext *context) {
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

void accepts_valid_packet(TestContext *context) {
  const AqlKernelDispatchPacket packet = valid_packet();
  const auto status =
      light_rocr::runtime::validate_kernel_dispatch_packet(packet);
  context->expect(static_cast<bool>(status), status.message);
  context->expect(packet.header ==
                          light_rocr::runtime::kAqlKernelDispatchHeader &&
                      packet.setup == 1,
                  "packet header or setup is incorrect");
  context->expect(packet.workgroup_size_x == 64 && packet.grid_size_x == 256 &&
                      packet.private_segment_size == 32 &&
                      packet.group_segment_size == 128,
                  "packet dispatch geometry or segment sizes are incorrect");
  context->expect(packet.kernel_object == 0x100000 &&
                      packet.kernarg_address == 0x200000 &&
                      packet.completion_signal == 0x300000,
                  "packet handles are incorrect");
}

void rejects_invalid_geometry(TestContext *context) {
  AqlKernelDispatchPacket packet = valid_packet();
  packet.setup = 0;
  auto status = light_rocr::runtime::validate_kernel_dispatch_packet(packet);
  context->expect(status.error == AqlPacketError::InvalidDimensions,
                  "zero dimensions were accepted");

  packet = valid_packet();
  packet.workgroup_size_x = 0;
  status = light_rocr::runtime::validate_kernel_dispatch_packet(packet);
  context->expect(status.error == AqlPacketError::InvalidWorkgroupSize,
                  "zero workgroup size was accepted");

  packet = valid_packet();
  packet.grid_size_x = 32;
  status = light_rocr::runtime::validate_kernel_dispatch_packet(packet);
  context->expect(status.error == AqlPacketError::InvalidGridSize,
                  "grid smaller than a workgroup was accepted");

  packet = valid_packet();
  packet.workgroup_size_y = 2;
  status = light_rocr::runtime::validate_kernel_dispatch_packet(packet);
  context->expect(status.error == AqlPacketError::InvalidWorkgroupSize,
                  "unused workgroup dimension was accepted");
}

void enforces_gfx1101_workgroup_limits(TestContext *context) {
  AqlKernelDispatchPacket packet = valid_packet();
  packet.workgroup_size_x = static_cast<uint16_t>(
      light_rocr::runtime::kGfx1101WorkgroupMaximumDimension + 1);
  packet.grid_size_x = packet.workgroup_size_x;
  auto status = light_rocr::runtime::validate_kernel_dispatch_packet(packet);
  context->expect(status.error == AqlPacketError::InvalidWorkgroupSize,
                  "oversized gfx1101 workgroup dimension was accepted");

  packet = valid_packet();
  packet.setup = 2;
  packet.workgroup_size_x =
      light_rocr::runtime::kGfx1101WorkgroupMaximumDimension;
  packet.workgroup_size_y = 2;
  packet.grid_size_x = packet.workgroup_size_x;
  packet.grid_size_y = packet.workgroup_size_y;
  status = light_rocr::runtime::validate_kernel_dispatch_packet(packet);
  context->expect(status.error == AqlPacketError::InvalidWorkgroupSize,
                  "oversized total gfx1101 workgroup was accepted");

  packet = valid_packet();
  packet.workgroup_size_x =
      light_rocr::runtime::kGfx1101WorkgroupMaximumDimension;
  packet.grid_size_x = packet.workgroup_size_x;
  status = light_rocr::runtime::validate_kernel_dispatch_packet(packet);
  context->expect(static_cast<bool>(status),
                  "maximum valid gfx1101 workgroup was rejected");
}

void enforces_gfx1101_group_segment_limit(TestContext *context) {
  AqlKernelDispatchPacket packet = valid_packet();
  packet.group_segment_size =
      light_rocr::runtime::kGfx1101GroupSegmentMaximumSize;
  auto status = light_rocr::runtime::validate_kernel_dispatch_packet(packet);
  context->expect(static_cast<bool>(status),
                  "maximum valid gfx1101 group segment was rejected");

  packet.group_segment_size =
      light_rocr::runtime::kGfx1101GroupSegmentMaximumSize + 1U;
  status = light_rocr::runtime::validate_kernel_dispatch_packet(packet);
  context->expect(status.error == AqlPacketError::InvalidGroupSegmentSize,
                  "oversized gfx1101 group segment was accepted");
}

void rejects_misaligned_handles(TestContext *context) {
  AqlKernelDispatchPacket packet = valid_packet();
  packet.kernel_object += 1;
  auto status = light_rocr::runtime::validate_kernel_dispatch_packet(packet);
  context->expect(status.error == AqlPacketError::InvalidKernelObject,
                  "misaligned kernel descriptor was accepted");

  packet = valid_packet();
  packet.kernarg_address += 1;
  status = light_rocr::runtime::validate_kernel_dispatch_packet(packet);
  context->expect(status.error == AqlPacketError::InvalidKernargAddress,
                  "misaligned kernarg was accepted");

  packet = valid_packet();
  packet.completion_signal += 1;
  status = light_rocr::runtime::validate_kernel_dispatch_packet(packet);
  context->expect(status.error == AqlPacketError::InvalidCompletionSignal,
                  "misaligned completion signal was accepted");
}

void permits_optional_zero_handles(TestContext *context) {
  AqlKernelDispatchPacket packet = valid_packet();
  packet.kernarg_address = 0;
  packet.completion_signal = 0;
  const auto status =
      light_rocr::runtime::validate_kernel_dispatch_packet(packet);
  context->expect(static_cast<bool>(status), status.message);
}

void validates_packet_before_publication(TestContext *context) {
  AqlKernelDispatchPacket packet = valid_packet();
  packet.header = light_rocr::runtime::kAqlPacketTypeInvalid;
  auto status = light_rocr::runtime::validate_kernel_dispatch_packet(packet);
  context->expect(status.error == AqlPacketError::InvalidHeader,
                  "invalid packet header was accepted");

  packet = valid_packet();
  packet.reserved2 = 1;
  status = light_rocr::runtime::validate_kernel_dispatch_packet(packet);
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
      {"accepts_valid_packet", accepts_valid_packet},
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
