#include "light_rocr/runtime/aql.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

using light_rocr::runtime::AqlKernelDispatchPacket;

struct TestContext {
  int failures = 0;

  void expect(bool condition, const std::string &message) {
    if (!condition) {
      ++failures;
      std::cerr << "  FAIL: " << message << '\n';
    }
  }
};

void abi_layout(TestContext *context) {
  context->expect(sizeof(AqlKernelDispatchPacket) == 64,
                  "AQL packet size is not 64 bytes");
  context->expect(alignof(AqlKernelDispatchPacket) == 64,
                  "AQL packet alignment is not 64 bytes");
  context->expect(
      offsetof(AqlKernelDispatchPacket, header) == 0 &&
          offsetof(AqlKernelDispatchPacket, setup) == 2 &&
          offsetof(AqlKernelDispatchPacket, workgroup_size_x) == 4 &&
          offsetof(AqlKernelDispatchPacket, grid_size_x) == 12 &&
          offsetof(AqlKernelDispatchPacket, private_segment_size) == 24 &&
          offsetof(AqlKernelDispatchPacket, kernel_object) == 32 &&
          offsetof(AqlKernelDispatchPacket, kernarg_address) == 40 &&
          offsetof(AqlKernelDispatchPacket, reserved2) == 48 &&
          offsetof(AqlKernelDispatchPacket, completion_signal) == 56,
      "AQL packet fields have incorrect offsets");
}

void abi_constants(TestContext *context) {
  context->expect(light_rocr::runtime::kAqlPacketTypeInvalid == 1 &&
                      light_rocr::runtime::kAqlPacketTypeKernelDispatch == 2,
                  "AQL packet type values are incorrect");
  context->expect(light_rocr::runtime::kAqlKernelDispatchHeader == 0x1402,
                  "kernel-dispatch header bits are incorrect");
  context->expect(light_rocr::runtime::kAqlKernelDispatchDimensionsShift == 0,
                  "kernel-dispatch dimensions shift is incorrect");
}

void default_packet_is_invalid(TestContext *context) {
  const AqlKernelDispatchPacket packet;
  context->expect(packet.header == light_rocr::runtime::kAqlPacketTypeInvalid,
                  "default packet header is not invalid");
  context->expect(packet.setup == 0 && packet.kernel_object == 0 &&
                      packet.kernarg_address == 0 &&
                      packet.completion_signal == 0,
                  "default packet contains non-zero payload fields");
}

} // namespace

int main() {
  TestContext context;
  std::cout << "[ RUN      ] abi_layout\n";
  abi_layout(&context);
  std::cout << "[ RUN      ] abi_constants\n";
  abi_constants(&context);
  std::cout << "[ RUN      ] default_packet_is_invalid\n";
  default_packet_is_invalid(&context);

  if (context.failures != 0) {
    std::cerr << context.failures << " assertion(s) failed\n";
    return 1;
  }
  std::cout << "3 test(s) passed\n";
  return 0;
}
