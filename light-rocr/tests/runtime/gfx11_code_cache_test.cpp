#include "light_rocr/arch/gfx11/code_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

using light_rocr::arch::gfx11::build_code_cache_invalidate;
using light_rocr::arch::gfx11::CodeCacheCommandError;

struct TestContext {
  int failures = 0;

  void expect(bool condition, const std::string &message) {
    if (!condition) {
      ++failures;
      std::cerr << "  FAIL: " << message << '\n';
    }
  }
};

void packet_layout(TestContext *context) {
  using light_rocr::arch::gfx11::AqlPm4IndirectBufferPacket;
  context->expect(sizeof(AqlPm4IndirectBufferPacket) == 64 &&
                      alignof(AqlPm4IndirectBufferPacket) == 64,
                  "PM4 indirect-buffer packet has the wrong ABI size");
  context->expect(
      offsetof(AqlPm4IndirectBufferPacket, header) == 0 &&
          offsetof(AqlPm4IndirectBufferPacket, vendor_header) == 2 &&
          offsetof(AqlPm4IndirectBufferPacket, indirect_buffer) == 4 &&
          offsetof(AqlPm4IndirectBufferPacket, dword_count_remaining) == 20 &&
          offsetof(AqlPm4IndirectBufferPacket, completion_signal) == 56,
      "PM4 indirect-buffer packet fields have incorrect offsets");
}

void gfx11_command_encoding(TestContext *context) {
  const auto result = build_code_cache_invalidate(
      0x123456780000ULL, 0x3000, 0x234567890000ULL, 0x3456789abc00ULL);
  context->expect(static_cast<bool>(result),
                  "valid cache command was rejected");
  context->expect(result.command[0] == 0xc0065800U && result.command[1] == 0 &&
                      result.command[2] == 0x30U && result.command[3] == 0 &&
                      result.command[4] == 0x34567800U &&
                      result.command[5] == 0x12U && result.command[6] == 0 &&
                      result.command[7] == 0x4381U,
                  "gfx11 ACQUIRE_MEM command encoding is incorrect");
  context->expect(result.packet.header == 0 &&
                      result.packet.vendor_header == 1 &&
                      result.packet.indirect_buffer[0] == 0xc0023f00U &&
                      result.packet.indirect_buffer[1] == 0x67890000U &&
                      result.packet.indirect_buffer[2] == 0x2345U &&
                      result.packet.indirect_buffer[3] == 0x00800008U &&
                      result.packet.dword_count_remaining == 0xaU &&
                      result.packet.completion_signal == 0x3456789abc00ULL,
                  "gfx11 PM4 indirect-buffer packet encoding is incorrect");
}

void rejects_invalid_ranges(TestContext *context) {
  context->expect(
      build_code_cache_invalidate(0x1001, 0x1000, 0x2000, 0x3000).error ==
          CodeCacheCommandError::InvalidCodeRange,
      "unaligned code address was accepted");
  context->expect(
      build_code_cache_invalidate(0x1000, 0x1001, 0x2000, 0x3000).error ==
          CodeCacheCommandError::InvalidCodeRange,
      "unaligned code size was accepted");
  context->expect(
      build_code_cache_invalidate(0x1000, 0x1000, 0x2002, 0x3000).error ==
          CodeCacheCommandError::InvalidCommandAddress,
      "unaligned command address was accepted");
  context->expect(
      build_code_cache_invalidate(0x1000, 0x1000, 0x2000, 0x3001).error ==
          CodeCacheCommandError::InvalidCompletionSignal,
      "unaligned completion signal was accepted");
}

} // namespace

int main() {
  TestContext context;
  packet_layout(&context);
  gfx11_command_encoding(&context);
  rejects_invalid_ranges(&context);
  if (context.failures != 0) {
    std::cerr << context.failures << " assertion(s) failed\n";
    return 1;
  }
  std::cout << "3 test(s) passed\n";
  return 0;
}
