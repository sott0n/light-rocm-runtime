#include "light_rocr/arch/gfx11/code_cache.hpp"

#include <limits>

namespace light_rocr::arch::gfx11 {
namespace {

constexpr uint32_t kPm4Type3 = 3U << 30U;
constexpr uint32_t kPm4AcquireMemoryOpcode = 0x58U;
constexpr uint32_t kPm4IndirectBufferOpcode = 0x3fU;
constexpr uint32_t kPm4GlobalInvalidate =
    1U | (1U << 7U) | (1U << 8U) | (1U << 9U) | (1U << 14U);
constexpr uint32_t kPm4IndirectBufferValid = 1U << 23U;
constexpr uint64_t kCodeCacheGranule = 256;
constexpr uint64_t kMaximumCodeRangeGranules = (uint64_t{1} << 40U) - 1U;
constexpr uint64_t kMaximumIndirectBufferAddress = (uint64_t{1} << 48U) - 1U;

constexpr uint32_t pm4_header(uint32_t opcode, uint32_t dword_count) {
  return kPm4Type3 | (opcode << 8U) | ((dword_count - 2U) << 16U);
}

bool add_overflows(uint64_t left, uint64_t right) {
  return left > std::numeric_limits<uint64_t>::max() - right;
}

} // namespace

const char *code_cache_command_error_name(CodeCacheCommandError error) {
  switch (error) {
  case CodeCacheCommandError::None:
    return "none";
  case CodeCacheCommandError::InvalidCodeRange:
    return "invalid_code_range";
  case CodeCacheCommandError::InvalidCommandAddress:
    return "invalid_command_address";
  case CodeCacheCommandError::InvalidCompletionSignal:
    return "invalid_completion_signal";
  }
  return "unknown";
}

CodeCacheCommandResult build_code_cache_invalidate(uint64_t code_gpu_address,
                                                   uint64_t code_size,
                                                   uint64_t command_gpu_address,
                                                   uint64_t completion_signal) {
  if (code_size == 0 || code_gpu_address % kCodeCacheGranule != 0 ||
      code_size % kCodeCacheGranule != 0 ||
      code_size / kCodeCacheGranule > kMaximumCodeRangeGranules ||
      add_overflows(code_gpu_address, code_size)) {
    return {CodeCacheCommandError::InvalidCodeRange, {}, {}};
  }
  if (command_gpu_address % alignof(uint32_t) != 0 ||
      command_gpu_address > kMaximumIndirectBufferAddress ||
      add_overflows(command_gpu_address, 8U * sizeof(uint32_t)) ||
      command_gpu_address + 8U * sizeof(uint32_t) - 1U >
          kMaximumIndirectBufferAddress) {
    return {CodeCacheCommandError::InvalidCommandAddress, {}, {}};
  }
  if (completion_signal == 0 || completion_signal % 64U != 0) {
    return {CodeCacheCommandError::InvalidCompletionSignal, {}, {}};
  }

  CodeCacheCommandResult result;
  const uint64_t range_granules = code_size / kCodeCacheGranule;
  result.command[0] = pm4_header(kPm4AcquireMemoryOpcode, 8);
  result.command[1] = 0;
  result.command[2] = static_cast<uint32_t>(range_granules);
  result.command[3] = static_cast<uint32_t>(range_granules >> 32U);
  result.command[4] = static_cast<uint32_t>(code_gpu_address >> 8U);
  result.command[5] = static_cast<uint32_t>(code_gpu_address >> 40U);
  result.command[6] = 0;
  result.command[7] = kPm4GlobalInvalidate;

  result.packet.header = kAqlPacketTypeVendorSpecific;
  result.packet.vendor_header = kAmdAqlFormatPm4IndirectBuffer;
  result.packet.indirect_buffer[0] = pm4_header(kPm4IndirectBufferOpcode, 4);
  result.packet.indirect_buffer[1] =
      static_cast<uint32_t>(command_gpu_address) & 0xfffffffcU;
  result.packet.indirect_buffer[2] =
      static_cast<uint32_t>(command_gpu_address >> 32U) & 0xffffU;
  result.packet.indirect_buffer[3] =
      static_cast<uint32_t>(result.command.size()) | kPm4IndirectBufferValid;
  result.packet.dword_count_remaining = 0xa;
  result.packet.completion_signal = completion_signal;
  return result;
}

} // namespace light_rocr::arch::gfx11
