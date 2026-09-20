#ifndef LIGHT_ROCR_ARCH_GFX11_CODE_CACHE_HPP
#define LIGHT_ROCR_ARCH_GFX11_CODE_CACHE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace light_rocr::arch::gfx11 {

inline constexpr uint16_t kAqlPacketTypeInvalid = 1;
inline constexpr uint16_t kAqlPacketTypeVendorSpecific = 0;
inline constexpr uint16_t kAmdAqlFormatPm4IndirectBuffer = 1;

struct alignas(64) AqlPm4IndirectBufferPacket {
  uint16_t header = kAqlPacketTypeInvalid;
  uint16_t vendor_header = 0;
  uint32_t indirect_buffer[4]{};
  uint32_t dword_count_remaining = 0;
  uint32_t reserved[8]{};
  uint64_t completion_signal = 0;
};

static_assert(std::is_standard_layout_v<AqlPm4IndirectBufferPacket>);
static_assert(sizeof(AqlPm4IndirectBufferPacket) == 64);
static_assert(alignof(AqlPm4IndirectBufferPacket) == 64);
static_assert(offsetof(AqlPm4IndirectBufferPacket, header) == 0);
static_assert(offsetof(AqlPm4IndirectBufferPacket, vendor_header) == 2);
static_assert(offsetof(AqlPm4IndirectBufferPacket, indirect_buffer) == 4);
static_assert(offsetof(AqlPm4IndirectBufferPacket, dword_count_remaining) ==
              20);
static_assert(offsetof(AqlPm4IndirectBufferPacket, completion_signal) == 56);

enum class CodeCacheCommandError {
  None,
  InvalidCodeRange,
  InvalidCommandAddress,
  InvalidCompletionSignal,
};

struct CodeCacheCommandResult {
  CodeCacheCommandError error = CodeCacheCommandError::None;
  std::array<uint32_t, 8> command{};
  AqlPm4IndirectBufferPacket packet;

  explicit operator bool() const {
    return error == CodeCacheCommandError::None;
  }
};

[[nodiscard]] CodeCacheCommandResult
build_code_cache_invalidate(uint64_t code_gpu_address, uint64_t code_size,
                            uint64_t command_gpu_address,
                            uint64_t completion_signal);

[[nodiscard]] const char *
code_cache_command_error_name(CodeCacheCommandError error);

} // namespace light_rocr::arch::gfx11

#endif
