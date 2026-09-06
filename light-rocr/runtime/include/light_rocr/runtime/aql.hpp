#ifndef LIGHT_ROCR_RUNTIME_AQL_HPP
#define LIGHT_ROCR_RUNTIME_AQL_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace light_rocr::runtime {

inline constexpr uint16_t kAqlPacketTypeInvalid = 1;
inline constexpr uint16_t kAqlPacketTypeKernelDispatch = 2;
inline constexpr uint16_t kAqlFenceScopeSystem = 2;
inline constexpr uint16_t kAqlPacketHeaderTypeShift = 0;
inline constexpr uint16_t kAqlPacketHeaderAcquireFenceScopeShift = 9;
inline constexpr uint16_t kAqlPacketHeaderReleaseFenceScopeShift = 11;
inline constexpr uint16_t kAqlKernelDispatchDimensionsShift = 0;
inline constexpr uint16_t kAqlKernelDispatchHeader = static_cast<uint16_t>(
    (kAqlPacketTypeKernelDispatch << kAqlPacketHeaderTypeShift) |
    (kAqlFenceScopeSystem << kAqlPacketHeaderAcquireFenceScopeShift) |
    (kAqlFenceScopeSystem << kAqlPacketHeaderReleaseFenceScopeShift));
inline constexpr uint64_t kAmdKernelDescriptorAlignment = 64;
inline constexpr uint64_t kAmdKernargMinimumAlignment = 16;
inline constexpr uint64_t kAmdSignalAlignment = 64;
// The first runtime target is gfx1101. Keep its dispatch limits explicit until
// topology discovery grows a transport-independent ISA-limit model.
inline constexpr uint16_t kGfx1101WorkgroupMaximumDimension = 1024;
inline constexpr uint32_t kGfx1101WorkgroupMaximumSize = 1024;

// Self-authored HSA AQL kernel-dispatch packet ABI. The queue transport
// publishes header last with release ordering after writing the remaining 62
// bytes into a 64-byte-aligned ring slot.
struct alignas(64) AqlKernelDispatchPacket {
  uint16_t header = kAqlPacketTypeInvalid;
  uint16_t setup = 0;
  uint16_t workgroup_size_x = 0;
  uint16_t workgroup_size_y = 0;
  uint16_t workgroup_size_z = 0;
  uint16_t reserved0 = 0;
  uint32_t grid_size_x = 0;
  uint32_t grid_size_y = 0;
  uint32_t grid_size_z = 0;
  uint32_t private_segment_size = 0;
  uint32_t group_segment_size = 0;
  uint64_t kernel_object = 0;
  uint64_t kernarg_address = 0;
  uint64_t reserved2 = 0;
  uint64_t completion_signal = 0;
};

static_assert(std::is_standard_layout_v<AqlKernelDispatchPacket>);
static_assert(sizeof(AqlKernelDispatchPacket) == 64);
static_assert(alignof(AqlKernelDispatchPacket) == 64);
static_assert(offsetof(AqlKernelDispatchPacket, header) == 0);
static_assert(offsetof(AqlKernelDispatchPacket, setup) == 2);
static_assert(offsetof(AqlKernelDispatchPacket, workgroup_size_x) == 4);
static_assert(offsetof(AqlKernelDispatchPacket, grid_size_x) == 12);
static_assert(offsetof(AqlKernelDispatchPacket, private_segment_size) == 24);
static_assert(offsetof(AqlKernelDispatchPacket, kernel_object) == 32);
static_assert(offsetof(AqlKernelDispatchPacket, kernarg_address) == 40);
static_assert(offsetof(AqlKernelDispatchPacket, reserved2) == 48);
static_assert(offsetof(AqlKernelDispatchPacket, completion_signal) == 56);

struct KernelDispatchSpec {
  uint16_t dimensions = 1;
  uint16_t workgroup_size_x = 1;
  uint16_t workgroup_size_y = 1;
  uint16_t workgroup_size_z = 1;
  uint32_t grid_size_x = 1;
  uint32_t grid_size_y = 1;
  uint32_t grid_size_z = 1;
  uint32_t private_segment_size = 0;
  uint32_t group_segment_size = 0;
  uint64_t kernel_object = 0;
  uint64_t kernarg_address = 0;
  uint64_t completion_signal = 0;
};

enum class AqlPacketError {
  None,
  InvalidHeader,
  InvalidDimensions,
  InvalidWorkgroupSize,
  InvalidGridSize,
  InvalidKernelObject,
  InvalidKernargAddress,
  InvalidCompletionSignal,
  NonzeroReservedField,
};

struct AqlPacketStatus {
  AqlPacketError error = AqlPacketError::None;
  std::string message;

  explicit operator bool() const { return error == AqlPacketError::None; }
};

struct AqlPacketResult {
  AqlPacketStatus status;
  AqlKernelDispatchPacket packet;

  explicit operator bool() const { return static_cast<bool>(status); }
};

[[nodiscard]] const char *aql_packet_error_name(AqlPacketError error);
[[nodiscard]] AqlPacketStatus
validate_kernel_dispatch_packet(const AqlKernelDispatchPacket &packet);
[[nodiscard]] AqlPacketResult
make_kernel_dispatch_packet(const KernelDispatchSpec &spec);

} // namespace light_rocr::runtime

#endif
