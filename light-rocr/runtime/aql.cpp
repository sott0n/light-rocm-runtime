#include "light_rocr/runtime/aql.hpp"

#include <utility>

namespace light_rocr::runtime {
namespace {

AqlPacketStatus failure(AqlPacketError error, const char *message) {
  return {error, message};
}

} // namespace

const char *aql_packet_error_name(AqlPacketError error) {
  switch (error) {
  case AqlPacketError::None:
    return "none";
  case AqlPacketError::InvalidHeader:
    return "invalid_header";
  case AqlPacketError::InvalidDimensions:
    return "invalid_dimensions";
  case AqlPacketError::InvalidWorkgroupSize:
    return "invalid_workgroup_size";
  case AqlPacketError::InvalidGridSize:
    return "invalid_grid_size";
  case AqlPacketError::InvalidKernelObject:
    return "invalid_kernel_object";
  case AqlPacketError::InvalidKernargAddress:
    return "invalid_kernarg_address";
  case AqlPacketError::InvalidCompletionSignal:
    return "invalid_completion_signal";
  case AqlPacketError::NonzeroReservedField:
    return "nonzero_reserved_field";
  }
  return "unknown";
}

AqlPacketStatus
validate_kernel_dispatch_packet(const AqlKernelDispatchPacket &packet) {
  if (packet.header != kAqlKernelDispatchHeader) {
    return failure(AqlPacketError::InvalidHeader,
                   "initial dispatch requires a kernel packet with system "
                   "acquire and release fences");
  }
  constexpr uint16_t kDimensionsMask = 0x3;
  if ((packet.setup & static_cast<uint16_t>(~kDimensionsMask)) != 0) {
    return failure(AqlPacketError::InvalidDimensions,
                   "kernel-dispatch setup contains reserved bits");
  }
  const uint16_t dimensions =
      static_cast<uint16_t>(packet.setup >> kAqlKernelDispatchDimensionsShift);
  if (dimensions < 1 || dimensions > 3) {
    return failure(AqlPacketError::InvalidDimensions,
                   "kernel-dispatch dimensions must be 1, 2, or 3");
  }
  if (packet.workgroup_size_x == 0 || packet.workgroup_size_y == 0 ||
      packet.workgroup_size_z == 0) {
    return failure(AqlPacketError::InvalidWorkgroupSize,
                   "kernel-dispatch workgroup dimensions must be non-zero");
  }
  if ((dimensions < 2 && packet.workgroup_size_y != 1) ||
      (dimensions < 3 && packet.workgroup_size_z != 1)) {
    return failure(AqlPacketError::InvalidWorkgroupSize,
                   "unused workgroup dimensions must equal one");
  }
  if (packet.workgroup_size_x > kGfx1101WorkgroupMaximumDimension ||
      packet.workgroup_size_y > kGfx1101WorkgroupMaximumDimension ||
      packet.workgroup_size_z > kGfx1101WorkgroupMaximumDimension) {
    return failure(AqlPacketError::InvalidWorkgroupSize,
                   "workgroup dimension exceeds the gfx1101 limit");
  }
  const uint64_t workgroup_size =
      static_cast<uint64_t>(packet.workgroup_size_x) *
      static_cast<uint64_t>(packet.workgroup_size_y) *
      static_cast<uint64_t>(packet.workgroup_size_z);
  if (workgroup_size > kGfx1101WorkgroupMaximumSize) {
    return failure(AqlPacketError::InvalidWorkgroupSize,
                   "total workgroup size exceeds the gfx1101 limit");
  }
  if (packet.grid_size_x < packet.workgroup_size_x ||
      packet.grid_size_y < packet.workgroup_size_y ||
      packet.grid_size_z < packet.workgroup_size_z) {
    return failure(AqlPacketError::InvalidGridSize,
                   "grid dimensions must cover the workgroup dimensions");
  }
  if ((dimensions < 2 && packet.grid_size_y != 1) ||
      (dimensions < 3 && packet.grid_size_z != 1)) {
    return failure(AqlPacketError::InvalidGridSize,
                   "unused grid dimensions must equal one");
  }
  if (packet.kernel_object == 0 ||
      packet.kernel_object % kAmdKernelDescriptorAlignment != 0) {
    return failure(AqlPacketError::InvalidKernelObject,
                   "AMD kernel descriptor address must be 64-byte aligned");
  }
  if (packet.kernarg_address != 0 &&
      packet.kernarg_address % kAmdKernargMinimumAlignment != 0) {
    return failure(AqlPacketError::InvalidKernargAddress,
                   "kernarg address must be at least 16-byte aligned");
  }
  if (packet.completion_signal != 0 &&
      packet.completion_signal % kAmdSignalAlignment != 0) {
    return failure(AqlPacketError::InvalidCompletionSignal,
                   "AMD completion signal handle must be 64-byte aligned");
  }
  if (packet.reserved0 != 0 || packet.reserved2 != 0) {
    return failure(AqlPacketError::NonzeroReservedField,
                   "kernel-dispatch reserved fields must be zero");
  }
  return {};
}

AqlPacketResult make_kernel_dispatch_packet(const KernelDispatchSpec &spec) {
  AqlKernelDispatchPacket packet{};
  packet.header = kAqlKernelDispatchHeader;
  packet.setup = static_cast<uint16_t>(spec.dimensions
                                       << kAqlKernelDispatchDimensionsShift);
  packet.workgroup_size_x = spec.workgroup_size_x;
  packet.workgroup_size_y = spec.workgroup_size_y;
  packet.workgroup_size_z = spec.workgroup_size_z;
  packet.grid_size_x = spec.grid_size_x;
  packet.grid_size_y = spec.grid_size_y;
  packet.grid_size_z = spec.grid_size_z;
  packet.private_segment_size = spec.private_segment_size;
  packet.group_segment_size = spec.group_segment_size;
  packet.kernel_object = spec.kernel_object;
  packet.kernarg_address = spec.kernarg_address;
  packet.completion_signal = spec.completion_signal;

  AqlPacketStatus status = validate_kernel_dispatch_packet(packet);
  return {std::move(status), packet};
}

} // namespace light_rocr::runtime
