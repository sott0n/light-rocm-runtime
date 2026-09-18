#include "aql_producer.hpp"

#include <cstdint>
#include <cstring>

namespace lrrt_internal {
namespace {

bool is_power_of_two(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

uint16_t dispatch_dimensions(const AqlKernelDispatchParameters &parameters) {
  if (parameters.grid_size_z > 1 || parameters.workgroup_size_z > 1) {
    return 3;
  }
  if (parameters.grid_size_y > 1 || parameters.workgroup_size_y > 1) {
    return 2;
  }
  return 1;
}

void publish_packet(void *slot, const AqlKernelDispatchPacket &packet) {
  auto *slot_bytes = static_cast<uint8_t *>(slot);
  auto *slot_header = reinterpret_cast<uint16_t *>(slot_bytes);
  __atomic_store_n(slot_header, kAqlPacketTypeInvalid, __ATOMIC_RELAXED);
  const auto *packet_bytes = reinterpret_cast<const uint8_t *>(&packet);
  std::memcpy(slot_bytes + sizeof(packet.header),
              packet_bytes + sizeof(packet.header),
              sizeof(packet) - sizeof(packet.header));
  __atomic_store_n(slot_header, packet.header, __ATOMIC_RELEASE);
}

} // namespace

AqlKernelDispatchPacket
build_aql_kernel_dispatch_packet(const AqlKernelDispatchParameters &parameters,
                                 const AqlDispatchOrdering &ordering) {
  const bool wait_for_dependencies =
      ordering.pending_dispatch_count != 0 || ordering.has_pending_dependencies;
  AqlKernelDispatchPacket packet;
  packet.header = static_cast<uint16_t>(
      kAqlKernelDispatchHeader |
      (wait_for_dependencies ? uint16_t{1} << kAqlPacketHeaderBarrierShift
                             : uint16_t{0}));
  packet.setup = static_cast<uint16_t>(dispatch_dimensions(parameters)
                                       << kAqlKernelDispatchDimensionsShift);
  packet.workgroup_size_x = parameters.workgroup_size_x;
  packet.workgroup_size_y = parameters.workgroup_size_y;
  packet.workgroup_size_z = parameters.workgroup_size_z;
  packet.grid_size_x = parameters.grid_size_x;
  packet.grid_size_y = parameters.grid_size_y;
  packet.grid_size_z = parameters.grid_size_z;
  packet.private_segment_size = parameters.private_segment_size;
  packet.group_segment_size = parameters.group_segment_size;
  packet.kernel_object = parameters.kernel_object;
  packet.kernarg_address = parameters.kernarg_address;
  packet.completion_signal = parameters.completion_signal;
  return packet;
}

bool valid_aql_kernel_dispatch_packet(const AqlKernelDispatchPacket &packet) {
  constexpr uint16_t barrier_bit = uint16_t{1} << kAqlPacketHeaderBarrierShift;
  if ((packet.header & static_cast<uint16_t>(~barrier_bit)) !=
          kAqlKernelDispatchHeader ||
      (packet.setup & static_cast<uint16_t>(~uint16_t{0x3})) != 0) {
    return false;
  }
  const uint16_t dimensions =
      static_cast<uint16_t>(packet.setup >> kAqlKernelDispatchDimensionsShift);
  if (dimensions < 1 || dimensions > 3 || packet.workgroup_size_x == 0 ||
      packet.workgroup_size_y == 0 || packet.workgroup_size_z == 0 ||
      (dimensions < 2 && packet.workgroup_size_y != 1) ||
      (dimensions < 3 && packet.workgroup_size_z != 1) ||
      packet.grid_size_x < packet.workgroup_size_x ||
      packet.grid_size_y < packet.workgroup_size_y ||
      packet.grid_size_z < packet.workgroup_size_z ||
      (dimensions < 2 && packet.grid_size_y != 1) ||
      (dimensions < 3 && packet.grid_size_z != 1) ||
      packet.kernel_object == 0 || packet.reserved0 != 0 ||
      packet.reserved2 != 0) {
    return false;
  }
  return true;
}

AqlSubmitResult
submit_aql_kernel_dispatch(const AqlQueueProducerOps &queue,
                           const AqlKernelDispatchParameters &parameters,
                           const AqlDispatchOrdering &ordering) {
  if (queue.context == nullptr || queue.ring_base == nullptr ||
      !is_power_of_two(queue.packet_count) ||
      queue.load_read_index == nullptr || queue.load_write_index == nullptr ||
      queue.reserve_packet == nullptr || queue.ring_doorbell == nullptr) {
    return {AqlSubmitError::InvalidQueue, 0};
  }
  const AqlKernelDispatchPacket packet =
      build_aql_kernel_dispatch_packet(parameters, ordering);
  if (!valid_aql_kernel_dispatch_packet(packet) ||
      (queue.validate_packet != nullptr &&
       !queue.validate_packet(queue.context, packet))) {
    return {AqlSubmitError::InvalidPacket, 0};
  }

  const uint64_t read_index = queue.load_read_index(queue.context);
  const uint64_t write_index = queue.load_write_index(queue.context);
  if (write_index < read_index ||
      write_index - read_index >= queue.packet_count) {
    return {AqlSubmitError::QueueFull, 0};
  }

  uint64_t packet_id = 0;
  if (!queue.reserve_packet(queue.context, &packet_id)) {
    return {AqlSubmitError::ReserveFailed, 0};
  }
  const uint64_t slot_index = packet_id & (queue.packet_count - 1);
  auto *slot = static_cast<uint8_t *>(queue.ring_base) +
               static_cast<size_t>(slot_index * sizeof(packet));
  publish_packet(slot, packet);
  queue.ring_doorbell(queue.context, packet_id);
  return {AqlSubmitError::None, packet_id};
}

} // namespace lrrt_internal
