#ifndef LRRT_AQL_PRODUCER_HPP_
#define LRRT_AQL_PRODUCER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace lrrt_internal {

inline constexpr uint16_t kAqlPacketTypeInvalid = 1;
inline constexpr uint16_t kAqlPacketTypeKernelDispatch = 2;
inline constexpr uint16_t kAqlPacketTypeBarrierAnd = 3;
inline constexpr uint16_t kAqlFenceScopeSystem = 2;
inline constexpr uint16_t kAqlPacketHeaderTypeShift = 0;
inline constexpr uint16_t kAqlPacketHeaderBarrierShift = 8;
inline constexpr uint16_t kAqlPacketHeaderAcquireFenceScopeShift = 9;
inline constexpr uint16_t kAqlPacketHeaderReleaseFenceScopeShift = 11;
inline constexpr uint16_t kAqlKernelDispatchDimensionsShift = 0;
inline constexpr uint16_t kAqlKernelDispatchHeader = static_cast<uint16_t>(
    (kAqlPacketTypeKernelDispatch << kAqlPacketHeaderTypeShift) |
    (kAqlFenceScopeSystem << kAqlPacketHeaderAcquireFenceScopeShift) |
    (kAqlFenceScopeSystem << kAqlPacketHeaderReleaseFenceScopeShift));
inline constexpr uint16_t kAqlBarrierAndHeader = static_cast<uint16_t>(
    (kAqlPacketTypeBarrierAnd << kAqlPacketHeaderTypeShift) |
    (uint16_t{1} << kAqlPacketHeaderBarrierShift) |
    (kAqlFenceScopeSystem << kAqlPacketHeaderAcquireFenceScopeShift) |
    (kAqlFenceScopeSystem << kAqlPacketHeaderReleaseFenceScopeShift));

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

struct alignas(64) AqlBarrierAndPacket {
  uint16_t header = kAqlPacketTypeInvalid;
  uint16_t reserved0 = 0;
  uint32_t reserved1 = 0;
  std::array<uint64_t, 5> dep_signal{};
  uint64_t reserved2 = 0;
  uint64_t completion_signal = 0;
};

static_assert(std::is_standard_layout_v<AqlBarrierAndPacket>);
static_assert(sizeof(AqlBarrierAndPacket) == 64);
static_assert(alignof(AqlBarrierAndPacket) == 64);
static_assert(offsetof(AqlBarrierAndPacket, header) == 0);
static_assert(offsetof(AqlBarrierAndPacket, dep_signal) == 8);
static_assert(offsetof(AqlBarrierAndPacket, reserved2) == 48);
static_assert(offsetof(AqlBarrierAndPacket, completion_signal) == 56);

struct AqlKernelDispatchParameters {
  uint32_t grid_size_x = 0;
  uint32_t grid_size_y = 0;
  uint32_t grid_size_z = 0;
  uint16_t workgroup_size_x = 0;
  uint16_t workgroup_size_y = 0;
  uint16_t workgroup_size_z = 0;
  uint32_t private_segment_size = 0;
  uint32_t group_segment_size = 0;
  uint64_t kernel_object = 0;
  uint64_t kernarg_address = 0;
  uint64_t completion_signal = 0;
};

struct AqlDispatchOrdering {
  size_t pending_dispatch_count = 0;
  bool has_pending_dependencies = false;
};

struct AqlBarrierAndParameters {
  std::array<uint64_t, 5> dependency_signals{};
  uint64_t completion_signal = 0;
};

using AqlIndexLoad = uint64_t (*)(void *context);
// A successful reservation owns one contiguous, non-wrapping packet-ID range.
// A failed reservation must not change the write index.
using AqlPacketReserve = bool (*)(void *context, uint64_t packet_count,
                                  uint64_t *first_packet_id);
// The packet is already visible when this is called, so a doorbell adapter
// must satisfy its preconditions before submission and cannot fail recoverably.
using AqlDoorbellStore = void (*)(void *context, uint64_t packet_id);
using AqlPacketValidator = bool (*)(void *context,
                                    const AqlKernelDispatchPacket &packet);

struct AqlQueueProducerOps {
  void *context = nullptr;
  void *ring_base = nullptr;
  uint64_t packet_count = 0;
  AqlIndexLoad load_read_index = nullptr;
  AqlIndexLoad load_write_index = nullptr;
  AqlPacketReserve reserve_packet = nullptr;
  AqlDoorbellStore ring_doorbell = nullptr;
  AqlPacketValidator validate_packet = nullptr;
};

enum class AqlSubmitError {
  None,
  InvalidPacket,
  InvalidQueue,
  QueueFull,
  ReserveFailed,
};

struct AqlSubmitResult {
  AqlSubmitError error = AqlSubmitError::None;
  uint64_t packet_id = 0;

  explicit operator bool() const { return error == AqlSubmitError::None; }
};

[[nodiscard]] AqlKernelDispatchPacket
build_aql_kernel_dispatch_packet(const AqlKernelDispatchParameters &parameters,
                                 const AqlDispatchOrdering &ordering);
[[nodiscard]] bool
valid_aql_kernel_dispatch_packet(const AqlKernelDispatchPacket &packet);
[[nodiscard]] AqlBarrierAndPacket
build_aql_barrier_and_packet(const AqlBarrierAndParameters &parameters);
[[nodiscard]] bool
valid_aql_barrier_and_packet(const AqlBarrierAndPacket &packet);
[[nodiscard]] AqlSubmitResult
submit_aql_kernel_dispatch(const AqlQueueProducerOps &queue,
                           const AqlKernelDispatchParameters &parameters,
                           const AqlDispatchOrdering &ordering);
[[nodiscard]] AqlSubmitResult
submit_aql_barrier_and(const AqlQueueProducerOps &queue,
                       const AqlBarrierAndParameters &parameters);
[[nodiscard]] AqlSubmitResult submit_aql_barriers_and_kernel_dispatch(
    const AqlQueueProducerOps &queue,
    const AqlBarrierAndParameters *barrier_parameters, size_t barrier_count,
    const AqlKernelDispatchParameters &dispatch_parameters,
    const AqlDispatchOrdering &ordering);

} // namespace lrrt_internal

#endif
