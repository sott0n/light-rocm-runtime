#include "aql_producer.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

using lrrt_internal::AqlBarrierAndPacket;
using lrrt_internal::AqlKernelDispatchPacket;
using lrrt_internal::AqlQueueProducerOps;
using lrrt_internal::AqlSubmitError;

enum class QueueOperation {
  Validate,
  LoadReadIndex,
  LoadWriteIndex,
  Reserve,
  RingDoorbell,
};

struct FakeQueue {
  alignas(64) std::array<uint8_t, 512> ring{};
  uint64_t packet_count = 2;
  uint64_t read_index = 0;
  uint64_t write_index = 0;
  uint64_t doorbell = UINT64_MAX;
  bool reserve_succeeds = true;
  bool packet_is_accepted = true;
  const AqlKernelDispatchPacket *expected_at_doorbell = nullptr;
  const AqlBarrierAndPacket *expected_barrier_at_doorbell = nullptr;
  bool packet_complete_at_doorbell = false;
  std::array<QueueOperation, 5> operations{};
  size_t operation_count = 0;
};

void record(FakeQueue *queue, QueueOperation operation) {
  if (queue->operation_count < queue->operations.size()) {
    queue->operations[queue->operation_count] = operation;
  }
  ++queue->operation_count;
}

uint64_t load_read_index(void *context) {
  auto *queue = static_cast<FakeQueue *>(context);
  record(queue, QueueOperation::LoadReadIndex);
  return queue->read_index;
}

uint64_t load_write_index(void *context) {
  auto *queue = static_cast<FakeQueue *>(context);
  record(queue, QueueOperation::LoadWriteIndex);
  return queue->write_index;
}

bool reserve_packet(void *context, uint64_t packet_count,
                    uint64_t *first_packet_id) {
  auto *queue = static_cast<FakeQueue *>(context);
  record(queue, QueueOperation::Reserve);
  if (!queue->reserve_succeeds) {
    return false;
  }
  *first_packet_id = queue->write_index;
  queue->write_index += packet_count;
  return true;
}

template <typename Packet>
bool packet_at(const FakeQueue &queue, size_t index, const Packet &expected) {
  Packet actual;
  std::memcpy(&actual, queue.ring.data() + index * sizeof(actual),
              sizeof(actual));
  return std::memcmp(&actual, &expected, sizeof(actual)) == 0;
}

void ring_doorbell(void *context, uint64_t packet_id) {
  auto *queue = static_cast<FakeQueue *>(context);
  record(queue, QueueOperation::RingDoorbell);
  if (queue->expected_at_doorbell != nullptr) {
    queue->packet_complete_at_doorbell =
        packet_at(*queue, packet_id & (queue->packet_count - 1),
                  *queue->expected_at_doorbell);
  } else if (queue->expected_barrier_at_doorbell != nullptr) {
    queue->packet_complete_at_doorbell =
        packet_at(*queue, packet_id & (queue->packet_count - 1),
                  *queue->expected_barrier_at_doorbell);
  }
  queue->doorbell = packet_id;
}

bool validate_packet(void *context, const AqlKernelDispatchPacket &) {
  auto *queue = static_cast<FakeQueue *>(context);
  record(queue, QueueOperation::Validate);
  return queue->packet_is_accepted;
}

AqlQueueProducerOps producer_ops(FakeQueue *queue) {
  return {queue,           queue->ring.data(), queue->packet_count,
          load_read_index, load_write_index,   reserve_packet,
          ring_doorbell,   validate_packet};
}

lrrt_internal::AqlKernelDispatchParameters dispatch_parameters() {
  return {256, 4, 1, 64, 2, 1, 272, 1024, 0x1000, 0x2000, 0x3000};
}

template <size_t Size>
bool operations_are(const FakeQueue &queue,
                    const std::array<QueueOperation, Size> &expected) {
  if (queue.operation_count != expected.size()) {
    return false;
  }
  for (size_t index = 0; index < expected.size(); ++index) {
    if (queue.operations[index] != expected[index]) {
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  FakeQueue queue;
  queue.ring.fill(0xa5);
  const AqlQueueProducerOps ops = producer_ops(&queue);

  const AqlKernelDispatchPacket first =
      lrrt_internal::build_aql_kernel_dispatch_packet(dispatch_parameters(),
                                                      {0, false});
  queue.expected_at_doorbell = &first;
  const auto first_result = lrrt_internal::submit_aql_kernel_dispatch(
      ops, dispatch_parameters(), {0, false});
  if (!first_result || first_result.packet_id != 0 || queue.doorbell != 0 ||
      first.header != 0x1402 || !packet_at(queue, 0, first) ||
      !queue.packet_complete_at_doorbell ||
      !operations_are(queue, std::array{QueueOperation::Validate,
                                        QueueOperation::LoadReadIndex,
                                        QueueOperation::LoadWriteIndex,
                                        QueueOperation::Reserve,
                                        QueueOperation::RingDoorbell})) {
    std::cerr << "first dispatch was not published correctly\n";
    return 1;
  }

  const AqlKernelDispatchPacket second =
      lrrt_internal::build_aql_kernel_dispatch_packet(dispatch_parameters(),
                                                      {1, false});
  const AqlKernelDispatchPacket dependency_ordered =
      lrrt_internal::build_aql_kernel_dispatch_packet(dispatch_parameters(),
                                                      {0, true});
  if (dependency_ordered.header != 0x1502) {
    std::cerr << "dependency ordering did not select the barrier header\n";
    return 1;
  }
  queue.operation_count = 0;
  queue.packet_complete_at_doorbell = false;
  queue.expected_at_doorbell = &second;
  const auto second_result = lrrt_internal::submit_aql_kernel_dispatch(
      ops, dispatch_parameters(), {1, false});
  if (!second_result || second_result.packet_id != 1 || queue.doorbell != 1 ||
      second.header != 0x1502 || queue.ring[64] != 0x02 ||
      queue.ring[65] != 0x15 || !packet_at(queue, 1, second) ||
      !queue.packet_complete_at_doorbell ||
      !operations_are(queue, std::array{QueueOperation::Validate,
                                        QueueOperation::LoadReadIndex,
                                        QueueOperation::LoadWriteIndex,
                                        QueueOperation::Reserve,
                                        QueueOperation::RingDoorbell})) {
    std::cerr << "barrier dispatch was not published correctly\n";
    return 1;
  }

  FakeQueue barrier_queue;
  barrier_queue.ring.fill(0x7c);
  const lrrt_internal::AqlBarrierAndParameters barrier_parameters = {
      {0x100, 0x200, 0, 0x400, 0x500}, 0x600};
  const AqlBarrierAndPacket barrier =
      lrrt_internal::build_aql_barrier_and_packet(barrier_parameters);
  const auto barrier_result = lrrt_internal::submit_aql_barrier_and(
      producer_ops(&barrier_queue), barrier_parameters);
  if (!barrier_result || barrier_result.packet_id != 0 ||
      barrier.header != 0x1503 || barrier_queue.doorbell != 0 ||
      !packet_at(barrier_queue, 0, barrier) ||
      !operations_are(barrier_queue,
                      std::array{QueueOperation::LoadReadIndex,
                                 QueueOperation::LoadWriteIndex,
                                 QueueOperation::Reserve,
                                 QueueOperation::RingDoorbell})) {
    std::cerr << "barrier-and packet was not published correctly\n";
    return 1;
  }

  barrier_queue.operation_count = 0;
  barrier_queue.read_index = 0;
  barrier_queue.write_index = 2;
  barrier_queue.doorbell = UINT64_MAX;
  const auto barrier_ring_before_full = barrier_queue.ring;
  const auto full_barrier_result = lrrt_internal::submit_aql_barrier_and(
      producer_ops(&barrier_queue), barrier_parameters);
  if (full_barrier_result.error != AqlSubmitError::QueueFull ||
      barrier_queue.write_index != 2 || barrier_queue.doorbell != UINT64_MAX ||
      barrier_queue.ring != barrier_ring_before_full ||
      !operations_are(barrier_queue,
                      std::array{QueueOperation::LoadReadIndex,
                                 QueueOperation::LoadWriteIndex})) {
    std::cerr << "full queue was modified by barrier submission\n";
    return 1;
  }

  FakeQueue barrier_batch_queue;
  barrier_batch_queue.packet_count = 4;
  barrier_batch_queue.ring.fill(0x4d);
  const std::array<lrrt_internal::AqlBarrierAndParameters, 2>
      barrier_batch_parameters = {
          {{{0x10, 0x20, 0, 0, 0}, 0x30}, {{0x40, 0x50, 0x60, 0, 0}, 0x70}}};
  const AqlBarrierAndPacket last_batch_barrier =
      lrrt_internal::build_aql_barrier_and_packet(barrier_batch_parameters[1]);
  barrier_batch_queue.expected_barrier_at_doorbell = &last_batch_barrier;
  const auto barrier_batch_result = lrrt_internal::submit_aql_barriers_and(
      producer_ops(&barrier_batch_queue), barrier_batch_parameters.data(),
      barrier_batch_parameters.size());
  if (!barrier_batch_result || barrier_batch_result.packet_id != 1 ||
      barrier_batch_queue.write_index != 2 ||
      barrier_batch_queue.doorbell != 1 ||
      !packet_at(barrier_batch_queue, 0,
                 lrrt_internal::build_aql_barrier_and_packet(
                     barrier_batch_parameters[0])) ||
      !packet_at(barrier_batch_queue, 1, last_batch_barrier) ||
      !barrier_batch_queue.packet_complete_at_doorbell ||
      !operations_are(barrier_batch_queue,
                      std::array{QueueOperation::LoadReadIndex,
                                 QueueOperation::LoadWriteIndex,
                                 QueueOperation::Reserve,
                                 QueueOperation::RingDoorbell})) {
    std::cerr << "barrier batch was not published atomically\n";
    return 1;
  }

  barrier_batch_queue.operation_count = 0;
  barrier_batch_queue.read_index = 3;
  barrier_batch_queue.write_index = 3;
  barrier_batch_queue.doorbell = UINT64_MAX;
  barrier_batch_queue.packet_complete_at_doorbell = false;
  const auto wrapping_barrier_batch_result =
      lrrt_internal::submit_aql_barriers_and(producer_ops(&barrier_batch_queue),
                                             barrier_batch_parameters.data(),
                                             barrier_batch_parameters.size());
  if (!wrapping_barrier_batch_result ||
      wrapping_barrier_batch_result.packet_id != 4 ||
      barrier_batch_queue.write_index != 5 ||
      barrier_batch_queue.doorbell != 4 ||
      !packet_at(barrier_batch_queue, 3,
                 lrrt_internal::build_aql_barrier_and_packet(
                     barrier_batch_parameters[0])) ||
      !packet_at(barrier_batch_queue, 0, last_batch_barrier) ||
      !barrier_batch_queue.packet_complete_at_doorbell ||
      !operations_are(barrier_batch_queue,
                      std::array{QueueOperation::LoadReadIndex,
                                 QueueOperation::LoadWriteIndex,
                                 QueueOperation::Reserve,
                                 QueueOperation::RingDoorbell})) {
    std::cerr << "wrapping barrier batch was not published atomically\n";
    return 1;
  }

  barrier_batch_queue.operation_count = 0;
  barrier_batch_queue.read_index = 0;
  barrier_batch_queue.write_index = 3;
  barrier_batch_queue.doorbell = UINT64_MAX;
  const auto barrier_batch_ring_before_full = barrier_batch_queue.ring;
  const auto full_barrier_batch_result = lrrt_internal::submit_aql_barriers_and(
      producer_ops(&barrier_batch_queue), barrier_batch_parameters.data(),
      barrier_batch_parameters.size());
  if (full_barrier_batch_result.error != AqlSubmitError::QueueFull ||
      barrier_batch_queue.write_index != 3 ||
      barrier_batch_queue.doorbell != UINT64_MAX ||
      barrier_batch_queue.ring != barrier_batch_ring_before_full ||
      !operations_are(barrier_batch_queue,
                      std::array{QueueOperation::LoadReadIndex,
                                 QueueOperation::LoadWriteIndex})) {
    std::cerr << "full queue was modified by barrier batch submission\n";
    return 1;
  }

  FakeQueue barrier_batch_reserve_failure_queue;
  barrier_batch_reserve_failure_queue.packet_count = 4;
  barrier_batch_reserve_failure_queue.ring.fill(0x5e);
  barrier_batch_reserve_failure_queue.reserve_succeeds = false;
  const auto barrier_batch_ring_before_reserve_failure =
      barrier_batch_reserve_failure_queue.ring;
  const auto barrier_batch_reserve_failure =
      lrrt_internal::submit_aql_barriers_and(
          producer_ops(&barrier_batch_reserve_failure_queue),
          barrier_batch_parameters.data(), barrier_batch_parameters.size());
  if (barrier_batch_reserve_failure.error != AqlSubmitError::ReserveFailed ||
      barrier_batch_reserve_failure_queue.write_index != 0 ||
      barrier_batch_reserve_failure_queue.doorbell != UINT64_MAX ||
      barrier_batch_reserve_failure_queue.ring !=
          barrier_batch_ring_before_reserve_failure ||
      !operations_are(barrier_batch_reserve_failure_queue,
                      std::array{QueueOperation::LoadReadIndex,
                                 QueueOperation::LoadWriteIndex,
                                 QueueOperation::Reserve})) {
    std::cerr << "barrier batch reserve failure changed queue state\n";
    return 1;
  }

  queue.operation_count = 0;
  const auto ring_before_full = queue.ring;
  const auto full_result = lrrt_internal::submit_aql_kernel_dispatch(
      ops, dispatch_parameters(), {0, false});
  if (full_result.error != AqlSubmitError::QueueFull ||
      queue.write_index != 2 || queue.doorbell != 1 ||
      queue.ring != ring_before_full ||
      !operations_are(queue, std::array{QueueOperation::Validate,
                                        QueueOperation::LoadReadIndex,
                                        QueueOperation::LoadWriteIndex})) {
    std::cerr << "full queue was modified\n";
    return 1;
  }

  FakeQueue invalid_queue;
  auto invalid_parameters = dispatch_parameters();
  invalid_parameters.kernel_object = 0;
  const auto invalid_result = lrrt_internal::submit_aql_kernel_dispatch(
      producer_ops(&invalid_queue), invalid_parameters, {0, false});
  if (invalid_result.error != AqlSubmitError::InvalidPacket ||
      invalid_queue.write_index != 0 || invalid_queue.doorbell != UINT64_MAX ||
      invalid_queue.operation_count != 0) {
    std::cerr << "invalid packet changed queue state\n";
    return 1;
  }

  FakeQueue rejected_queue;
  rejected_queue.ring.fill(0x5a);
  rejected_queue.packet_is_accepted = false;
  const auto ring_before_rejection = rejected_queue.ring;
  const auto rejected_result = lrrt_internal::submit_aql_kernel_dispatch(
      producer_ops(&rejected_queue), dispatch_parameters(), {0, false});
  if (rejected_result.error != AqlSubmitError::InvalidPacket ||
      rejected_queue.write_index != 0 ||
      rejected_queue.doorbell != UINT64_MAX ||
      rejected_queue.ring != ring_before_rejection ||
      !operations_are(rejected_queue, std::array{QueueOperation::Validate})) {
    std::cerr << "adapter-rejected packet changed queue state\n";
    return 1;
  }

  FakeQueue reserve_failure_queue;
  reserve_failure_queue.ring.fill(0x3c);
  reserve_failure_queue.reserve_succeeds = false;
  const auto ring_before_reserve_failure = reserve_failure_queue.ring;
  const auto reserve_failure = lrrt_internal::submit_aql_kernel_dispatch(
      producer_ops(&reserve_failure_queue), dispatch_parameters(), {0, false});
  if (reserve_failure.error != AqlSubmitError::ReserveFailed ||
      reserve_failure_queue.write_index != 0 ||
      reserve_failure_queue.doorbell != UINT64_MAX ||
      reserve_failure_queue.ring != ring_before_reserve_failure ||
      !operations_are(reserve_failure_queue,
                      std::array{QueueOperation::Validate,
                                 QueueOperation::LoadReadIndex,
                                 QueueOperation::LoadWriteIndex,
                                 QueueOperation::Reserve})) {
    std::cerr << "reserve failure changed queue state\n";
    return 1;
  }

  FakeQueue sequence_queue;
  sequence_queue.packet_count = 4;
  sequence_queue.ring.fill(0x3d);
  const std::array<lrrt_internal::AqlBarrierAndParameters, 2>
      sequence_barriers = {
          {{{0x10, 0x20, 0x30, 0x40, 0x50}, 0}, {{0x60, 0, 0, 0, 0}, 0}}};
  const AqlKernelDispatchPacket sequence_dispatch =
      lrrt_internal::build_aql_kernel_dispatch_packet(dispatch_parameters(),
                                                      {0, true});
  sequence_queue.expected_at_doorbell = &sequence_dispatch;
  const auto sequence_result =
      lrrt_internal::submit_aql_barriers_and_kernel_dispatch(
          producer_ops(&sequence_queue), sequence_barriers.data(),
          sequence_barriers.size(), dispatch_parameters(), {0, true});
  if (!sequence_result || sequence_result.packet_id != 2 ||
      sequence_queue.write_index != 3 || sequence_queue.doorbell != 2 ||
      !packet_at(
          sequence_queue, 0,
          lrrt_internal::build_aql_barrier_and_packet(sequence_barriers[0])) ||
      !packet_at(
          sequence_queue, 1,
          lrrt_internal::build_aql_barrier_and_packet(sequence_barriers[1])) ||
      !packet_at(sequence_queue, 2, sequence_dispatch) ||
      !sequence_queue.packet_complete_at_doorbell ||
      !operations_are(
          sequence_queue,
          std::array{QueueOperation::Validate, QueueOperation::LoadReadIndex,
                     QueueOperation::LoadWriteIndex, QueueOperation::Reserve,
                     QueueOperation::RingDoorbell})) {
    std::cerr << "barrier and dispatch sequence was not published atomically\n";
    return 1;
  }

  sequence_queue.operation_count = 0;
  sequence_queue.read_index = 0;
  sequence_queue.write_index = 2;
  sequence_queue.doorbell = UINT64_MAX;
  const auto sequence_ring_before_full = sequence_queue.ring;
  const auto full_sequence_result =
      lrrt_internal::submit_aql_barriers_and_kernel_dispatch(
          producer_ops(&sequence_queue), sequence_barriers.data(),
          sequence_barriers.size(), dispatch_parameters(), {0, true});
  if (full_sequence_result.error != AqlSubmitError::QueueFull ||
      sequence_queue.write_index != 2 ||
      sequence_queue.doorbell != UINT64_MAX ||
      sequence_queue.ring != sequence_ring_before_full ||
      !operations_are(sequence_queue,
                      std::array{QueueOperation::Validate,
                                 QueueOperation::LoadReadIndex,
                                 QueueOperation::LoadWriteIndex})) {
    std::cerr << "full queue was modified by sequence submission\n";
    return 1;
  }

  FakeQueue sequence_reserve_failure_queue;
  sequence_reserve_failure_queue.packet_count = 4;
  sequence_reserve_failure_queue.ring.fill(0x6a);
  sequence_reserve_failure_queue.reserve_succeeds = false;
  const auto sequence_ring_before_reserve_failure =
      sequence_reserve_failure_queue.ring;
  const auto sequence_reserve_failure =
      lrrt_internal::submit_aql_barriers_and_kernel_dispatch(
          producer_ops(&sequence_reserve_failure_queue),
          sequence_barriers.data(), sequence_barriers.size(),
          dispatch_parameters(), {0, true});
  if (sequence_reserve_failure.error != AqlSubmitError::ReserveFailed ||
      sequence_reserve_failure_queue.write_index != 0 ||
      sequence_reserve_failure_queue.doorbell != UINT64_MAX ||
      sequence_reserve_failure_queue.ring !=
          sequence_ring_before_reserve_failure ||
      !operations_are(sequence_reserve_failure_queue,
                      std::array{QueueOperation::Validate,
                                 QueueOperation::LoadReadIndex,
                                 QueueOperation::LoadWriteIndex,
                                 QueueOperation::Reserve})) {
    std::cerr << "sequence reserve failure changed queue state\n";
    return 1;
  }

  std::cout << "aql_producer: ok\n";
  return 0;
}
