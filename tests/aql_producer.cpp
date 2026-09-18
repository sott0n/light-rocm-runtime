#include "aql_producer.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

using lrrt_internal::AqlKernelDispatchPacket;
using lrrt_internal::AqlQueueProducerOps;
using lrrt_internal::AqlSubmitError;

struct FakeQueue {
  alignas(64) std::array<uint8_t, 128> ring{};
  uint64_t read_index = 0;
  uint64_t write_index = 0;
  uint64_t doorbell = UINT64_MAX;
  bool reserve_succeeds = true;
  bool doorbell_succeeds = true;
};

uint64_t load_read_index(void *context) {
  return static_cast<FakeQueue *>(context)->read_index;
}

uint64_t load_write_index(void *context) {
  return static_cast<FakeQueue *>(context)->write_index;
}

bool reserve_packet(void *context, uint64_t *packet_id) {
  auto *queue = static_cast<FakeQueue *>(context);
  if (!queue->reserve_succeeds) {
    return false;
  }
  *packet_id = queue->write_index++;
  return true;
}

bool ring_doorbell(void *context, uint64_t packet_id) {
  auto *queue = static_cast<FakeQueue *>(context);
  if (!queue->doorbell_succeeds) {
    return false;
  }
  queue->doorbell = packet_id;
  return true;
}

AqlQueueProducerOps producer_ops(FakeQueue *queue) {
  return {queue,
          queue->ring.data(),
          2,
          load_read_index,
          load_write_index,
          reserve_packet,
          ring_doorbell,
          nullptr};
}

lrrt_internal::AqlKernelDispatchParameters dispatch_parameters() {
  return {256, 4, 1, 64, 2, 1, 272, 1024, 0x1000, 0x2000, 0x3000};
}

bool packet_at(const FakeQueue &queue, size_t index,
               const AqlKernelDispatchPacket &expected) {
  AqlKernelDispatchPacket actual;
  std::memcpy(&actual, queue.ring.data() + index * sizeof(actual),
              sizeof(actual));
  return std::memcmp(&actual, &expected, sizeof(actual)) == 0;
}

} // namespace

int main() {
  FakeQueue queue;
  queue.ring.fill(0xa5);
  const AqlQueueProducerOps ops = producer_ops(&queue);

  const AqlKernelDispatchPacket first =
      lrrt_internal::build_aql_kernel_dispatch_packet(dispatch_parameters(),
                                                      {0, false});
  const auto first_result = lrrt_internal::submit_aql_kernel_dispatch(
      ops, dispatch_parameters(), {0, false});
  if (!first_result || first_result.packet_id != 0 || queue.doorbell != 0 ||
      first.header != 0x1402 || !packet_at(queue, 0, first)) {
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
  const auto second_result = lrrt_internal::submit_aql_kernel_dispatch(
      ops, dispatch_parameters(), {1, false});
  if (!second_result || second_result.packet_id != 1 || queue.doorbell != 1 ||
      second.header != 0x1502 || queue.ring[64] != 0x02 ||
      queue.ring[65] != 0x15 || !packet_at(queue, 1, second)) {
    std::cerr << "barrier dispatch was not published correctly\n";
    return 1;
  }

  const auto full_result = lrrt_internal::submit_aql_kernel_dispatch(
      ops, dispatch_parameters(), {0, false});
  if (full_result.error != AqlSubmitError::QueueFull ||
      queue.write_index != 2 || queue.doorbell != 1) {
    std::cerr << "full queue was modified\n";
    return 1;
  }

  FakeQueue invalid_queue;
  auto invalid_parameters = dispatch_parameters();
  invalid_parameters.kernel_object = 0;
  const auto invalid_result = lrrt_internal::submit_aql_kernel_dispatch(
      producer_ops(&invalid_queue), invalid_parameters, {0, false});
  if (invalid_result.error != AqlSubmitError::InvalidPacket ||
      invalid_queue.write_index != 0 || invalid_queue.doorbell != UINT64_MAX) {
    std::cerr << "invalid packet changed queue state\n";
    return 1;
  }

  std::cout << "aql_producer: ok\n";
  return 0;
}
