#ifndef LRRT_LIGHT_ROCR_AQL_QUEUE_HPP_
#define LRRT_LIGHT_ROCR_AQL_QUEUE_HPP_

#include "aql_producer.hpp"
#include "light_rocr/transport/hsakmt/queue.hpp"

#include <cassert>
#include <cstdint>

namespace lrrt_internal {

inline uint64_t light_rocr_load_read_index(void *context) {
  return static_cast<light_rocr::transport::hsakmt::AqlQueue *>(context)
      ->read_index_acquire();
}

inline uint64_t light_rocr_load_write_index(void *context) {
  return static_cast<light_rocr::transport::hsakmt::AqlQueue *>(context)
      ->write_index_relaxed();
}

inline bool light_rocr_reserve_packet(void *context, uint64_t packet_count,
                                      uint64_t *first_packet_id) {
  auto *queue = static_cast<light_rocr::transport::hsakmt::AqlQueue *>(context);
  const auto reserved = queue->add_write_index_scacq_screl(packet_count);
  if (!reserved) {
    return false;
  }
  *first_packet_id = reserved.previous_index;
  return true;
}

inline void light_rocr_ring_doorbell(void *context, uint64_t packet_id) {
  const auto status =
      static_cast<light_rocr::transport::hsakmt::AqlQueue *>(context)
          ->store_doorbell_screlease(packet_id);
  assert(status);
  (void)status;
}

inline AqlQueueProducerOps
light_rocr_producer_ops(light_rocr::transport::hsakmt::AqlQueue *queue,
                        AqlPacketValidator validate_packet = nullptr) {
  const bool valid = queue != nullptr && static_cast<bool>(*queue) &&
                     queue->ring_host_address() != nullptr &&
                     queue->doorbell_address() != 0;
  return {valid ? queue : nullptr,
          valid ? queue->ring_host_address() : nullptr,
          valid ? queue->packet_count() : 0,
          light_rocr_load_read_index,
          light_rocr_load_write_index,
          light_rocr_reserve_packet,
          light_rocr_ring_doorbell,
          validate_packet};
}

} // namespace lrrt_internal

#endif // LRRT_LIGHT_ROCR_AQL_QUEUE_HPP_
