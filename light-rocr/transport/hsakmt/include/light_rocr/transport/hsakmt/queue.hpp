#ifndef LIGHT_ROCR_TRANSPORT_HSAKMT_QUEUE_HPP
#define LIGHT_ROCR_TRANSPORT_HSAKMT_QUEUE_HPP

#include "light_rocr/transport/hsakmt/memory.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace light_rocr::transport::hsakmt {

inline constexpr uint64_t kAqlPacketSize = 64;
inline constexpr uint64_t kAqlRingMinimumSize = 4096;
inline constexpr uint64_t kAqlRingMaximumSize = 131072 * kAqlPacketSize;
inline constexpr uint64_t kAqlRingDefaultSize = 64 * 1024;

enum class AqlQueueError {
  None,
  InvalidSession,
  InvalidRingSize,
  InvalidNode,
  ConfigureScratch,
  AllocateRing,
  AllocateControl,
  AllocateScratch,
  CreateQueue,
  DestroyQueue,
  ReleaseScratch,
  ReleaseControl,
  ReleaseRing,
};

struct AqlQueueStatus {
  AqlQueueError error = AqlQueueError::None;
  uint32_t hsakmt_status = 0;
  std::string message;

  explicit operator bool() const { return error == AqlQueueError::None; }
};

struct AqlQueueState;

enum class AqlQueuePrimitiveError {
  None,
  InvalidQueue,
  InvalidIncrement,
  IndexOverflow,
  InvalidDoorbell,
};

struct AqlQueuePrimitiveStatus {
  AqlQueuePrimitiveError error = AqlQueuePrimitiveError::None;
  std::string message;

  explicit operator bool() const {
    return error == AqlQueuePrimitiveError::None;
  }
};

struct AqlQueueIndexResult {
  AqlQueuePrimitiveStatus status;
  uint64_t previous_index = 0;

  explicit operator bool() const { return static_cast<bool>(status); }
};

class AqlQueue {
public:
  AqlQueue();
  AqlQueue(const AqlQueue &) = delete;
  AqlQueue &operator=(const AqlQueue &) = delete;
  AqlQueue(AqlQueue &&other) noexcept;
  AqlQueue &operator=(AqlQueue &&other) noexcept;
  ~AqlQueue();

  [[nodiscard]] uint64_t queue_id() const;
  [[nodiscard]] uintptr_t doorbell_address() const;
  [[nodiscard]] void *ring_host_address() const;
  [[nodiscard]] uint64_t ring_gpu_address() const;
  [[nodiscard]] uint64_t ring_size() const;
  [[nodiscard]] uint64_t packet_count() const;
  [[nodiscard]] uint32_t scratch_private_segment_size() const;
  [[nodiscard]] uint64_t scratch_gpu_address() const;
  [[nodiscard]] uint64_t scratch_size() const;
  // Index accessors return zero after their control allocation is released.
  [[nodiscard]] uint64_t read_index_acquire() const;
  [[nodiscard]] uint64_t write_index_relaxed() const;
  // These packet-independent primitives mirror the producer-facing ROCr
  // queue operations. The caller owns capacity checks, ring writes, header
  // publication, and the choice of the doorbell value.
  [[nodiscard]] AqlQueueIndexResult
  add_write_index_scacq_screl(uint64_t increment);
  [[nodiscard]] AqlQueuePrimitiveStatus
  store_doorbell_screlease(uint64_t value);
  explicit operator bool() const;

  [[nodiscard]] AqlQueueStatus release();

private:
  friend class KfdSession;
  explicit AqlQueue(std::unique_ptr<AqlQueueState> state);
  void release_for_destruction() noexcept;

  std::unique_ptr<AqlQueueState> state_;
};

struct AqlQueueResult {
  AqlQueueStatus status;
  AqlQueue queue;

  explicit operator bool() const { return static_cast<bool>(status); }
};

[[nodiscard]] const char *aql_queue_error_name(AqlQueueError error);
[[nodiscard]] const char *
aql_queue_primitive_error_name(AqlQueuePrimitiveError error);

} // namespace light_rocr::transport::hsakmt

#endif
