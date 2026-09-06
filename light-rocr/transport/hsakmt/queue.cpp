#include "light_rocr/transport/hsakmt/queue.hpp"

#include "light_rocr/transport/hsakmt/status.hpp"

#include <hsakmt/hsakmt.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <utility>

#if defined(__x86_64__)
#include <emmintrin.h>
#endif

namespace light_rocr::transport::hsakmt {
namespace {

struct alignas(64) AqlQueueControl {
  alignas(64) uint64_t read_index = 0;
  alignas(64) uint64_t write_index = 0;
};

static_assert(sizeof(AqlQueueControl) <= kMemoryPageSize);
static_assert(offsetof(AqlQueueControl, read_index) == 0);
static_assert(offsetof(AqlQueueControl, write_index) == 64);
static_assert(sizeof(uintptr_t) == sizeof(uint64_t));
static_assert(__atomic_always_lock_free(sizeof(uint16_t), nullptr));
static_assert(__atomic_always_lock_free(sizeof(uint64_t), nullptr));

AqlQueueStatus failure(AqlQueueError error, HSAKMT_STATUS status,
                       const std::string &operation) {
  return {error, static_cast<uint32_t>(status),
          operation + " failed with " +
              hsakmt_status_name(static_cast<uint32_t>(status)) + " (" +
              std::to_string(static_cast<uint32_t>(status)) + ")"};
}

AqlQueueStatus allocation_failure(AqlQueueError error,
                                  const MemoryStatus &status,
                                  const char *allocation_name) {
  return {error, status.hsakmt_status,
          std::string(allocation_name) +
              " allocation failed: " + status.message};
}

bool is_power_of_two(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

void fence_before_doorbell_store() {
#if defined(__x86_64__)
  // The doorbell mapping is write-combined/uncached. A C++ release store does
  // not emit the hardware store fence needed to order ordinary packet writes
  // before this MMIO write on x86.
  _mm_sfence();
#elif defined(__aarch64__)
  // Order normal-memory packet writes before the outer-shareable MMIO store.
  __asm__ __volatile__("dmb oshst" ::: "memory");
#else
#error "light-rocr needs an audited host MMIO store fence for this architecture"
#endif
}

void initialize_ring(MemoryAllocation &ring) {
  auto *bytes = static_cast<uint8_t *>(ring.host_address());
  std::memset(bytes, 0, static_cast<size_t>(ring.size()));
  for (uint64_t offset = 0; offset < ring.size(); offset += kAqlPacketSize) {
    ::new (static_cast<void *>(bytes + offset))
        runtime::AqlKernelDispatchPacket;
  }
}

} // namespace

struct AqlQueueState {
  uint64_t queue_id = 0;
  uintptr_t doorbell_address = 0;
  uint64_t ring_size = 0;
  bool active = false;
  MemoryAllocation ring;
  MemoryAllocation control;
};

const char *aql_submit_error_name(AqlSubmitError error) {
  switch (error) {
  case AqlSubmitError::None:
    return "none";
  case AqlSubmitError::InvalidQueue:
    return "invalid_queue";
  case AqlSubmitError::InvalidPacket:
    return "invalid_packet";
  case AqlSubmitError::QueueFull:
    return "queue_full";
  }
  return "unknown";
}

const char *aql_queue_error_name(AqlQueueError error) {
  switch (error) {
  case AqlQueueError::None:
    return "none";
  case AqlQueueError::InvalidSession:
    return "invalid_session";
  case AqlQueueError::InvalidRingSize:
    return "invalid_ring_size";
  case AqlQueueError::AllocateRing:
    return "allocate_ring";
  case AqlQueueError::AllocateControl:
    return "allocate_control";
  case AqlQueueError::CreateQueue:
    return "create_queue";
  case AqlQueueError::DestroyQueue:
    return "destroy_queue";
  case AqlQueueError::ReleaseControl:
    return "release_control";
  case AqlQueueError::ReleaseRing:
    return "release_ring";
  }
  return "unknown";
}

AqlQueueResult KfdSession::create_aql_queue(uint32_t gpu_node_id,
                                            uint64_t ring_size) const {
  if (state_ == nullptr) {
    return {{AqlQueueError::InvalidSession, 0, "KFD session is not open"}, {}};
  }
  if (ring_size < kAqlRingMinimumSize || ring_size > kAqlRingMaximumSize ||
      !is_power_of_two(ring_size) || ring_size % kMemoryPageSize != 0) {
    return {{AqlQueueError::InvalidRingSize, 0,
             "AQL ring size must be a power-of-two number of bytes from 4096 "
             "through 8388608"},
            {}};
  }

  auto ring = allocate(0, gpu_node_id, ring_size, MemoryKind::Gtt, true,
                       MemoryError::AllocateGtt, "AQL ring", true, true);
  if (!ring) {
    return {allocation_failure(AqlQueueError::AllocateRing, ring.status,
                               "AQL ring"),
            {}};
  }
  initialize_ring(ring.allocation);

  auto control = allocate_gtt(gpu_node_id, kMemoryPageSize);
  if (!control) {
    return {allocation_failure(AqlQueueError::AllocateControl, control.status,
                               "AQL queue control"),
            {}};
  }
  std::memset(control.allocation.host_address(), 0,
              static_cast<size_t>(control.allocation.size()));
  ::new (control.allocation.host_address()) AqlQueueControl;

  // Allocate all host bookkeeping before the KMT call. Once KMT creates the
  // queue, no throwing operation may occur before its handle is owned.
  auto queue_state = std::make_unique<AqlQueueState>();
  queue_state->ring_size = ring_size;
  queue_state->ring = std::move(ring.allocation);
  queue_state->control = std::move(control.allocation);

  HsaQueueResource resource{};
  resource.QueueRptrValue = queue_state->control.gpu_address() +
                            offsetof(AqlQueueControl, read_index);
  resource.QueueWptrValue = queue_state->control.gpu_address() +
                            offsetof(AqlQueueControl, write_index);

  const HSAKMT_STATUS status = hsaKmtCreateQueue(
      gpu_node_id, HSA_QUEUE_COMPUTE_AQL, 100, HSA_QUEUE_PRIORITY_NORMAL,
      reinterpret_cast<void *>(
          static_cast<uintptr_t>(queue_state->ring.gpu_address())),
      ring_size, nullptr, &resource);
  if (status != HSAKMT_STATUS_SUCCESS) {
    return {failure(AqlQueueError::CreateQueue, status, "hsaKmtCreateQueue"),
            {}};
  }

  queue_state->queue_id = resource.QueueId;
  queue_state->doorbell_address =
      reinterpret_cast<uintptr_t>(resource.Queue_DoorBell_aql);
  queue_state->active = true;
  return {{}, AqlQueue(std::move(queue_state))};
}

AqlQueue::AqlQueue(std::unique_ptr<AqlQueueState> state)
    : state_(std::move(state)) {}

AqlQueue::AqlQueue() = default;

AqlQueue::AqlQueue(AqlQueue &&other) noexcept
    : state_(std::move(other.state_)) {}

AqlQueue &AqlQueue::operator=(AqlQueue &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  release_for_destruction();
  state_ = std::move(other.state_);
  return *this;
}

AqlQueue::~AqlQueue() { release_for_destruction(); }

uint64_t AqlQueue::queue_id() const {
  return state_ != nullptr ? state_->queue_id : 0;
}

uintptr_t AqlQueue::doorbell_address() const {
  return state_ != nullptr ? state_->doorbell_address : 0;
}

void *AqlQueue::ring_host_address() const {
  return state_ != nullptr ? state_->ring.host_address() : nullptr;
}

uint64_t AqlQueue::ring_gpu_address() const {
  return state_ != nullptr ? state_->ring.gpu_address() : 0;
}

uint64_t AqlQueue::ring_size() const {
  return state_ != nullptr ? state_->ring_size : 0;
}

uint64_t AqlQueue::read_index_acquire() const {
  if (state_ == nullptr || state_->control.host_address() == nullptr) {
    return 0;
  }
  const auto *control =
      static_cast<const AqlQueueControl *>(state_->control.host_address());
  return __atomic_load_n(&control->read_index, __ATOMIC_ACQUIRE);
}

uint64_t AqlQueue::write_index_relaxed() const {
  if (state_ == nullptr || state_->control.host_address() == nullptr) {
    return 0;
  }
  const auto *control =
      static_cast<const AqlQueueControl *>(state_->control.host_address());
  return __atomic_load_n(&control->write_index, __ATOMIC_RELAXED);
}

AqlSubmitResult AqlQueue::submit_kernel_dispatch(
    const runtime::AqlKernelDispatchPacket &packet) {
  if (state_ == nullptr || !state_->active || state_->doorbell_address == 0 ||
      state_->ring.host_address() == nullptr ||
      state_->control.host_address() == nullptr) {
    return {AqlSubmitError::InvalidQueue, 0, 0, 0, "AQL queue is not active"};
  }
  const runtime::AqlPacketStatus packet_status =
      runtime::validate_kernel_dispatch_packet(packet);
  if (!packet_status) {
    return {AqlSubmitError::InvalidPacket, 0, read_index_acquire(),
            write_index_relaxed(), packet_status.message};
  }

  auto *control =
      static_cast<AqlQueueControl *>(state_->control.host_address());
  const uint64_t read_index =
      __atomic_load_n(&control->read_index, __ATOMIC_ACQUIRE);
  const uint64_t write_index =
      __atomic_load_n(&control->write_index, __ATOMIC_RELAXED);
  const uint64_t packet_count = state_->ring_size / kAqlPacketSize;
  if (write_index - read_index >= packet_count) {
    return {AqlSubmitError::QueueFull, write_index, read_index, write_index,
            "AQL queue has no free packet slot"};
  }

  const uint64_t slot_index = write_index & (packet_count - 1);
  auto *slot = static_cast<uint8_t *>(state_->ring.host_address()) +
               static_cast<size_t>(slot_index * kAqlPacketSize);
  auto *slot_header = reinterpret_cast<uint16_t *>(slot);
  __atomic_store_n(slot_header, runtime::kAqlPacketTypeInvalid,
                   __ATOMIC_RELAXED);
  const auto *packet_bytes = reinterpret_cast<const uint8_t *>(&packet);
  std::memcpy(slot + sizeof(packet.header),
              packet_bytes + sizeof(packet.header),
              sizeof(packet) - sizeof(packet.header));

  // Header publication makes the complete packet visible. The write index and
  // MMIO doorbell are updated only after that release point.
  __atomic_store_n(slot_header, packet.header, __ATOMIC_RELEASE);
  __atomic_store_n(&control->write_index, write_index + 1, __ATOMIC_RELEASE);
  fence_before_doorbell_store();
  auto *doorbell = reinterpret_cast<uint64_t *>(state_->doorbell_address);
  __atomic_store_n(doorbell, write_index, __ATOMIC_RELAXED);

  return {AqlSubmitError::None, write_index, read_index, write_index + 1, {}};
}

AqlQueue::operator bool() const { return state_ != nullptr && state_->active; }

AqlQueueStatus AqlQueue::release() {
  if (state_ == nullptr) {
    return {};
  }
  if (state_->active) {
    const HSAKMT_STATUS status = hsaKmtDestroyQueue(state_->queue_id);
    if (status != HSAKMT_STATUS_SUCCESS) {
      return failure(AqlQueueError::DestroyQueue, status, "hsaKmtDestroyQueue");
    }
    state_->active = false;
  }

  const MemoryStatus control_status = state_->control.release();
  if (!control_status) {
    return {AqlQueueError::ReleaseControl, control_status.hsakmt_status,
            "AQL queue control cleanup failed: " + control_status.message};
  }
  const MemoryStatus ring_status = state_->ring.release();
  if (!ring_status) {
    return {AqlQueueError::ReleaseRing, ring_status.hsakmt_status,
            "AQL ring cleanup failed: " + ring_status.message};
  }

  state_.reset();
  return {};
}

void AqlQueue::release_for_destruction() noexcept {
  const AqlQueueStatus status = release();
  if (!status && state_ != nullptr && state_->active) {
    // Retaining the mappings is safer than freeing memory still referenced by
    // a queue that KMT failed to destroy. Process teardown remains the final
    // cleanup boundary for this exceptional path.
    (void)state_.release();
  }
}

} // namespace light_rocr::transport::hsakmt
