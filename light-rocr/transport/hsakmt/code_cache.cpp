#include "light_rocr/transport/hsakmt/code_cache.hpp"

#include "light_rocr/arch/gfx11/code_cache.hpp"
#include "light_rocr/transport/hsakmt/executable_image.hpp"
#include "light_rocr/transport/hsakmt/signal.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#if defined(__x86_64__)
#include <emmintrin.h>
#endif

namespace light_rocr::transport::hsakmt {
namespace {

CodeCacheStatus failure(CodeCacheError error, std::string message,
                        uint32_t hsakmt_status = 0) {
  return {error, hsakmt_status, std::move(message)};
}

void flush_host_writes(void *address, uint64_t size) {
  auto *last_byte = static_cast<volatile uint8_t *>(address) + size - 1U;
#if defined(__x86_64__)
  _mm_sfence();
  *last_byte = *last_byte;
  _mm_mfence();
#elif defined(__aarch64__)
  __asm__ __volatile__("dmb oshst" ::: "memory");
#else
#error "light-rocr needs an audited executable-memory flush for this host"
#endif
  const uint8_t readback = *last_byte;
  (void)readback;
}

void publish_packet(void *slot,
                    const arch::gfx11::AqlPm4IndirectBufferPacket &packet) {
  auto *bytes = static_cast<uint8_t *>(slot);
  auto *header = reinterpret_cast<uint32_t *>(bytes);
  __atomic_store_n(header,
                   static_cast<uint32_t>(arch::gfx11::kAqlPacketTypeInvalid),
                   __ATOMIC_RELAXED);
  const auto *packet_bytes = reinterpret_cast<const uint8_t *>(&packet);
  std::memcpy(bytes + sizeof(uint32_t), packet_bytes + sizeof(uint32_t),
              sizeof(packet) - sizeof(uint32_t));
  uint32_t published_header = 0;
  std::memcpy(&published_header, packet_bytes, sizeof(published_header));
  __atomic_store_n(header, published_header, __ATOMIC_RELEASE);
}

CodeCacheStatus cleanup(MemoryAllocation *command, UserSignal *signal,
                        CodeCacheStatus status) {
  const UserSignalStatus signal_status = signal->release();
  if (!signal_status && status) {
    status = failure(CodeCacheError::ReleaseSignal, signal_status.message,
                     signal_status.hsakmt_status);
  }
  const MemoryStatus command_status = command->release();
  if (!command_status && status) {
    status = failure(CodeCacheError::ReleaseCommand, command_status.message,
                     command_status.hsakmt_status);
  }
  return status;
}

} // namespace

const char *code_cache_error_name(CodeCacheError error) {
  switch (error) {
  case CodeCacheError::None:
    return "none";
  case CodeCacheError::InvalidArgument:
    return "invalid_argument";
  case CodeCacheError::AllocateCommand:
    return "allocate_command";
  case CodeCacheError::AllocateSignal:
    return "allocate_signal";
  case CodeCacheError::QueueFull:
    return "queue_full";
  case CodeCacheError::ReservePacket:
    return "reserve_packet";
  case CodeCacheError::RingDoorbell:
    return "ring_doorbell";
  case CodeCacheError::WaitForCompletion:
    return "wait_for_completion";
  case CodeCacheError::ReleaseSignal:
    return "release_signal";
  case CodeCacheError::ReleaseCommand:
    return "release_command";
  }
  return "unknown";
}

CodeCacheStatus freeze_executable_image(const KfdSession &session,
                                        uint32_t gpu_node_id, AqlQueue &queue,
                                        const ExecutableImage &image) {
  if (!session || !queue || !image || image.host_address() == nullptr ||
      image.allocation_size() == 0 ||
      image.allocation_size() > std::numeric_limits<size_t>::max() ||
      queue.ring_host_address() == nullptr || queue.packet_count() == 0 ||
      queue.doorbell_address() == 0) {
    return failure(CodeCacheError::InvalidArgument,
                   "invalid code-cache invalidation arguments");
  }

  auto command = session.allocate_executable_gtt(gpu_node_id, kMemoryPageSize);
  if (!command) {
    return failure(CodeCacheError::AllocateCommand, command.status.message,
                   command.status.hsakmt_status);
  }
  auto signal = session.create_user_signal(gpu_node_id, 1);
  if (!signal) {
    const MemoryStatus released = command.allocation.release();
    std::string message = signal.status.message;
    if (!released) {
      message += "; command cleanup failed: " + released.message;
    }
    return failure(CodeCacheError::AllocateSignal, std::move(message),
                   signal.status.hsakmt_status);
  }

  const auto built = arch::gfx11::build_code_cache_invalidate(
      image.gpu_address(), image.allocation_size(),
      command.allocation.gpu_address(), signal.signal.gpu_handle());
  if (!built) {
    return cleanup(
        &command.allocation, &signal.signal,
        failure(CodeCacheError::InvalidArgument,
                std::string("invalid gfx11 cache command: ") +
                    arch::gfx11::code_cache_command_error_name(built.error)));
  }

  std::memcpy(command.allocation.host_address(), built.command.data(),
              built.command.size() * sizeof(uint32_t));
  flush_host_writes(image.host_address(), image.allocation_size());
  flush_host_writes(command.allocation.host_address(),
                    built.command.size() * sizeof(uint32_t));

  const uint64_t read_index = queue.read_index_acquire();
  const uint64_t write_index = queue.write_index_relaxed();
  if (write_index < read_index ||
      write_index - read_index >= queue.packet_count()) {
    return cleanup(&command.allocation, &signal.signal,
                   failure(CodeCacheError::QueueFull,
                           "AQL queue has no free cache-invalidation slot"));
  }

  const AqlQueueIndexResult reserved = queue.add_write_index_scacq_screl(1);
  if (!reserved) {
    return cleanup(
        &command.allocation, &signal.signal,
        failure(CodeCacheError::ReservePacket, reserved.status.message));
  }
  const uint64_t slot_index =
      reserved.previous_index & (queue.packet_count() - 1U);
  auto *slot = static_cast<uint8_t *>(queue.ring_host_address()) +
               slot_index * sizeof(built.packet);
  publish_packet(slot, built.packet);

  const AqlQueuePrimitiveStatus doorbell =
      queue.store_doorbell_screlease(reserved.previous_index);
  if (!doorbell) {
    return cleanup(&command.allocation, &signal.signal,
                   failure(CodeCacheError::RingDoorbell, doorbell.message));
  }

  const auto waited = signal.signal.wait_until_equal(
      0, std::chrono::steady_clock::time_point::max());
  if (!waited) {
    return failure(CodeCacheError::WaitForCompletion,
                   "cache-invalidation packet did not complete");
  }
  return cleanup(&command.allocation, &signal.signal, {});
}

} // namespace light_rocr::transport::hsakmt
