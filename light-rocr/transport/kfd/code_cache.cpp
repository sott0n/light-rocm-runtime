#include "light_rocr/transport/kfd/code_cache.hpp"

#include "light_rocr/arch/gfx11/code_cache.hpp"
#include "light_rocr/transport/kfd/executable_image.hpp"
#include "light_rocr/transport/kfd/signal.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <utility>

#if defined(__x86_64__)
#include <emmintrin.h>
#endif

namespace light_rocr::transport::kfd {

struct CodeCacheOperationState {
  std::optional<GttAllocation> command;
  std::optional<UserSignal> signal;
  AqlQueue *submitted_queue = nullptr;
  bool submitted = false;
};

namespace {

CodeCacheStatus failure(CodeCacheError error, std::string message,
                        int system_error = 0) {
  return {error, system_error, std::move(message)};
}

CodeCacheStatus merge_cleanup_failure(CodeCacheStatus primary,
                                      const CodeCacheStatus &cleanup) {
  if (cleanup) {
    return primary;
  }
  if (primary) {
    return cleanup;
  }
  primary.message += "; cleanup failed: " + cleanup.message;
  if (primary.system_error == 0) {
    primary.system_error = cleanup.system_error;
  }
  return primary;
}

void append_cleanup_failure(CodeCacheStatus *status,
                            const CodeCacheStatus &additional) {
  if (additional) {
    return;
  }
  if (*status) {
    *status = additional;
    return;
  }
  status->message += "; additional cleanup failure: " + additional.message;
  if (status->system_error == 0) {
    status->system_error = additional.system_error;
  }
}

CodeCacheStatus release_resources(CodeCacheOperationState *state) {
  CodeCacheStatus status;
  if (state->signal.has_value()) {
    const UserSignalStatus released = state->signal->release();
    if (!released) {
      append_cleanup_failure(&status,
                             failure(CodeCacheError::ReleaseSignal,
                                     released.message, released.system_error));
    } else {
      state->signal.reset();
    }
  }
  if (state->command.has_value()) {
    const MemoryStatus released = state->command->release();
    if (!released) {
      append_cleanup_failure(&status,
                             failure(CodeCacheError::ReleaseCommand,
                                     released.message, released.system_error));
    } else {
      state->command.reset();
    }
  }
  return status;
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

CodeCacheResult finish_before_submission(CodeCacheStatus status,
                                         CodeCacheOperation operation,
                                         AqlQueue &queue) {
  const CodeCacheStatus cleanup = operation.release(queue);
  return {merge_cleanup_failure(std::move(status), cleanup),
          std::move(operation)};
}

} // namespace

CodeCacheOperation::CodeCacheOperation() = default;

CodeCacheOperation::CodeCacheOperation(CodeCacheOperation &&other) noexcept
    : state_(std::move(other.state_)) {}

CodeCacheOperation::CodeCacheOperation(
    std::unique_ptr<CodeCacheOperationState> state)
    : state_(std::move(state)) {}

CodeCacheOperation::~CodeCacheOperation() { release_for_destruction(); }

bool CodeCacheOperation::owns_resources() const {
  return state_ != nullptr &&
         ((state_->command.has_value() &&
           static_cast<bool>(*state_->command)) ||
          (state_->signal.has_value() && static_cast<bool>(*state_->signal)));
}

CodeCacheStatus CodeCacheOperation::release(AqlQueue &queue) {
  if (state_ == nullptr) {
    return {};
  }
  if (state_->submitted) {
    if (state_->submitted_queue != &queue) {
      return failure(CodeCacheError::InvalidArgument,
                     "cache operation must be released with its submitted "
                     "AQL queue");
    }
    if (!state_->signal.has_value() || !*state_->signal) {
      return failure(CodeCacheError::InvalidArgument,
                     "submitted cache operation has no completion signal");
    }
    if (state_->signal->load_acquire() != 0) {
      const AqlQueueStatus queue_status = queue.release();
      if (!queue_status) {
        return failure(CodeCacheError::DestroyQueue, queue_status.message,
                       queue_status.system_error);
      }
    }
    state_->submitted = false;
    state_->submitted_queue = nullptr;
  }

  const CodeCacheStatus status = release_resources(state_.get());
  if (!owns_resources()) {
    state_.reset();
  }
  return status;
}

void CodeCacheOperation::release_for_destruction() noexcept {
  if (state_ == nullptr) {
    return;
  }
  if (state_->submitted && state_->signal.has_value() && *state_->signal &&
      state_->signal->load_acquire() != 0) {
    // The GPU may still dereference both resources. Preserve them until
    // process teardown rather than creating a use-after-free from a destructor.
    (void)state_.release();
    return;
  }
  state_->submitted = false;
  state_->submitted_queue = nullptr;
  CodeCacheStatus status = release_resources(state_.get());
  if (!status) {
    status = release_resources(state_.get());
  }
  if (owns_resources()) {
    (void)state_.release();
  } else {
    state_.reset();
  }
}

const char *code_cache_error_name(CodeCacheError error) {
  switch (error) {
  case CodeCacheError::None:
    return "none";
  case CodeCacheError::InvalidArgument:
    return "invalid_argument";
  case CodeCacheError::AllocateState:
    return "allocate_state";
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
  case CodeCacheError::DestroyQueue:
    return "destroy_queue";
  case CodeCacheError::ReleaseSignal:
    return "release_signal";
  case CodeCacheError::ReleaseCommand:
    return "release_command";
  }
  return "unknown";
}

CodeCacheResult
freeze_executable_image(const KfdSession &session, const runtime::Node &node,
                        AqlQueue &queue, const ExecutableImage &image,
                        std::chrono::steady_clock::time_point deadline,
                        const std::string &dri_root) {
  if (!session || !node.is_gpu() || node.gpu_id == 0 || !queue || !image ||
      !session.owns_aql_queue(queue, node) ||
      !session.owns_executable_image(image, node) ||
      image.host_address() == nullptr || image.allocation_size() == 0 ||
      image.allocation_size() > std::numeric_limits<size_t>::max() ||
      queue.ring_host_address() == nullptr || queue.packet_count() == 0 ||
      queue.doorbell_address() == 0) {
    return {failure(CodeCacheError::InvalidArgument,
                    "code-cache resources must be valid and belong to the "
                    "same KFD session and GPU node"),
            {}};
  }

  std::unique_ptr<CodeCacheOperationState> operation_state(
      new (std::nothrow) CodeCacheOperationState);
  if (operation_state == nullptr) {
    return {failure(CodeCacheError::AllocateState,
                    "failed to allocate code-cache operation state"),
            {}};
  }
  CodeCacheOperation operation(std::move(operation_state));

  auto command =
      session.allocate_executable_gtt(node, kMemoryPageSize, dri_root);
  if (command.allocation) {
    operation.state_->command.emplace(std::move(command.allocation));
  }
  if (!command) {
    return finish_before_submission(failure(CodeCacheError::AllocateCommand,
                                            command.status.message,
                                            command.status.system_error),
                                    std::move(operation), queue);
  }

  auto signal = session.create_user_signal(node, 1, dri_root);
  if (signal.signal) {
    operation.state_->signal.emplace(std::move(signal.signal));
  }
  if (!signal) {
    return finish_before_submission(failure(CodeCacheError::AllocateSignal,
                                            signal.status.message,
                                            signal.status.system_error),
                                    std::move(operation), queue);
  }

  const auto built = arch::gfx11::build_code_cache_invalidate(
      image.gpu_address(), image.allocation_size(),
      operation.state_->command->gpu_address(),
      operation.state_->signal->gpu_handle());
  if (!built) {
    return finish_before_submission(
        failure(CodeCacheError::InvalidArgument,
                std::string("invalid gfx11 cache command: ") +
                    arch::gfx11::code_cache_command_error_name(built.error)),
        std::move(operation), queue);
  }

  std::memcpy(operation.state_->command->host_address(), built.command.data(),
              built.command.size() * sizeof(uint32_t));
  flush_host_writes(image.host_address(), image.allocation_size());
  flush_host_writes(operation.state_->command->host_address(),
                    built.command.size() * sizeof(uint32_t));

  const uint64_t read_index = queue.read_index_acquire();
  const uint64_t write_index = queue.write_index_relaxed();
  if (write_index < read_index ||
      write_index - read_index >= queue.packet_count()) {
    return finish_before_submission(
        failure(CodeCacheError::QueueFull,
                "AQL queue has no free cache-invalidation slot"),
        std::move(operation), queue);
  }

  const AqlQueueIndexResult reserved = queue.add_write_index_scacq_screl(1);
  if (!reserved) {
    return finish_before_submission(
        failure(CodeCacheError::ReservePacket, reserved.status.message),
        std::move(operation), queue);
  }
  const uint64_t slot_index =
      reserved.previous_index & (queue.packet_count() - 1U);
  auto *slot = static_cast<uint8_t *>(queue.ring_host_address()) +
               slot_index * sizeof(built.packet);
  publish_packet(slot, built.packet);
  // Once the packet is published, a later producer can make it visible with
  // its doorbell store even if this producer's store fails. Treat both backing
  // allocations as GPU-visible from this point onward.
  operation.state_->submitted = true;
  operation.state_->submitted_queue = &queue;

  const AqlQueuePrimitiveStatus doorbell =
      queue.store_doorbell_screlease(reserved.previous_index);
  if (!doorbell) {
    return finish_before_submission(
        failure(CodeCacheError::RingDoorbell, doorbell.message),
        std::move(operation), queue);
  }

  const auto waited = operation.state_->signal->wait_until_equal(0, deadline);
  CodeCacheStatus status;
  if (!waited) {
    status = failure(CodeCacheError::WaitForCompletion,
                     "cache-invalidation packet did not complete before the "
                     "deadline");
  }
  const CodeCacheStatus cleanup = operation.release(queue);
  return {merge_cleanup_failure(std::move(status), cleanup),
          std::move(operation)};
}

} // namespace light_rocr::transport::kfd
