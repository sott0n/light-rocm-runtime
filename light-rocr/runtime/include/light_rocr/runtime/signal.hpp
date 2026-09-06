#ifndef LIGHT_ROCR_RUNTIME_SIGNAL_HPP
#define LIGHT_ROCR_RUNTIME_SIGNAL_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace light_rocr::runtime {

inline constexpr int64_t kAmdSignalKindUser = 1;

// Hardware-visible AMD user-signal ABI. The completion-signal handle carried
// by an AQL packet is the GPU virtual address of this complete object, not the
// address of value alone.
struct alignas(64) AmdSignal {
  int64_t kind = 0;
  int64_t value = 0;
  uint64_t event_mailbox_ptr = 0;
  uint32_t event_id = 0;
  uint32_t reserved1 = 0;
  uint64_t start_ts = 0;
  uint64_t end_ts = 0;
  uint64_t queue_ptr = 0;
  uint32_t reserved3[2] = {};
};

static_assert(sizeof(AmdSignal) == 64);
static_assert(alignof(AmdSignal) == 64);
static_assert(offsetof(AmdSignal, kind) == 0);
static_assert(offsetof(AmdSignal, value) == 8);
static_assert(offsetof(AmdSignal, event_mailbox_ptr) == 16);
static_assert(offsetof(AmdSignal, event_id) == 24);
static_assert(offsetof(AmdSignal, start_ts) == 32);
static_assert(offsetof(AmdSignal, end_ts) == 40);
static_assert(offsetof(AmdSignal, queue_ptr) == 48);

enum class SignalWaitState {
  Satisfied,
  TimedOut,
};

struct SignalWaitResult {
  SignalWaitState state = SignalWaitState::TimedOut;
  int64_t observed_value = 0;

  explicit operator bool() const { return state == SignalWaitState::Satisfied; }
};

void initialize_user_signal(AmdSignal &signal, int64_t initial_value);
[[nodiscard]] int64_t signal_load_relaxed(const AmdSignal &signal);
[[nodiscard]] int64_t signal_load_acquire(const AmdSignal &signal);
void signal_store_relaxed(AmdSignal &signal, int64_t value);
void signal_store_release(AmdSignal &signal, int64_t value);
[[nodiscard]] SignalWaitResult
signal_wait_until_equal(const AmdSignal &signal, int64_t expected_value,
                        std::chrono::steady_clock::time_point deadline);

} // namespace light_rocr::runtime

#endif
