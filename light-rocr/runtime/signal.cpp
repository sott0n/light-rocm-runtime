#include "light_rocr/runtime/signal.hpp"

#include <thread>

namespace light_rocr::runtime {
namespace {

static_assert(sizeof(void *) == sizeof(uint64_t),
              "light-rocr requires the AMDGPU large machine model");
static_assert(__atomic_always_lock_free(sizeof(int64_t), nullptr),
              "hardware-visible signals require lock-free 64-bit atomics");

} // namespace

void initialize_user_signal(AmdSignal &signal, int64_t initial_value) {
  signal = {};
  signal.kind = kAmdSignalKindUser;
  signal_store_relaxed(signal, initial_value);
}

int64_t signal_load_relaxed(const AmdSignal &signal) {
  return __atomic_load_n(&signal.value, __ATOMIC_RELAXED);
}

int64_t signal_load_acquire(const AmdSignal &signal) {
  return __atomic_load_n(&signal.value, __ATOMIC_ACQUIRE);
}

void signal_store_relaxed(AmdSignal &signal, int64_t value) {
  __atomic_store_n(&signal.value, value, __ATOMIC_RELAXED);
}

void signal_store_release(AmdSignal &signal, int64_t value) {
  __atomic_store_n(&signal.value, value, __ATOMIC_RELEASE);
}

SignalWaitResult
signal_wait_until_equal(const AmdSignal &signal, int64_t expected_value,
                        std::chrono::steady_clock::time_point deadline) {
  for (;;) {
    const int64_t observed_value = signal_load_acquire(signal);
    if (observed_value == expected_value) {
      return {SignalWaitState::Satisfied, observed_value};
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return {SignalWaitState::TimedOut, observed_value};
    }
    std::this_thread::yield();
  }
}

} // namespace light_rocr::runtime
