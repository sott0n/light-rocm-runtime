#include "light_rocr/runtime/signal.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using light_rocr::runtime::AmdSignal;
using light_rocr::runtime::SignalWaitState;

struct TestContext {
  int failures = 0;

  void expect(bool condition, const std::string &message) {
    if (!condition) {
      ++failures;
      std::cerr << "  FAIL: " << message << '\n';
    }
  }
};

using TestFunction = std::function<void(TestContext *)>;

void abi_layout(TestContext *context) {
  alignas(64) std::array<uint8_t, sizeof(AmdSignal)> storage{};
  const auto address = reinterpret_cast<uintptr_t>(storage.data());
  context->expect(sizeof(AmdSignal) == 64, "signal ABI size is not 64 bytes");
  context->expect(alignof(AmdSignal) == 64,
                  "signal ABI alignment is not 64 bytes");
  context->expect(address % alignof(AmdSignal) == 0,
                  "aligned signal storage is misaligned");
  context->expect(offsetof(AmdSignal, value) == 8,
                  "signal value is not at ABI offset 8");
}

void initialization_zeros_abi_fields(TestContext *context) {
  AmdSignal signal;
  signal.kind = -1;
  signal.event_mailbox_ptr = 1;
  signal.event_id = 2;
  signal.reserved1 = 3;
  signal.start_ts = 4;
  signal.end_ts = 5;
  signal.queue_ptr = 6;
  signal.reserved3[0] = 7;
  signal.reserved3[1] = 8;

  light_rocr::runtime::initialize_user_signal(signal, 9);
  context->expect(signal.kind == light_rocr::runtime::kAmdSignalKindUser,
                  "signal kind was not initialized as USER");
  context->expect(light_rocr::runtime::signal_load_relaxed(signal) == 9,
                  "initial signal value was not stored");
  context->expect(signal.event_mailbox_ptr == 0 && signal.event_id == 0 &&
                      signal.reserved1 == 0 && signal.start_ts == 0 &&
                      signal.end_ts == 0 && signal.queue_ptr == 0 &&
                      signal.reserved3[0] == 0 && signal.reserved3[1] == 0,
                  "unused user-signal ABI fields were not zeroed");
}

void atomic_load_store(TestContext *context) {
  AmdSignal signal;
  light_rocr::runtime::initialize_user_signal(signal, 1);
  light_rocr::runtime::signal_store_relaxed(signal, 2);
  context->expect(light_rocr::runtime::signal_load_relaxed(signal) == 2,
                  "relaxed atomic round trip failed");
  light_rocr::runtime::signal_store_release(signal, -3);
  context->expect(light_rocr::runtime::signal_load_acquire(signal) == -3,
                  "release/acquire atomic round trip failed");
}

void wait_observes_release(TestContext *context) {
  AmdSignal signal;
  light_rocr::runtime::initialize_user_signal(signal, 1);
  int published_value = 0;
  std::thread producer([&signal, &published_value] {
    published_value = 42;
    light_rocr::runtime::signal_store_release(signal, 0);
  });
  const auto result = light_rocr::runtime::signal_wait_until_equal(
      signal, 0, std::chrono::steady_clock::now() + std::chrono::seconds(1));
  const int observed_published_value = result ? published_value : 0;
  producer.join();

  context->expect(static_cast<bool>(result),
                  "wait did not observe the satisfying value");
  context->expect(result.observed_value == 0,
                  "wait returned the wrong observed value");
  context->expect(observed_published_value == 42,
                  "acquire wait did not observe preceding host writes");
}

void wait_reports_timeout(TestContext *context) {
  AmdSignal signal;
  light_rocr::runtime::initialize_user_signal(signal, 7);
  const auto result = light_rocr::runtime::signal_wait_until_equal(
      signal, 0, std::chrono::steady_clock::now());
  context->expect(!result, "expired wait unexpectedly succeeded");
  context->expect(result.state == SignalWaitState::TimedOut,
                  "expired wait returned the wrong state");
  context->expect(result.observed_value == 7,
                  "timeout did not report the last observed value");
}

void expired_wait_can_already_be_satisfied(TestContext *context) {
  AmdSignal signal;
  light_rocr::runtime::initialize_user_signal(signal, 0);
  const auto result = light_rocr::runtime::signal_wait_until_equal(
      signal, 0, std::chrono::steady_clock::time_point::min());
  context->expect(static_cast<bool>(result),
                  "satisfied signal lost to an expired deadline");
}

} // namespace

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"abi_layout", abi_layout},
      {"initialization_zeros_abi_fields", initialization_zeros_abi_fields},
      {"atomic_load_store", atomic_load_store},
      {"wait_observes_release", wait_observes_release},
      {"wait_reports_timeout", wait_reports_timeout},
      {"expired_wait_can_already_be_satisfied",
       expired_wait_can_already_be_satisfied},
  };

  TestContext context;
  for (const auto &test : tests) {
    std::cout << "[ RUN      ] " << test.first << '\n';
    const int failures_before = context.failures;
    test.second(&context);
    if (context.failures == failures_before) {
      std::cout << "[       OK ] " << test.first << '\n';
    } else {
      std::cout << "[  FAILED  ] " << test.first << '\n';
    }
  }

  if (context.failures != 0) {
    std::cerr << context.failures << " assertion(s) failed\n";
    return 1;
  }
  std::cout << tests.size() << " test(s) passed\n";
  return 0;
}
