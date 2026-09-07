#include "light_rocr/runtime/signal.hpp"
#include "light_rocr/transport/hsakmt/signal.hpp"

#include <hsakmt/hsakmt.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using light_rocr::runtime::AmdSignal;
using light_rocr::transport::hsakmt::kMemoryPageSize;

static_assert(
    std::is_move_constructible_v<light_rocr::transport::hsakmt::UserSignal>);
static_assert(
    !std::is_move_assignable_v<light_rocr::transport::hsakmt::UserSignal>);

constexpr uint64_t kSignalGpuAddress = 0x500000;
alignas(4096) std::array<uint8_t, kMemoryPageSize> signal_memory;

struct FakeKmt {
  HSAKMT_STATUS open_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS acquire_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS allocate_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS map_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS unmap_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS free_status = HSAKMT_STATUS_SUCCESS;
  void *allocation_address = signal_memory.data();
  uint64_t alternate_gpu_address = kSignalGpuAddress;
  uint32_t preferred_node = 0;
  HsaMemFlags allocation_flags{};
  HsaMemMapFlags map_flags{};
  std::vector<std::string> calls;
};

FakeKmt fake;

void reset_fake() {
  fake = {};
  std::fill(signal_memory.begin(), signal_memory.end(), uint8_t{0xa5});
}

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

void expect_calls(TestContext *context,
                  const std::vector<std::string> &expected) {
  context->expect(fake.calls == expected, "unexpected KMT call sequence");
}

void successful_round_trip(TestContext *context) {
  reset_fake();
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto created = opened.session.create_user_signal(7, 3);
    context->expect(static_cast<bool>(created), created.status.message);
    context->expect(
        created.signal.host_address() ==
            reinterpret_cast<const AmdSignal *>(signal_memory.data()),
        "signal host address was not retained");
    context->expect(created.signal.gpu_handle() == kSignalGpuAddress,
                    "signal GPU handle was not retained");
    context->expect(created.signal.host_address()->kind ==
                        light_rocr::runtime::kAmdSignalKindUser,
                    "signal kind was not initialized as USER");
    context->expect(created.signal.load_relaxed() == 3,
                    "signal initial value was not retained");
    context->expect(std::all_of(signal_memory.begin() + 16,
                                signal_memory.begin() + sizeof(AmdSignal),
                                [](uint8_t value) { return value == 0; }),
                    "unused signal ABI fields were not zeroed");

    HsaMemFlags expected_allocation_flags{};
    expected_allocation_flags.ui32.NonPaged = 1;
    expected_allocation_flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    expected_allocation_flags.ui32.HostAccess = 1;
    expected_allocation_flags.ui32.NoNUMABind = 1;
    context->expect(fake.preferred_node == 0 &&
                        fake.allocation_flags.Value ==
                            expected_allocation_flags.Value,
                    "signal did not use the GTT allocation policy");
    HsaMemMapFlags expected_map_flags{};
    expected_map_flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    expected_map_flags.ui32.HostAccess = 1;
    context->expect(fake.map_flags.Value == expected_map_flags.Value,
                    "signal did not use the host-accessible map policy");

    created.signal.store_release(0);
    const auto waited =
        created.signal.wait_until_equal(0, std::chrono::steady_clock::now());
    context->expect(static_cast<bool>(waited),
                    "signal CPU atomic round trip failed");
    const auto released = created.signal.release();
    context->expect(static_cast<bool>(released), released.message);
    expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:7:4096",
                           "unmap", "free:4096"});
  }
  expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:7:4096",
                         "unmap", "free:4096", "release", "close"});
}

void invalid_session_does_not_allocate(TestContext *context) {
  reset_fake();
  light_rocr::transport::hsakmt::KfdSession session;
  const auto created = session.create_user_signal(1, 1);
  context->expect(!created, "invalid session unexpectedly created a signal");
  context->expect(
      created.status.error ==
          light_rocr::transport::hsakmt::UserSignalError::InvalidSession,
      "invalid session returned the wrong error");
  expect_calls(context, {});
}

void allocation_failure_is_reported(TestContext *context) {
  reset_fake();
  fake.allocate_status = HSAKMT_STATUS_NO_MEMORY;
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto created = opened.session.create_user_signal(2, 1);
    context->expect(!created, "allocation failure unexpectedly succeeded");
    context->expect(
        created.status.error ==
            light_rocr::transport::hsakmt::UserSignalError::AllocateStorage,
        "allocation failure returned the wrong error");
  }
  expect_calls(context,
               {"open", "acquire", "allocate:0:4096", "release", "close"});
}

void misaligned_host_storage_is_rejected(TestContext *context) {
  reset_fake();
  fake.allocation_address = signal_memory.data() + 1;
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto created = opened.session.create_user_signal(3, 1);
    context->expect(!created, "misaligned host storage was accepted");
    context->expect(
        created.status.error ==
            light_rocr::transport::hsakmt::UserSignalError::MisalignedStorage,
        "misaligned host storage returned the wrong error");
  }
  expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:3:4096",
                         "unmap", "free:4096", "release", "close"});
}

void misaligned_gpu_storage_is_rejected(TestContext *context) {
  reset_fake();
  fake.alternate_gpu_address = kSignalGpuAddress + 1;
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto created = opened.session.create_user_signal(4, 1);
    context->expect(!created, "misaligned GPU storage was accepted");
    context->expect(
        created.status.error ==
            light_rocr::transport::hsakmt::UserSignalError::MisalignedStorage,
        "misaligned GPU storage returned the wrong error");
  }
  expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:4:4096",
                         "unmap", "free:4096", "release", "close"});
}

void cleanup_failure_can_be_retried(TestContext *context) {
  reset_fake();
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto created = opened.session.create_user_signal(5, 1);
    fake.unmap_status = HSAKMT_STATUS_ERROR;
    auto released = created.signal.release();
    context->expect(
        released.error ==
            light_rocr::transport::hsakmt::UserSignalError::ReleaseStorage,
        "signal unmap failure was not reported");
    context->expect(static_cast<bool>(created.signal),
                    "failed cleanup discarded signal ownership");
    fake.unmap_status = HSAKMT_STATUS_SUCCESS;
    released = created.signal.release();
    context->expect(static_cast<bool>(released), released.message);
  }
  expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:5:4096",
                         "unmap", "unmap", "free:4096", "release", "close"});
}

void moved_signal_keeps_session_open(TestContext *context) {
  reset_fake();
  auto make_signal = [context] {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto created = opened.session.create_user_signal(6, 1);
    context->expect(static_cast<bool>(created), created.status.message);
    auto signal = std::move(created.signal);
    context->expect(!created.signal, "moved-from signal retained ownership");
    return signal;
  };
  auto signal = make_signal();
  context->expect(fake.calls.back() == "map:6:4096",
                  "KFD closed while the moved signal was live");
  const auto released = signal.release();
  context->expect(static_cast<bool>(released), released.message);
  expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:6:4096",
                         "unmap", "free:4096", "release", "close"});
}

void destructor_releases_storage(TestContext *context) {
  reset_fake();
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    {
      const auto created = opened.session.create_user_signal(8, 1);
      context->expect(static_cast<bool>(created), created.status.message);
    }
  }
  expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:8:4096",
                         "unmap", "free:4096", "release", "close"});
}

} // namespace

extern "C" HSAKMT_STATUS hsaKmtOpenKFD() {
  fake.calls.emplace_back("open");
  return fake.open_status;
}

extern "C" HSAKMT_STATUS hsaKmtCloseKFD() {
  fake.calls.emplace_back("close");
  return HSAKMT_STATUS_SUCCESS;
}

extern "C" HSAKMT_STATUS
hsaKmtAcquireSystemProperties(HsaSystemProperties *properties) {
  fake.calls.emplace_back("acquire");
  if (fake.acquire_status == HSAKMT_STATUS_SUCCESS) {
    properties->NumNodes = 2;
  }
  return fake.acquire_status;
}

extern "C" HSAKMT_STATUS hsaKmtReleaseSystemProperties() {
  fake.calls.emplace_back("release");
  return HSAKMT_STATUS_SUCCESS;
}

extern "C" HSAKMT_STATUS hsaKmtAllocMemory(HSAuint32 preferred_node,
                                           HSAuint64 size, HsaMemFlags flags,
                                           void **address) {
  fake.calls.push_back("allocate:" + std::to_string(preferred_node) + ":" +
                       std::to_string(size));
  fake.preferred_node = preferred_node;
  fake.allocation_flags = flags;
  if (fake.allocate_status == HSAKMT_STATUS_SUCCESS) {
    *address = fake.allocation_address;
  }
  return fake.allocate_status;
}

extern "C" HSAKMT_STATUS hsaKmtMapMemoryToGPUNodes(
    void *address, HSAuint64 size, HSAuint64 *alternate_gpu_address,
    HsaMemMapFlags flags, HSAuint64 node_count, HSAuint32 *nodes) {
  const uint32_t node = node_count == 1 ? nodes[0] : 0;
  fake.calls.push_back("map:" + std::to_string(node) + ":" +
                       std::to_string(size));
  fake.map_flags = flags;
  if (address != fake.allocation_address || node_count != 1) {
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }
  if (fake.map_status == HSAKMT_STATUS_SUCCESS) {
    *alternate_gpu_address = fake.alternate_gpu_address;
  }
  return fake.map_status;
}

extern "C" HSAKMT_STATUS hsaKmtMapMemoryToGPU(void *, HSAuint64, HSAuint64 *) {
  return HSAKMT_STATUS_NOT_IMPLEMENTED;
}

extern "C" HSAKMT_STATUS hsaKmtUnmapMemoryToGPU(void *address) {
  fake.calls.emplace_back("unmap");
  if (address != fake.allocation_address) {
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }
  return fake.unmap_status;
}

extern "C" HSAKMT_STATUS hsaKmtFreeMemory(void *address, HSAuint64 size) {
  fake.calls.push_back("free:" + std::to_string(size));
  if (address != fake.allocation_address) {
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }
  return fake.free_status;
}

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"successful_round_trip", successful_round_trip},
      {"invalid_session_does_not_allocate", invalid_session_does_not_allocate},
      {"allocation_failure_is_reported", allocation_failure_is_reported},
      {"misaligned_host_storage_is_rejected",
       misaligned_host_storage_is_rejected},
      {"misaligned_gpu_storage_is_rejected",
       misaligned_gpu_storage_is_rejected},
      {"cleanup_failure_can_be_retried", cleanup_failure_can_be_retried},
      {"moved_signal_keeps_session_open", moved_signal_keeps_session_open},
      {"destructor_releases_storage", destructor_releases_storage},
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
