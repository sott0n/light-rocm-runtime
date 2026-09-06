#include "light_rocr/transport/hsakmt/memory.hpp"

#include <hsakmt/hsakmt.h>

#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uintptr_t kCpuAddress = 0x100000;
constexpr uint64_t kAlternateGpuAddress = 0x200000;

struct FakeKmt {
  HSAKMT_STATUS open_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS acquire_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS allocate_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS map_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS unmap_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS free_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS close_status = HSAKMT_STATUS_SUCCESS;
  uint64_t alternate_gpu_address = kAlternateGpuAddress;
  std::vector<std::string> calls;
};

FakeKmt fake;

void reset_fake() { fake = {}; }

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
    context->expect(static_cast<bool>(opened), opened.status.message);
    auto allocated = opened.session.allocate_gtt(7, 8192);
    context->expect(static_cast<bool>(allocated), allocated.status.message);
    context->expect(allocated.allocation.cpu_address() ==
                        reinterpret_cast<void *>(kCpuAddress),
                    "CPU address was not retained");
    context->expect(allocated.allocation.gpu_address() == kAlternateGpuAddress,
                    "alternate GPU address was not retained");
    context->expect(allocated.allocation.size() == 8192,
                    "allocation size was not retained");
    const auto released = allocated.allocation.release();
    context->expect(static_cast<bool>(released), released.message);
    expect_calls(context, {"open", "acquire", "allocate:0:8192", "map:7:8192",
                           "unmap", "free:8192"});
  }
  expect_calls(context, {"open", "acquire", "allocate:0:8192", "map:7:8192",
                         "unmap", "free:8192", "release", "close"});
}

void identity_mapping_uses_cpu_address(TestContext *context) {
  reset_fake();
  fake.alternate_gpu_address = 0;
  auto opened = light_rocr::transport::hsakmt::KfdSession::open();
  auto allocated = opened.session.allocate_gtt(1, 4096);
  context->expect(allocated.allocation.gpu_address() == kCpuAddress,
                  "identity mapping did not use the CPU address");
}

void invalid_inputs_do_not_allocate(TestContext *context) {
  reset_fake();
  light_rocr::transport::hsakmt::KfdSession invalid;
  auto allocated = invalid.allocate_gtt(1, 4096);
  context->expect(
      allocated.status.error ==
          light_rocr::transport::hsakmt::MemoryError::InvalidSession,
      "invalid session was accepted");

  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    for (const uint64_t size : {0ULL, 4095ULL, 4097ULL}) {
      allocated = opened.session.allocate_gtt(1, size);
      context->expect(
          allocated.status.error ==
              light_rocr::transport::hsakmt::MemoryError::InvalidSize,
          "invalid size was accepted");
    }
    expect_calls(context, {"open", "acquire"});
  }
  expect_calls(context, {"open", "acquire", "release", "close"});
}

void open_failure_has_no_cleanup(TestContext *context) {
  reset_fake();
  fake.open_status = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;
  const auto opened = light_rocr::transport::hsakmt::KfdSession::open();
  context->expect(!opened, "open failure unexpectedly succeeded");
  context->expect(opened.status.error ==
                      light_rocr::transport::hsakmt::MemoryError::OpenKfd,
                  "wrong open failure");
  expect_calls(context, {"open"});
}

void acquire_failure_closes_kfd(TestContext *context) {
  reset_fake();
  fake.acquire_status = HSAKMT_STATUS_ERROR;
  const auto opened = light_rocr::transport::hsakmt::KfdSession::open();
  context->expect(!opened, "acquire failure unexpectedly succeeded");
  context->expect(
      opened.status.error ==
          light_rocr::transport::hsakmt::MemoryError::AcquireSystemProperties,
      "wrong acquire failure");
  expect_calls(context, {"open", "acquire", "close"});
}

void allocation_failure_closes_session(TestContext *context) {
  reset_fake();
  fake.allocate_status = HSAKMT_STATUS_NO_MEMORY;
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto allocated = opened.session.allocate_gtt(1, 4096);
    context->expect(!allocated, "allocation failure unexpectedly succeeded");
    context->expect(allocated.status.error ==
                        light_rocr::transport::hsakmt::MemoryError::AllocateGtt,
                    "wrong allocation failure");
  }
  expect_calls(context,
               {"open", "acquire", "allocate:0:4096", "release", "close"});
}

void map_failure_frees_allocation(TestContext *context) {
  reset_fake();
  fake.map_status = HSAKMT_STATUS_INVALID_NODE_UNIT;
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto allocated = opened.session.allocate_gtt(9, 4096);
    context->expect(!allocated, "map failure unexpectedly succeeded");
    context->expect(allocated.status.error ==
                        light_rocr::transport::hsakmt::MemoryError::MapToGpu,
                    "wrong map failure");
  }
  expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:9:4096",
                         "free:4096", "release", "close"});
}

void cleanup_failures_can_be_retried(TestContext *context) {
  reset_fake();
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto allocated = opened.session.allocate_gtt(1, 4096);
    fake.unmap_status = HSAKMT_STATUS_ERROR;
    auto released = allocated.allocation.release();
    context->expect(
        released.error ==
            light_rocr::transport::hsakmt::MemoryError::UnmapFromGpu,
        "unmap failure was not reported");
    fake.unmap_status = HSAKMT_STATUS_SUCCESS;
    fake.free_status = HSAKMT_STATUS_ERROR;
    released = allocated.allocation.release();
    context->expect(released.error ==
                        light_rocr::transport::hsakmt::MemoryError::FreeMemory,
                    "free failure was not reported");
    fake.free_status = HSAKMT_STATUS_SUCCESS;
    released = allocated.allocation.release();
    context->expect(static_cast<bool>(released), released.message);
  }
  expect_calls(context,
               {"open", "acquire", "allocate:0:4096", "map:1:4096", "unmap",
                "unmap", "free:4096", "free:4096", "release", "close"});
}

void allocation_keeps_kfd_open(TestContext *context) {
  reset_fake();
  light_rocr::transport::hsakmt::GttAllocation allocation;
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto allocated = opened.session.allocate_gtt(1, 4096);
    allocation = std::move(allocated.allocation);
  }
  context->expect(fake.calls.back() == "map:1:4096",
                  "KFD closed while an allocation was live");
  allocation = {};
  expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:1:4096",
                         "unmap", "free:4096", "release", "close"});
}

} // namespace

extern "C" HSAKMT_STATUS hsaKmtOpenKFD() {
  fake.calls.emplace_back("open");
  return fake.open_status;
}

extern "C" HSAKMT_STATUS hsaKmtCloseKFD() {
  fake.calls.emplace_back("close");
  return fake.close_status;
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
  if (flags.ui32.NonPaged != 1 || flags.ui32.HostAccess != 1 ||
      flags.ui32.PageSize != HSA_PAGE_SIZE_4KB || flags.ui32.NoNUMABind != 1) {
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }
  if (fake.allocate_status == HSAKMT_STATUS_SUCCESS) {
    *address = reinterpret_cast<void *>(kCpuAddress);
  }
  return fake.allocate_status;
}

extern "C" HSAKMT_STATUS hsaKmtMapMemoryToGPUNodes(
    void *address, HSAuint64 size, HSAuint64 *alternate_gpu_address,
    HsaMemMapFlags flags, HSAuint64 node_count, HSAuint32 *nodes) {
  const uint32_t node = node_count == 1 ? nodes[0] : 0;
  fake.calls.push_back("map:" + std::to_string(node) + ":" +
                       std::to_string(size));
  if (address != reinterpret_cast<void *>(kCpuAddress) || node_count != 1 ||
      flags.ui32.HostAccess != 1 || flags.ui32.PageSize != HSA_PAGE_SIZE_4KB) {
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }
  if (fake.map_status == HSAKMT_STATUS_SUCCESS) {
    *alternate_gpu_address = fake.alternate_gpu_address;
  }
  return fake.map_status;
}

extern "C" HSAKMT_STATUS hsaKmtUnmapMemoryToGPU(void *address) {
  fake.calls.emplace_back("unmap");
  if (address != reinterpret_cast<void *>(kCpuAddress)) {
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }
  return fake.unmap_status;
}

extern "C" HSAKMT_STATUS hsaKmtFreeMemory(void *address, HSAuint64 size) {
  fake.calls.push_back("free:" + std::to_string(size));
  if (address != reinterpret_cast<void *>(kCpuAddress)) {
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }
  return fake.free_status;
}

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"successful_round_trip", successful_round_trip},
      {"identity_mapping_uses_cpu_address", identity_mapping_uses_cpu_address},
      {"invalid_inputs_do_not_allocate", invalid_inputs_do_not_allocate},
      {"open_failure_has_no_cleanup", open_failure_has_no_cleanup},
      {"acquire_failure_closes_kfd", acquire_failure_closes_kfd},
      {"allocation_failure_closes_session", allocation_failure_closes_session},
      {"map_failure_frees_allocation", map_failure_frees_allocation},
      {"cleanup_failures_can_be_retried", cleanup_failures_can_be_retried},
      {"allocation_keeps_kfd_open", allocation_keeps_kfd_open},
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
