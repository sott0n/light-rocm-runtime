#include "light_rocr/transport/hsakmt/kernarg.hpp"

#include <hsakmt/hsakmt.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t kGpuAddress = 0x200000;

struct FakeKmt {
  HSAKMT_STATUS allocate_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS map_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS unmap_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS free_status = HSAKMT_STATUS_SUCCESS;
  uint64_t alternate_gpu_address = kGpuAddress;
  uint32_t preferred_node = 0;
  HsaMemFlags allocation_flags{};
  HsaMemMapFlags map_flags{};
  std::vector<uint8_t> storage;
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
  context->expect(fake.calls == expected, "unexpected KMT call order");
  if (fake.calls != expected) {
    std::cerr << "    expected:";
    for (const std::string &call : expected) {
      std::cerr << ' ' << call;
    }
    std::cerr << "\n    observed:";
    for (const std::string &call : fake.calls) {
      std::cerr << ' ' << call;
    }
    std::cerr << '\n';
  }
}

light_rocr::loader::KernelInfo make_kernel(uint32_t kernarg_size = 32) {
  light_rocr::loader::KernelInfo kernel;
  kernel.name = "test";
  kernel.symbol_name = "test.kd";
  kernel.kernarg_size = kernarg_size;
  kernel.metadata_kernarg_alignment = 8;
  kernel.kernarg_alignment = 16;
  return kernel;
}

void materializes_in_owned_gtt(TestContext *context) {
  reset_fake();
  std::array<uint8_t, 16> arguments{};
  for (size_t index = 0; index < arguments.size(); ++index) {
    arguments[index] = static_cast<uint8_t>(index + 1U);
  }

  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto created = light_rocr::transport::hsakmt::create_kernarg_buffer(
        opened.session, 7, make_kernel(), arguments.data(), arguments.size());
    context->expect(static_cast<bool>(created), created.status.message);
    context->expect(created.buffer && created.buffer.owns_allocation() &&
                        created.buffer.host_address() == fake.storage.data() &&
                        created.buffer.gpu_address() == kGpuAddress &&
                        created.buffer.allocation_size() == 4096 &&
                        created.buffer.kernarg_size() == 32 &&
                        created.buffer.alignment() == 16,
                    "transport retained incorrect kernarg properties");
    context->expect(created.buffer.runtime_buffer() &&
                        created.buffer.runtime_buffer().gpu_address() ==
                            kGpuAddress,
                    "transport did not publish the common kernarg buffer");
    context->expect(
        std::equal(arguments.begin(), arguments.end(), fake.storage.begin()),
        "kernarg prefix was not copied to GTT");
    context->expect(std::all_of(fake.storage.begin() + arguments.size(),
                                fake.storage.begin() + 32,
                                [](uint8_t byte) { return byte == 0; }),
                    "metadata-declared kernarg tail was not zero initialized");
    context->expect(
        std::all_of(fake.storage.begin() + 32, fake.storage.end(),
                    [](uint8_t byte) { return byte == 0xa5; }),
        "GTT backing beyond the logical kernarg segment was modified");

    HsaMemFlags expected_flags{};
    expected_flags.ui32.NonPaged = 1;
    expected_flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    expected_flags.ui32.HostAccess = 1;
    expected_flags.ui32.NoNUMABind = 1;
    context->expect(fake.preferred_node == 0 &&
                        fake.allocation_flags.Value == expected_flags.Value,
                    "kernarg buffer did not use ordinary GTT allocation");
    HsaMemMapFlags expected_map_flags{};
    expected_map_flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    expected_map_flags.ui32.HostAccess = 1;
    context->expect(fake.map_flags.Value == expected_map_flags.Value,
                    "kernarg buffer used unexpected GPU map flags");

    light_rocr::transport::hsakmt::KernargBuffer moved(
        std::move(created.buffer));
    context->expect(!created.buffer && moved,
                    "kernarg move retained source validity");
    const auto released = moved.release();
    context->expect(static_cast<bool>(released) && !moved &&
                        !moved.owns_allocation(),
                    "kernarg release did not clear ownership");
  }
  expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:7:4096",
                         "unmap", "free:4096", "release", "close"});
}

void invalid_request_does_not_allocate(TestContext *context) {
  reset_fake();
  std::array<uint8_t, 33> arguments{};
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto created = light_rocr::transport::hsakmt::create_kernarg_buffer(
        opened.session, 7, make_kernel(), arguments.data(), arguments.size());
    context->expect(
        created.status.error ==
            light_rocr::runtime::KernargBufferError::ArgumentSizeExceeded,
        "oversized kernarg did not preserve its validation error");
  }
  expect_calls(context, {"open", "acquire", "release", "close"});
}

void unsupported_transport_alignment_does_not_allocate(TestContext *context) {
  reset_fake();
  std::array<uint8_t, 16> arguments{};
  auto kernel = make_kernel();
  kernel.kernarg_alignment = 8192;
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto created = light_rocr::transport::hsakmt::create_kernarg_buffer(
        opened.session, 7, kernel, arguments.data(), arguments.size());
    context->expect(
        created.status.error ==
            light_rocr::runtime::KernargBufferError::UnsupportedAlignment,
        "over-page kernarg alignment was accepted by the KMT transport");
  }
  expect_calls(context, {"open", "acquire", "release", "close"});
}

void allocation_failure_is_preserved(TestContext *context) {
  reset_fake();
  fake.allocate_status = HSAKMT_STATUS_NO_MEMORY;
  std::array<uint8_t, 16> arguments{};
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto created = light_rocr::transport::hsakmt::create_kernarg_buffer(
        opened.session, 7, make_kernel(), arguments.data(), arguments.size());
    context->expect(
        created.status.error ==
                light_rocr::runtime::KernargBufferError::AllocationFailed &&
            created.status.memory_status.error ==
                light_rocr::transport::hsakmt::MemoryError::AllocateGtt &&
            created.status.memory_status.hsakmt_status ==
                static_cast<uint32_t>(HSAKMT_STATUS_NO_MEMORY),
        "kernarg allocation failure lost its KMT detail");
  }
  expect_calls(context,
               {"open", "acquire", "allocate:0:4096", "release", "close"});
}

void invalid_gpu_mapping_is_cleaned_up(TestContext *context) {
  reset_fake();
  fake.alternate_gpu_address = kGpuAddress + 1U;
  std::array<uint8_t, 16> arguments{};
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto created = light_rocr::transport::hsakmt::create_kernarg_buffer(
        opened.session, 7, make_kernel(), arguments.data(), arguments.size());
    context->expect(
        created.status.error ==
            light_rocr::runtime::KernargBufferError::MisalignedGpuAddress,
        "misaligned mapped GPU address was accepted");
  }
  expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:7:4096",
                         "unmap", "free:4096", "release", "close"});
}

void materialization_unmap_failure_returns_cleanup_ownership(
    TestContext *context) {
  reset_fake();
  fake.alternate_gpu_address = kGpuAddress + 1U;
  fake.unmap_status = HSAKMT_STATUS_ERROR;
  std::array<uint8_t, 16> arguments{};
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto created = light_rocr::transport::hsakmt::create_kernarg_buffer(
        opened.session, 7, make_kernel(), arguments.data(), arguments.size());
    context->expect(
        created.status.error ==
                light_rocr::runtime::KernargBufferError::MisalignedGpuAddress &&
            created.status.memory_status.error ==
                light_rocr::transport::hsakmt::MemoryError::UnmapFromGpu,
        "post-map validation lost its unmap cleanup failure");
    context->expect(!created.buffer && created.buffer.owns_allocation() &&
                        created.buffer.host_address() == nullptr &&
                        created.buffer.gpu_address() == 0 &&
                        created.buffer.allocation_size() == 0,
                    "failed unmap published invalid kernarg state or lost "
                    "cleanup ownership");

    fake.unmap_status = HSAKMT_STATUS_SUCCESS;
    const auto released = created.buffer.release();
    context->expect(static_cast<bool>(released) &&
                        !created.buffer.owns_allocation(),
                    "returned kernarg cleanup ownership was not retryable");
  }
  expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:7:4096",
                         "unmap", "unmap", "free:4096", "release", "close"});
}

void materialization_free_failure_returns_cleanup_ownership(
    TestContext *context) {
  reset_fake();
  fake.alternate_gpu_address = kGpuAddress + 1U;
  fake.free_status = HSAKMT_STATUS_ERROR;
  std::array<uint8_t, 16> arguments{};
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto created = light_rocr::transport::hsakmt::create_kernarg_buffer(
        opened.session, 7, make_kernel(), arguments.data(), arguments.size());
    context->expect(
        created.status.error ==
                light_rocr::runtime::KernargBufferError::MisalignedGpuAddress &&
            created.status.memory_status.error ==
                light_rocr::transport::hsakmt::MemoryError::FreeMemory,
        "post-map validation lost its free cleanup failure");
    context->expect(!created.buffer && created.buffer.owns_allocation() &&
                        created.buffer.host_address() == nullptr &&
                        created.buffer.gpu_address() == 0 &&
                        created.buffer.allocation_size() == 0,
                    "failed free published invalid kernarg state or lost "
                    "cleanup ownership");

    fake.free_status = HSAKMT_STATUS_SUCCESS;
    const auto released = created.buffer.release();
    context->expect(static_cast<bool>(released) &&
                        !created.buffer.owns_allocation(),
                    "returned kernarg free ownership was not retryable");
  }
  expect_calls(context,
               {"open", "acquire", "allocate:0:4096", "map:7:4096", "unmap",
                "free:4096", "free:4096", "release", "close"});
}

void empty_kernarg_owns_no_allocation(TestContext *context) {
  reset_fake();
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto created = light_rocr::transport::hsakmt::create_kernarg_buffer(
        opened.session, 7, make_kernel(0), nullptr, 0);
    context->expect(created && created.buffer &&
                        !created.buffer.owns_allocation() &&
                        created.buffer.host_address() == nullptr &&
                        created.buffer.gpu_address() == 0 &&
                        created.buffer.kernarg_size() == 0,
                    "empty kernarg did not use canonical allocation-free form");
    const auto released = created.buffer.release();
    context->expect(static_cast<bool>(released) && !created.buffer,
                    "empty kernarg release did not invalidate the view");
  }
  expect_calls(context, {"open", "acquire", "release", "close"});
}

void unmap_failure_is_retryable(TestContext *context) {
  reset_fake();
  std::array<uint8_t, 16> arguments{};
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto created = light_rocr::transport::hsakmt::create_kernarg_buffer(
        opened.session, 7, make_kernel(), arguments.data(), arguments.size());
    fake.unmap_status = HSAKMT_STATUS_ERROR;
    auto released = created.buffer.release();
    context->expect(
        released.error ==
                light_rocr::transport::hsakmt::MemoryError::UnmapFromGpu &&
            created.buffer && created.buffer.owns_allocation(),
        "failed unmap discarded usable kernarg state");
    fake.unmap_status = HSAKMT_STATUS_SUCCESS;
    released = created.buffer.release();
    context->expect(static_cast<bool>(released) && !created.buffer &&
                        !created.buffer.owns_allocation(),
                    "kernarg unmap retry did not complete cleanup");
  }
  expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:7:4096",
                         "unmap", "unmap", "free:4096", "release", "close"});
}

void free_failure_invalidates_gpu_state_and_is_retryable(TestContext *context) {
  reset_fake();
  std::array<uint8_t, 16> arguments{};
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto created = light_rocr::transport::hsakmt::create_kernarg_buffer(
        opened.session, 7, make_kernel(), arguments.data(), arguments.size());
    fake.free_status = HSAKMT_STATUS_ERROR;
    auto released = created.buffer.release();
    context->expect(released.error ==
                        light_rocr::transport::hsakmt::MemoryError::FreeMemory,
                    "kernarg free failure was not reported");
    context->expect(!created.buffer && created.buffer.owns_allocation() &&
                        !created.buffer.runtime_buffer() &&
                        created.buffer.host_address() == nullptr &&
                        created.buffer.gpu_address() == 0 &&
                        created.buffer.allocation_size() == 0 &&
                        created.buffer.kernarg_size() == 0,
                    "unmapped kernarg retained stale GPU state");

    fake.free_status = HSAKMT_STATUS_SUCCESS;
    released = created.buffer.release();
    context->expect(static_cast<bool>(released) &&
                        !created.buffer.owns_allocation(),
                    "pending kernarg allocation could not be freed on retry");
  }
  expect_calls(context,
               {"open", "acquire", "allocate:0:4096", "map:7:4096", "unmap",
                "free:4096", "free:4096", "release", "close"});
}

static_assert(!std::is_copy_constructible_v<
              light_rocr::transport::hsakmt::KernargBuffer>);
static_assert(
    !std::is_copy_assignable_v<light_rocr::transport::hsakmt::KernargBuffer>);
static_assert(std::is_nothrow_move_constructible_v<
              light_rocr::transport::hsakmt::KernargBuffer>);
static_assert(
    !std::is_move_assignable_v<light_rocr::transport::hsakmt::KernargBuffer>);

} // namespace

extern "C" HSAKMT_STATUS hsaKmtOpenKFD() {
  fake.calls.emplace_back("open");
  return HSAKMT_STATUS_SUCCESS;
}

extern "C" HSAKMT_STATUS hsaKmtCloseKFD() {
  fake.calls.emplace_back("close");
  return HSAKMT_STATUS_SUCCESS;
}

extern "C" HSAKMT_STATUS
hsaKmtAcquireSystemProperties(HsaSystemProperties *properties) {
  fake.calls.emplace_back("acquire");
  properties->NumNodes = 2;
  return HSAKMT_STATUS_SUCCESS;
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
    fake.storage.assign(static_cast<size_t>(size), 0xa5);
    *address = fake.storage.data();
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
  if (address != fake.storage.data() || node_count != 1) {
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
  if (address != fake.storage.data()) {
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }
  return fake.unmap_status;
}

extern "C" HSAKMT_STATUS hsaKmtFreeMemory(void *address, HSAuint64 size) {
  fake.calls.push_back("free:" + std::to_string(size));
  if (address != fake.storage.data()) {
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }
  return fake.free_status;
}

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"materializes_in_owned_gtt", materializes_in_owned_gtt},
      {"invalid_request_does_not_allocate", invalid_request_does_not_allocate},
      {"unsupported_transport_alignment_does_not_allocate",
       unsupported_transport_alignment_does_not_allocate},
      {"allocation_failure_is_preserved", allocation_failure_is_preserved},
      {"invalid_gpu_mapping_is_cleaned_up", invalid_gpu_mapping_is_cleaned_up},
      {"materialization_unmap_failure_returns_cleanup_ownership",
       materialization_unmap_failure_returns_cleanup_ownership},
      {"materialization_free_failure_returns_cleanup_ownership",
       materialization_free_failure_returns_cleanup_ownership},
      {"empty_kernarg_owns_no_allocation", empty_kernarg_owns_no_allocation},
      {"unmap_failure_is_retryable", unmap_failure_is_retryable},
      {"free_failure_invalidates_gpu_state_and_is_retryable",
       free_failure_invalidates_gpu_state_and_is_retryable},
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
