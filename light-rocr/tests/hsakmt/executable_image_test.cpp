#include "light_rocr/transport/hsakmt/executable_image.hpp"

#include <hsakmt/hsakmt.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
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

std::vector<uint8_t> make_hsaco_bytes() {
  std::vector<uint8_t> bytes(0xa0);
  for (size_t index = 0; index < bytes.size(); ++index) {
    bytes[index] = static_cast<uint8_t>((index * 13U + 7U) & 0xffU);
  }
  return bytes;
}

light_rocr::loader::CodeObject make_code_object() {
  light_rocr::loader::CodeObject object;
  object.target_machine = light_rocr::loader::kAmdgpuMachineGfx1101;
  object.target_isa = "amdgcn-amd-amdhsa--gfx1101";
  object.has_metadata = true;
  object.load_plan.image_virtual_address = 0x1000;
  object.load_plan.image_size = 0x900;
  object.load_plan.alignment = 0x1000;
  object.load_plan.copies = {
      {0x10, 0x1040, 0x40},
      {0x60, 0x1800, 0x20},
  };
  object.load_plan.zero_fills = {
      {0x1080, 0x20},
      {0x1820, 0x20},
  };
  object.load_plan.protections = {
      {0x1040, 0x60, 4},
      {0x1800, 0x40, 5},
  };

  light_rocr::loader::KernelInfo kernel;
  kernel.name = "store";
  kernel.symbol_name = "store.kd";
  kernel.descriptor_virtual_address = 0x1040;
  kernel.code_entry_byte_offset = 0x7c0;
  kernel.code_entry_virtual_address = 0x1800;
  object.kernels.push_back(std::move(kernel));
  return object;
}

void materializes_copies_zero_fill_and_addresses(TestContext *context) {
  reset_fake();
  const std::vector<uint8_t> bytes = make_hsaco_bytes();
  const light_rocr::loader::CodeObject object = make_code_object();
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto loaded = light_rocr::transport::hsakmt::materialize_executable_image(
        opened.session, 7, bytes.data(), bytes.size(), object);
    context->expect(static_cast<bool>(loaded), loaded.status.message);
    context->expect(loaded.image.allocation_size() == 4096,
                    "image size was not rounded to one 4 KiB page");
    context->expect(loaded.image.image_virtual_address() == 0x1000 &&
                        loaded.image.image_size() == 0x900,
                    "image virtual span changed during materialization");
    context->expect(loaded.image.gpu_address() == kGpuAddress,
                    "image retained the wrong GPU base");
    context->expect(loaded.image.host_address() == fake.storage.data(),
                    "image retained the wrong host mapping");
    context->expect(loaded.image.runtime_image() &&
                        loaded.image.runtime_image().gpu_address() ==
                            kGpuAddress,
                    "transport did not publish the common runtime image");

    HsaMemFlags expected_flags{};
    expected_flags.ui32.NonPaged = 1;
    expected_flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    expected_flags.ui32.HostAccess = 1;
    expected_flags.ui32.ExecuteAccess = 1;
    expected_flags.ui32.NoNUMABind = 1;
    context->expect(fake.preferred_node == 0 &&
                        fake.allocation_flags.Value == expected_flags.Value,
                    "materializer did not request executable GTT");
    context->expect(std::equal(bytes.begin() + 0x10, bytes.begin() + 0x50,
                               fake.storage.begin() + 0x40),
                    "descriptor copy was not materialized at its image offset");
    context->expect(std::equal(bytes.begin() + 0x60, bytes.begin() + 0x80,
                               fake.storage.begin() + 0x800),
                    "code copy was not materialized at its image offset");
    context->expect(std::all_of(fake.storage.begin() + 0x80,
                                fake.storage.begin() + 0xa0,
                                [](uint8_t byte) { return byte == 0; }) &&
                        std::all_of(fake.storage.begin() + 0x820,
                                    fake.storage.begin() + 0x840,
                                    [](uint8_t byte) { return byte == 0; }) &&
                        fake.storage.back() == 0,
                    "BSS or page padding was not zero initialized");

    context->expect(loaded.image.kernels().size() == 1,
                    "resolved kernel was not retained");
    if (loaded.image.kernels().size() == 1) {
      context->expect(
          loaded.image.kernels()[0].descriptor_gpu_address == 0x200040 &&
              loaded.image.kernels()[0].code_entry_gpu_address == 0x200800,
          "kernel virtual addresses were translated incorrectly");
    }
    const auto translated = loaded.image.translate(0x1808, 8, 8);
    context->expect(translated && translated.gpu_address == 0x200808,
                    "general virtual-address translation failed");
    const auto outside = loaded.image.translate(0x1900, 1);
    context->expect(outside.status.error ==
                        light_rocr::transport::hsakmt::ExecutableImageError::
                            InvalidVirtualAddress,
                    "one-past-the-image address was accepted");
    const auto zero_size = loaded.image.translate(0x1800, 0);
    context->expect(zero_size.status.error ==
                        light_rocr::transport::hsakmt::ExecutableImageError::
                            InvalidVirtualAddress,
                    "zero-sized address range was accepted");
    const auto invalid_alignment = loaded.image.translate(0x1800, 1, 3);
    context->expect(invalid_alignment.status.error ==
                        light_rocr::transport::hsakmt::ExecutableImageError::
                            InvalidVirtualAddress,
                    "non-power-of-two address alignment was accepted");

    light_rocr::transport::hsakmt::ExecutableImage moved_image(
        std::move(loaded.image));
    context->expect(!loaded.image && loaded.image.image_size() == 0 &&
                        loaded.image.kernels().empty(),
                    "moved-from image retained published state");
    const auto released = moved_image.release();
    context->expect(static_cast<bool>(released), released.message);
    context->expect(!moved_image && moved_image.kernels().empty(),
                    "released image retained published state");
  }
  expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:7:4096",
                         "unmap", "free:4096", "release", "close"});
}

void relocations_are_rejected_before_allocation(TestContext *context) {
  reset_fake();
  const std::vector<uint8_t> bytes = make_hsaco_bytes();
  auto object = make_code_object();
  object.load_plan.relocations.push_back(
      {light_rocr::loader::RelocationEncoding::Rela, 0x1800, 3, 0, 0});
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto loaded =
        light_rocr::transport::hsakmt::materialize_executable_image(
            opened.session, 7, bytes.data(), bytes.size(), object);
    context->expect(loaded.status.error ==
                        light_rocr::transport::hsakmt::ExecutableImageError::
                            UnsupportedRelocations,
                    "non-empty relocation plan was accepted");
  }
  expect_calls(context, {"open", "acquire", "release", "close"});
}

void invalid_operations_are_rejected_before_allocation(TestContext *context) {
  const std::vector<uint8_t> bytes = make_hsaco_bytes();

  reset_fake();
  {
    auto object = make_code_object();
    object.load_plan.copies[0].file_offset = bytes.size();
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto loaded =
        light_rocr::transport::hsakmt::materialize_executable_image(
            opened.session, 7, bytes.data(), bytes.size(), object);
    context->expect(loaded.status.error ==
                        light_rocr::transport::hsakmt::ExecutableImageError::
                            InvalidCopyOperation,
                    "out-of-file copy was accepted");
  }
  expect_calls(context, {"open", "acquire", "release", "close"});

  reset_fake();
  {
    auto object = make_code_object();
    object.load_plan.zero_fills[0].virtual_address = 0x1070;
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto loaded =
        light_rocr::transport::hsakmt::materialize_executable_image(
            opened.session, 7, bytes.data(), bytes.size(), object);
    context->expect(loaded.status.error ==
                        light_rocr::transport::hsakmt::ExecutableImageError::
                            InvalidZeroFillOperation,
                    "copy/zero-fill overlap was accepted");
  }
  expect_calls(context, {"open", "acquire", "release", "close"});

  reset_fake();
  {
    auto object = make_code_object();
    object.load_plan.protections.clear();
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto loaded =
        light_rocr::transport::hsakmt::materialize_executable_image(
            opened.session, 7, bytes.data(), bytes.size(), object);
    context->expect(loaded.status.error ==
                        light_rocr::transport::hsakmt::ExecutableImageError::
                            InvalidProtectionOperation,
                    "unprotected copy ranges were accepted");
  }
  expect_calls(context, {"open", "acquire", "release", "close"});
}

void invalid_alignment_and_kernel_are_rejected(TestContext *context) {
  const std::vector<uint8_t> bytes = make_hsaco_bytes();

  reset_fake();
  {
    auto object = make_code_object();
    object.load_plan.alignment = 8192;
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto loaded =
        light_rocr::transport::hsakmt::materialize_executable_image(
            opened.session, 7, bytes.data(), bytes.size(), object);
    context->expect(loaded.status.error ==
                        light_rocr::transport::hsakmt::ExecutableImageError::
                            UnsupportedAlignment,
                    "alignment above the transport guarantee was accepted");
  }
  expect_calls(context, {"open", "acquire", "release", "close"});

  reset_fake();
  {
    auto object = make_code_object();
    object.kernels[0].descriptor_virtual_address = 0x1041;
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto loaded =
        light_rocr::transport::hsakmt::materialize_executable_image(
            opened.session, 7, bytes.data(), bytes.size(), object);
    context->expect(loaded.status.error ==
                        light_rocr::transport::hsakmt::ExecutableImageError::
                            InvalidKernelDescriptor,
                    "misaligned kernel descriptor was accepted");
  }
  expect_calls(context, {"open", "acquire", "release", "close"});
}

void allocation_failures_are_preserved(TestContext *context) {
  reset_fake();
  fake.allocate_status = HSAKMT_STATUS_NO_MEMORY;
  const std::vector<uint8_t> bytes = make_hsaco_bytes();
  const auto object = make_code_object();
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto loaded =
        light_rocr::transport::hsakmt::materialize_executable_image(
            opened.session, 7, bytes.data(), bytes.size(), object);
    context->expect(loaded.status.error ==
                            light_rocr::transport::hsakmt::
                                ExecutableImageError::AllocationFailed &&
                        loaded.status.memory_status.error ==
                            light_rocr::transport::hsakmt::MemoryError::
                                AllocateExecutableGtt &&
                        loaded.status.memory_status.hsakmt_status ==
                            static_cast<uint32_t>(HSAKMT_STATUS_NO_MEMORY),
                    "KMT allocation failure details were not preserved");
  }
  expect_calls(context,
               {"open", "acquire", "allocate:0:4096", "release", "close"});
}

void invalid_gpu_mapping_is_cleaned_up(TestContext *context) {
  reset_fake();
  fake.alternate_gpu_address = kGpuAddress + 1;
  const std::vector<uint8_t> bytes = make_hsaco_bytes();
  const auto object = make_code_object();
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto loaded =
        light_rocr::transport::hsakmt::materialize_executable_image(
            opened.session, 7, bytes.data(), bytes.size(), object);
    context->expect(loaded.status.error ==
                        light_rocr::transport::hsakmt::ExecutableImageError::
                            MisalignedAllocation,
                    "misaligned GPU mapping was accepted");
    context->expect(!loaded.image,
                    "failed materialization published an executable image");
  }
  expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:7:4096",
                         "unmap", "free:4096", "release", "close"});
}

void address_overflow_is_rejected_and_cleaned_up(TestContext *context) {
  const std::vector<uint8_t> bytes = make_hsaco_bytes();

  reset_fake();
  {
    auto object = make_code_object();
    object.load_plan.image_virtual_address =
        std::numeric_limits<uint64_t>::max() - 0x800;
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto loaded =
        light_rocr::transport::hsakmt::materialize_executable_image(
            opened.session, 7, bytes.data(), bytes.size(), object);
    context->expect(loaded.status.error ==
                        light_rocr::transport::hsakmt::ExecutableImageError::
                            ImageSizeOverflow,
                    "overflowing virtual image range was accepted");
  }
  expect_calls(context, {"open", "acquire", "release", "close"});

  reset_fake();
  fake.alternate_gpu_address =
      std::numeric_limits<uint64_t>::max() & ~uint64_t{0xfff};
  {
    const auto object = make_code_object();
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto loaded =
        light_rocr::transport::hsakmt::materialize_executable_image(
            opened.session, 7, bytes.data(), bytes.size(), object);
    context->expect(loaded.status.error ==
                        light_rocr::transport::hsakmt::ExecutableImageError::
                            GpuAddressOverflow,
                    "overflowing GPU allocation range was accepted");
    context->expect(!loaded.image,
                    "GPU overflow published an executable image");
  }
  expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:7:4096",
                         "unmap", "free:4096", "release", "close"});
}

void cleanup_failure_is_retryable(TestContext *context) {
  reset_fake();
  const std::vector<uint8_t> bytes = make_hsaco_bytes();
  const auto object = make_code_object();
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto loaded = light_rocr::transport::hsakmt::materialize_executable_image(
        opened.session, 7, bytes.data(), bytes.size(), object);
    context->expect(static_cast<bool>(loaded), loaded.status.message);
    fake.unmap_status = HSAKMT_STATUS_ERROR;
    auto released = loaded.image.release();
    context->expect(
        released.error ==
                light_rocr::transport::hsakmt::MemoryError::UnmapFromGpu &&
            static_cast<bool>(loaded.image),
        "failed image cleanup discarded retryable ownership");
    fake.unmap_status = HSAKMT_STATUS_SUCCESS;
    released = loaded.image.release();
    context->expect(static_cast<bool>(released) && !loaded.image,
                    "image cleanup retry did not succeed");
  }
  expect_calls(context, {"open", "acquire", "allocate:0:4096", "map:7:4096",
                         "unmap", "unmap", "free:4096", "release", "close"});
}

void free_failure_invalidates_gpu_state_and_is_retryable(TestContext *context) {
  reset_fake();
  const std::vector<uint8_t> bytes = make_hsaco_bytes();
  const auto object = make_code_object();
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto loaded = light_rocr::transport::hsakmt::materialize_executable_image(
        opened.session, 7, bytes.data(), bytes.size(), object);
    context->expect(static_cast<bool>(loaded), loaded.status.message);

    fake.free_status = HSAKMT_STATUS_ERROR;
    auto released = loaded.image.release();
    context->expect(released.error ==
                        light_rocr::transport::hsakmt::MemoryError::FreeMemory,
                    "image free failure was not reported");
    context->expect(!loaded.image && loaded.image.owns_allocation(),
                    "unmapped image did not separate usability from cleanup "
                    "ownership");
    context->expect(!loaded.image.runtime_image(),
                    "unmapped transport retained a usable runtime image");
    context->expect(loaded.image.host_address() == nullptr &&
                        loaded.image.gpu_address() == 0 &&
                        loaded.image.allocation_size() == 0 &&
                        loaded.image.image_size() == 0 &&
                        loaded.image.kernels().empty(),
                    "unmapped image retained published GPU state");
    const auto translated = loaded.image.translate(0x1800, 1);
    context->expect(translated.status.error ==
                        light_rocr::transport::hsakmt::ExecutableImageError::
                            InvalidArgument,
                    "unmapped image translated a stale GPU address");

    fake.free_status = HSAKMT_STATUS_SUCCESS;
    released = loaded.image.release();
    context->expect(static_cast<bool>(released) &&
                        !loaded.image.owns_allocation(),
                    "pending image allocation could not be freed on retry");
  }
  expect_calls(context,
               {"open", "acquire", "allocate:0:4096", "map:7:4096", "unmap",
                "free:4096", "free:4096", "release", "close"});
}

void large_load_plan_scales(TestContext *context) {
  reset_fake();
  constexpr size_t kOperationCount = 8192;
  std::vector<uint8_t> bytes(kOperationCount);
  light_rocr::loader::CodeObject object;
  object.load_plan.image_size = kOperationCount * 2U;
  object.load_plan.alignment = 4096;
  object.load_plan.copies.reserve(kOperationCount);
  object.load_plan.zero_fills.reserve(kOperationCount);
  object.load_plan.protections.reserve(kOperationCount);
  for (size_t index = 0; index < kOperationCount; ++index) {
    bytes[index] = static_cast<uint8_t>((index % 251U) + 1U);
    const uint64_t virtual_address = static_cast<uint64_t>(index) * 2U;
    object.load_plan.copies.push_back(
        {static_cast<uint64_t>(index), virtual_address, 1});
    object.load_plan.zero_fills.push_back({virtual_address + 1U, 1});
    object.load_plan.protections.push_back({virtual_address, 2, 4});
  }

  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto loaded = light_rocr::transport::hsakmt::materialize_executable_image(
        opened.session, 7, bytes.data(), bytes.size(), object);
    context->expect(static_cast<bool>(loaded), loaded.status.message);
    context->expect(loaded.image.allocation_size() == 16384,
                    "large image has the wrong page-rounded size");
    bool contents_match = true;
    for (size_t index = 0; index < kOperationCount; ++index) {
      contents_match = contents_match &&
                       fake.storage[index * 2U] == bytes[index] &&
                       fake.storage[index * 2U + 1U] == 0;
    }
    context->expect(contents_match,
                    "large image copy or zero-fill contents mismatch");
    const auto released = loaded.image.release();
    context->expect(static_cast<bool>(released), released.message);
  }
  expect_calls(context, {"open", "acquire", "allocate:0:16384", "map:7:16384",
                         "unmap", "free:16384", "release", "close"});
}

static_assert(!std::is_copy_constructible_v<
              light_rocr::transport::hsakmt::ExecutableImage>);
static_assert(
    !std::is_copy_assignable_v<light_rocr::transport::hsakmt::ExecutableImage>);
static_assert(std::is_nothrow_move_constructible_v<
              light_rocr::transport::hsakmt::ExecutableImage>);
static_assert(
    !std::is_move_assignable_v<light_rocr::transport::hsakmt::ExecutableImage>);

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
      {"materializes_copies_zero_fill_and_addresses",
       materializes_copies_zero_fill_and_addresses},
      {"relocations_are_rejected_before_allocation",
       relocations_are_rejected_before_allocation},
      {"invalid_operations_are_rejected_before_allocation",
       invalid_operations_are_rejected_before_allocation},
      {"invalid_alignment_and_kernel_are_rejected",
       invalid_alignment_and_kernel_are_rejected},
      {"allocation_failures_are_preserved", allocation_failures_are_preserved},
      {"invalid_gpu_mapping_is_cleaned_up", invalid_gpu_mapping_is_cleaned_up},
      {"address_overflow_is_rejected_and_cleaned_up",
       address_overflow_is_rejected_and_cleaned_up},
      {"cleanup_failure_is_retryable", cleanup_failure_is_retryable},
      {"free_failure_invalidates_gpu_state_and_is_retryable",
       free_failure_invalidates_gpu_state_and_is_retryable},
      {"large_load_plan_scales", large_load_plan_scales},
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
