#include "light_rocr/runtime/launch.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t kKernargGpuAddress = 0x300000;

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

light_rocr::loader::KernelInfo make_kernel(uint32_t kernarg_size = 32) {
  light_rocr::loader::KernelInfo kernel;
  kernel.name = "test";
  kernel.symbol_name = "test.kd";
  kernel.descriptor_virtual_address = 0x1040;
  kernel.code_entry_byte_offset = 0x7c0;
  kernel.code_entry_virtual_address = 0x1800;
  kernel.kernarg_size = kernarg_size;
  kernel.metadata_kernarg_alignment = 8;
  kernel.kernarg_alignment = 16;
  kernel.group_segment_size = 64;
  kernel.private_segment_size = 0;
  kernel.wavefront_size = 32;
  return kernel;
}

struct KernargFixture {
  std::array<uint8_t, 16> arguments{};
  alignas(4096) std::array<uint8_t, 4096> storage{};

  light_rocr::runtime::KernargBufferMaterializationResult
  materialize(const light_rocr::loader::KernelInfo &kernel) {
    for (size_t index = 0; index < arguments.size(); ++index) {
      arguments[index] = static_cast<uint8_t>(index + 1U);
    }
    storage.fill(0xa5);
    return light_rocr::runtime::materialize_kernarg_buffer(
        kernel, arguments.data(), arguments.size(), storage.data(),
        storage.size(), kKernargGpuAddress);
  }
};

void materializes_argument_prefix_and_zero_tail(TestContext *context) {
  const auto kernel = make_kernel();
  KernargFixture fixture;
  const auto requirements = light_rocr::runtime::kernarg_buffer_requirements(
      kernel, fixture.arguments.data(), fixture.arguments.size());
  context->expect(requirements && requirements.storage_size == 32 &&
                      requirements.alignment == 16,
                  requirements.status.message);

  auto materialized = fixture.materialize(kernel);
  context->expect(static_cast<bool>(materialized), materialized.status.message);
  context->expect(materialized.buffer.gpu_address() == kKernargGpuAddress &&
                      materialized.buffer.kernarg_size() == 32 &&
                      materialized.buffer.alignment() == 16,
                  "kernarg buffer retained incorrect metadata");
  context->expect(std::equal(fixture.arguments.begin(), fixture.arguments.end(),
                             fixture.storage.begin()),
                  "caller-provided kernarg prefix was not copied");
  context->expect(
      std::all_of(fixture.storage.begin() + fixture.arguments.size(),
                  fixture.storage.begin() + kernel.kernarg_size,
                  [](uint8_t byte) { return byte == 0; }),
      "metadata-declared kernarg tail was not zero initialized");
  context->expect(std::all_of(fixture.storage.begin() + kernel.kernarg_size,
                              fixture.storage.end(),
                              [](uint8_t byte) { return byte == 0xa5; }),
                  "storage beyond the logical kernarg segment was modified");

  light_rocr::runtime::KernargBufferInfo moved(std::move(materialized.buffer));
  context->expect(!materialized.buffer && moved &&
                      moved.gpu_address() == kKernargGpuAddress,
                  "kernarg buffer move did not transfer validity");
}

void supports_overlapping_argument_source(TestContext *context) {
  const auto kernel = make_kernel();
  alignas(4096) std::array<uint8_t, 4096> storage{};
  for (size_t index = 0; index < 16; ++index) {
    storage[index + 8U] = static_cast<uint8_t>(index + 1U);
  }

  const auto materialized = light_rocr::runtime::materialize_kernarg_buffer(
      kernel, storage.data() + 8, 16, storage.data(), storage.size(),
      kKernargGpuAddress);
  context->expect(static_cast<bool>(materialized), materialized.status.message);
  bool prefix_matches = true;
  for (size_t index = 0; index < 16; ++index) {
    prefix_matches =
        prefix_matches && storage[index] == static_cast<uint8_t>(index + 1U);
  }
  context->expect(prefix_matches,
                  "overlapping kernarg source was not copied safely");
}

void supports_canonical_empty_kernarg(TestContext *context) {
  const auto kernel = make_kernel(0);
  const auto requirements =
      light_rocr::runtime::kernarg_buffer_requirements(kernel, nullptr, 0);
  context->expect(requirements && requirements.storage_size == 0,
                  requirements.status.message);
  auto materialized = light_rocr::runtime::materialize_kernarg_buffer(
      kernel, nullptr, 0, nullptr, 0, 0);
  context->expect(materialized && materialized.buffer &&
                      materialized.buffer.gpu_address() == 0 &&
                      materialized.buffer.kernarg_size() == 0,
                  materialized.status.message);

  std::array<uint8_t, 1> storage{};
  materialized = light_rocr::runtime::materialize_kernarg_buffer(
      kernel, nullptr, 0, storage.data(), storage.size(), 0);
  context->expect(
      materialized.status.error ==
          light_rocr::runtime::KernargBufferError::InvalidDestination,
      "non-canonical storage was accepted for an empty kernarg segment");
}

void rejects_invalid_requests_before_writing(TestContext *context) {
  const std::array<uint8_t, 33> arguments{};
  auto kernel = make_kernel();
  auto requirements =
      light_rocr::runtime::kernarg_buffer_requirements(kernel, nullptr, 1);
  context->expect(requirements.status.error ==
                      light_rocr::runtime::KernargBufferError::InvalidArguments,
                  "null non-empty kernarg bytes were accepted");

  requirements = light_rocr::runtime::kernarg_buffer_requirements(
      kernel, arguments.data(), arguments.size());
  context->expect(
      requirements.status.error ==
          light_rocr::runtime::KernargBufferError::ArgumentSizeExceeded,
      "oversized kernarg bytes were accepted");

  kernel.kernarg_alignment = 8;
  requirements = light_rocr::runtime::kernarg_buffer_requirements(
      kernel, arguments.data(), 1);
  context->expect(
      requirements.status.error ==
          light_rocr::runtime::KernargBufferError::InvalidKernelMetadata,
      "sub-minimum kernel alignment was accepted");

  kernel = make_kernel();
  kernel.kernarg_alignment = 8192;
  requirements = light_rocr::runtime::kernarg_buffer_requirements(
      kernel, arguments.data(), 1);
  context->expect(requirements && requirements.storage_size == 32 &&
                      requirements.alignment == 8192,
                  "transport-independent kernarg requirements rejected a "
                  "valid over-page alignment");

  kernel = make_kernel();
  alignas(4096) std::array<uint8_t, 4096> destination{};
  destination.fill(0xa5);
  auto materialized = light_rocr::runtime::materialize_kernarg_buffer(
      kernel, arguments.data(), 16, nullptr, destination.size(),
      kKernargGpuAddress);
  context->expect(
      materialized.status.error ==
          light_rocr::runtime::KernargBufferError::InvalidDestination,
      "null kernarg destination was accepted");

  materialized = light_rocr::runtime::materialize_kernarg_buffer(
      kernel, arguments.data(), 16, destination.data(),
      kernel.kernarg_size - 1U, kKernargGpuAddress);
  context->expect(
      materialized.status.error ==
          light_rocr::runtime::KernargBufferError::DestinationTooSmall,
      "undersized kernarg destination was accepted");

  materialized = light_rocr::runtime::materialize_kernarg_buffer(
      kernel, arguments.data(), 16, destination.data(), destination.size(),
      kKernargGpuAddress + 1U);
  context->expect(
      materialized.status.error ==
          light_rocr::runtime::KernargBufferError::MisalignedGpuAddress,
      "misaligned kernarg GPU address was accepted");

  const uint64_t overflowing_gpu_address =
      std::numeric_limits<uint64_t>::max() & ~uint64_t{0xf};
  materialized = light_rocr::runtime::materialize_kernarg_buffer(
      kernel, arguments.data(), 16, destination.data(), destination.size(),
      overflowing_gpu_address);
  context->expect(
      materialized.status.error ==
          light_rocr::runtime::KernargBufferError::GpuAddressOverflow,
      "overflowing kernarg GPU range was accepted");
  context->expect(std::all_of(destination.begin(), destination.end(),
                              [](uint8_t byte) { return byte == 0xa5; }),
                  "destination changed before invalid-request rejection");
}

void enum_names(TestContext *context) {
  context->expect(
      std::string(light_rocr::runtime::kernarg_buffer_error_name(
          light_rocr::runtime::KernargBufferError::ArgumentSizeExceeded)) ==
          "argument_size_exceeded",
      "unexpected kernarg error name");
}

static_assert(
    !std::is_copy_constructible_v<light_rocr::runtime::KernargBufferInfo>);
static_assert(
    !std::is_copy_assignable_v<light_rocr::runtime::KernargBufferInfo>);
static_assert(std::is_nothrow_move_constructible_v<
              light_rocr::runtime::KernargBufferInfo>);
static_assert(
    std::is_nothrow_move_assignable_v<light_rocr::runtime::KernargBufferInfo>);

} // namespace

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"materializes_argument_prefix_and_zero_tail",
       materializes_argument_prefix_and_zero_tail},
      {"supports_overlapping_argument_source",
       supports_overlapping_argument_source},
      {"supports_canonical_empty_kernarg", supports_canonical_empty_kernarg},
      {"rejects_invalid_requests_before_writing",
       rejects_invalid_requests_before_writing},
      {"enum_names", enum_names},
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
