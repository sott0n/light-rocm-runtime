#include "light_rocr/runtime/launch.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t kImageGpuAddress = 0x200000;
constexpr uint64_t kKernargGpuAddress = 0x300000;
constexpr uint64_t kSignalGpuAddress = 0x400000;

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
  object.kernels.push_back(make_kernel());
  return object;
}

struct ImageFixture {
  std::vector<uint8_t> hsaco = make_hsaco_bytes();
  light_rocr::loader::CodeObject code_object = make_code_object();
  alignas(4096) std::array<uint8_t, 4096> storage{};

  light_rocr::runtime::ExecutableImageMaterializationResult materialize() {
    return light_rocr::runtime::materialize_executable_image(
        hsaco.data(), hsaco.size(), code_object, storage.data(), storage.size(),
        kImageGpuAddress);
  }
};

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

void builds_launch_from_loaded_kernel_metadata(TestContext *context) {
  ImageFixture image_fixture;
  auto image = image_fixture.materialize();
  context->expect(static_cast<bool>(image), image.status.message);
  KernargFixture kernarg_fixture;
  auto kernarg =
      kernarg_fixture.materialize(image.image.code_object().kernels.front());
  context->expect(static_cast<bool>(kernarg), kernarg.status.message);

  light_rocr::runtime::KernelLaunchConfiguration configuration;
  configuration.dimensions = 2;
  configuration.workgroup_size_x = 8;
  configuration.workgroup_size_y = 4;
  configuration.grid_size_x = 16;
  configuration.grid_size_y = 8;
  configuration.dynamic_group_segment_size = 32;
  const auto launch = light_rocr::runtime::make_kernel_launch_packet(
      image.image, 0, kernarg.buffer, configuration, kSignalGpuAddress);
  context->expect(static_cast<bool>(launch), launch.status.message);
  context->expect(
      launch.packet.workgroup_size_x == 8 &&
          launch.packet.workgroup_size_y == 4 &&
          launch.packet.grid_size_x == 16 && launch.packet.grid_size_y == 8 &&
          launch.packet.private_segment_size == 0 &&
          launch.packet.group_segment_size == 96 &&
          launch.packet.kernel_object == kImageGpuAddress + 0x40 &&
          launch.packet.kernarg_address == kKernargGpuAddress &&
          launch.packet.completion_signal == kSignalGpuAddress,
      "launch packet did not combine configuration and loaded metadata");
}

void rejects_invalid_launch_inputs(TestContext *context) {
  ImageFixture fixture;
  auto image = fixture.materialize();
  KernargFixture kernarg_fixture;
  auto kernarg =
      kernarg_fixture.materialize(image.image.code_object().kernels.front());
  light_rocr::runtime::KernelLaunchConfiguration configuration;

  light_rocr::runtime::ExecutableImageInfo invalid_image;
  auto launch = light_rocr::runtime::make_kernel_launch_packet(
      invalid_image, 0, kernarg.buffer, configuration, kSignalGpuAddress);
  context->expect(
      launch.status.error ==
          light_rocr::runtime::KernelLaunchError::InvalidExecutableImage,
      "invalid executable image was accepted");

  launch = light_rocr::runtime::make_kernel_launch_packet(
      image.image, 1, kernarg.buffer, configuration, kSignalGpuAddress);
  context->expect(
      launch.status.error ==
          light_rocr::runtime::KernelLaunchError::InvalidKernelIndex,
      "out-of-range kernel index was accepted");

  light_rocr::runtime::KernargBufferInfo invalid_kernarg;
  launch = light_rocr::runtime::make_kernel_launch_packet(
      image.image, 0, invalid_kernarg, configuration, kSignalGpuAddress);
  context->expect(
      launch.status.error ==
          light_rocr::runtime::KernelLaunchError::InvalidKernargBuffer,
      "invalid kernarg buffer was accepted");

  const auto smaller_kernel = make_kernel(16);
  auto incompatible = kernarg_fixture.materialize(smaller_kernel);
  launch = light_rocr::runtime::make_kernel_launch_packet(
      image.image, 0, incompatible.buffer, configuration, kSignalGpuAddress);
  context->expect(
      launch.status.error ==
          light_rocr::runtime::KernelLaunchError::IncompatibleKernargBuffer,
      "incompatible kernarg buffer was accepted");

  configuration.dimensions = 0;
  launch = light_rocr::runtime::make_kernel_launch_packet(
      image.image, 0, kernarg.buffer, configuration, kSignalGpuAddress);
  context->expect(
      launch.status.error ==
              light_rocr::runtime::KernelLaunchError::InvalidDispatchPacket &&
          launch.status.aql_status.error ==
              light_rocr::runtime::AqlPacketError::InvalidDimensions,
      "invalid geometry did not preserve its AQL validation detail");
}

void rejects_unsupported_kernel_requirements(TestContext *context) {
  ImageFixture private_fixture;
  private_fixture.code_object.kernels[0].private_segment_size = 24;
  auto private_image = private_fixture.materialize();
  KernargFixture kernarg_fixture;
  auto private_kernarg = kernarg_fixture.materialize(
      private_image.image.code_object().kernels.front());
  light_rocr::runtime::KernelLaunchConfiguration configuration;
  auto launch = light_rocr::runtime::make_kernel_launch_packet(
      private_image.image, 0, private_kernarg.buffer, configuration,
      kSignalGpuAddress);
  context->expect(
      launch.status.error ==
          light_rocr::runtime::KernelLaunchError::UnsupportedPrivateSegment,
      "private-segment kernel was accepted without scratch backing");

  ImageFixture dynamic_fixture;
  dynamic_fixture.code_object.kernels[0].uses_dynamic_stack = true;
  auto dynamic_image = dynamic_fixture.materialize();
  auto kernarg = kernarg_fixture.materialize(
      dynamic_image.image.code_object().kernels.front());
  launch = light_rocr::runtime::make_kernel_launch_packet(
      dynamic_image.image, 0, kernarg.buffer, configuration, kSignalGpuAddress);
  context->expect(
      launch.status.error ==
          light_rocr::runtime::KernelLaunchError::UnsupportedDynamicStack,
      "dynamic-stack kernel was accepted");

  ImageFixture lds_fixture;
  auto lds_image = lds_fixture.materialize();
  auto lds_kernarg = kernarg_fixture.materialize(
      lds_image.image.code_object().kernels.front());
  configuration.dynamic_group_segment_size =
      light_rocr::runtime::kGfx1101GroupSegmentMaximumSize -
      lds_image.image.code_object().kernels.front().group_segment_size;
  launch = light_rocr::runtime::make_kernel_launch_packet(
      lds_image.image, 0, lds_kernarg.buffer, configuration, kSignalGpuAddress);
  context->expect(static_cast<bool>(launch),
                  "maximum valid gfx1101 LDS allocation was rejected");

  ++configuration.dynamic_group_segment_size;
  launch = light_rocr::runtime::make_kernel_launch_packet(
      lds_image.image, 0, lds_kernarg.buffer, configuration, kSignalGpuAddress);
  context->expect(
      launch.status.error ==
              light_rocr::runtime::KernelLaunchError::InvalidDispatchPacket &&
          launch.status.aql_status.error ==
              light_rocr::runtime::AqlPacketError::InvalidGroupSegmentSize,
      "oversized gfx1101 LDS allocation was accepted");

  ImageFixture overflow_fixture;
  overflow_fixture.code_object.kernels[0].group_segment_size =
      std::numeric_limits<uint32_t>::max();
  auto overflow_image = overflow_fixture.materialize();
  auto overflow_kernarg = kernarg_fixture.materialize(
      overflow_image.image.code_object().kernels.front());
  configuration.dynamic_group_segment_size = 1;
  launch = light_rocr::runtime::make_kernel_launch_packet(
      overflow_image.image, 0, overflow_kernarg.buffer, configuration,
      kSignalGpuAddress);
  context->expect(
      launch.status.error ==
          light_rocr::runtime::KernelLaunchError::GroupSegmentSizeOverflow,
      "overflowing group segment size was accepted");
}

void enum_names(TestContext *context) {
  context->expect(
      std::string(light_rocr::runtime::kernarg_buffer_error_name(
          light_rocr::runtime::KernargBufferError::ArgumentSizeExceeded)) ==
          "argument_size_exceeded",
      "unexpected kernarg error name");
  context->expect(
      std::string(light_rocr::runtime::kernel_launch_error_name(
          light_rocr::runtime::KernelLaunchError::UnsupportedPrivateSegment)) ==
          "unsupported_private_segment",
      "unexpected launch error name");
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
      {"builds_launch_from_loaded_kernel_metadata",
       builds_launch_from_loaded_kernel_metadata},
      {"rejects_invalid_launch_inputs", rejects_invalid_launch_inputs},
      {"rejects_unsupported_kernel_requirements",
       rejects_unsupported_kernel_requirements},
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
