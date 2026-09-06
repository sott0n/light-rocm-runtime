#include "light_rocr/runtime/executable_image.hpp"

#include <algorithm>
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

constexpr uint64_t kGpuAddress = 0x200000;

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

void materializes_without_a_transport(TestContext *context) {
  const std::vector<uint8_t> bytes = make_hsaco_bytes();
  const auto object = make_code_object();
  const auto requirements = light_rocr::runtime::executable_image_requirements(
      bytes.data(), bytes.size(), object);
  context->expect(requirements && requirements.allocation_size == 4096,
                  requirements.status.message);

  std::vector<uint8_t> destination(
      static_cast<size_t>(requirements.allocation_size), 0xa5);
  const auto materialized = light_rocr::runtime::materialize_executable_image(
      bytes.data(), bytes.size(), object, destination.data(),
      destination.size(), kGpuAddress);
  context->expect(static_cast<bool>(materialized), materialized.status.message);
  context->expect(
      std::memcmp(destination.data() + 0x40, bytes.data() + 0x10, 0x40) == 0 &&
          std::memcmp(destination.data() + 0x800, bytes.data() + 0x60, 0x20) ==
              0,
      "file-backed ranges were not copied at their image offsets");
  context->expect(
      std::all_of(destination.begin() + 0x80, destination.begin() + 0x800,
                  [](uint8_t byte) { return byte == 0; }) &&
          std::all_of(destination.begin() + 0x820, destination.end(),
                      [](uint8_t byte) { return byte == 0; }),
      "BSS, image gaps, or allocation padding were not zeroed");
  context->expect(
      materialized.image.kernels().size() == 1 &&
          materialized.image.kernels()[0].descriptor_gpu_address == 0x200040 &&
          materialized.image.kernels()[0].code_entry_gpu_address == 0x200800,
      "kernel GPU addresses were resolved incorrectly");

  const auto image_translation = materialized.image.translate(0x1808, 8, 8);
  context->expect(image_translation &&
                      image_translation.gpu_address == 0x200808,
                  "transport-independent image view did not translate an "
                  "address");

  const auto translated = light_rocr::runtime::translate_executable_address(
      object.load_plan, kGpuAddress, 0x1808, 8, 8);
  context->expect(translated && translated.gpu_address == 0x200808,
                  "general virtual-address translation failed");
}

void rejects_invalid_plan_before_writing(TestContext *context) {
  const std::vector<uint8_t> bytes = make_hsaco_bytes();
  auto object = make_code_object();
  object.load_plan.relocations.push_back(
      {light_rocr::loader::RelocationEncoding::Rela, 0x1800, 3, 0, 0});
  auto requirements = light_rocr::runtime::executable_image_requirements(
      bytes.data(), bytes.size(), object);
  context->expect(
      requirements.status.error ==
          light_rocr::runtime::ExecutableImageError::UnsupportedRelocations,
      "non-empty relocation plan was accepted");

  object = make_code_object();
  object.load_plan.copies[0].file_offset = bytes.size();
  std::vector<uint8_t> destination(4096, 0xa5);
  const auto materialized = light_rocr::runtime::materialize_executable_image(
      bytes.data(), bytes.size(), object, destination.data(),
      destination.size(), kGpuAddress);
  context->expect(
      materialized.status.error ==
          light_rocr::runtime::ExecutableImageError::InvalidCopyOperation,
      "out-of-file copy was accepted");
  context->expect(std::all_of(destination.begin(), destination.end(),
                              [](uint8_t byte) { return byte == 0xa5; }),
                  "destination changed before invalid-plan rejection");
}

void rejects_invalid_destination_and_mapping(TestContext *context) {
  const std::vector<uint8_t> bytes = make_hsaco_bytes();
  const auto object = make_code_object();
  std::vector<uint8_t> destination(4096, 0xa5);

  auto materialized = light_rocr::runtime::materialize_executable_image(
      bytes.data(), bytes.size(), object, nullptr, destination.size(),
      kGpuAddress);
  context->expect(
      materialized.status.error ==
          light_rocr::runtime::ExecutableImageError::InvalidArgument,
      "null destination was accepted");

  materialized = light_rocr::runtime::materialize_executable_image(
      bytes.data(), bytes.size(), object, destination.data(),
      destination.size() - 1U, kGpuAddress);
  context->expect(
      materialized.status.error ==
          light_rocr::runtime::ExecutableImageError::InvalidArgument,
      "undersized destination was accepted");

  materialized = light_rocr::runtime::materialize_executable_image(
      bytes.data(), bytes.size(), object, destination.data(),
      destination.size(), kGpuAddress + 1U);
  context->expect(
      materialized.status.error ==
          light_rocr::runtime::ExecutableImageError::MisalignedAllocation,
      "misaligned GPU mapping was accepted");

  const uint64_t overflowing_gpu_address =
      std::numeric_limits<uint64_t>::max() & ~uint64_t{0xfff};
  materialized = light_rocr::runtime::materialize_executable_image(
      bytes.data(), bytes.size(), object, destination.data(),
      destination.size(), overflowing_gpu_address);
  context->expect(
      materialized.status.error ==
          light_rocr::runtime::ExecutableImageError::GpuAddressOverflow,
      "overflowing GPU mapping was accepted");
  context->expect(std::all_of(destination.begin(), destination.end(),
                              [](uint8_t byte) { return byte == 0xa5; }),
                  "destination changed before invalid-mapping rejection");
}

void rejects_invalid_address_translation(TestContext *context) {
  const auto object = make_code_object();
  auto translated = light_rocr::runtime::translate_executable_address(
      object.load_plan, kGpuAddress, 0x1900, 1);
  context->expect(
      translated.status.error ==
          light_rocr::runtime::ExecutableImageError::InvalidVirtualAddress,
      "one-past-the-image address was accepted");

  translated = light_rocr::runtime::translate_executable_address(
      object.load_plan, kGpuAddress, 0x1800, 0);
  context->expect(
      translated.status.error ==
          light_rocr::runtime::ExecutableImageError::InvalidVirtualAddress,
      "zero-sized address range was accepted");

  translated = light_rocr::runtime::translate_executable_address(
      object.load_plan, kGpuAddress, 0x1800, 1, 3);
  context->expect(
      translated.status.error ==
          light_rocr::runtime::ExecutableImageError::InvalidVirtualAddress,
      "non-power-of-two address alignment was accepted");
}

void enum_names(TestContext *context) {
  context->expect(std::string(light_rocr::runtime::executable_image_error_name(
                      light_rocr::runtime::ExecutableImageError::
                          InvalidKernelDescriptor)) ==
                      "invalid_kernel_descriptor",
                  "unexpected executable-image error name");
}

static_assert(
    !std::is_copy_constructible_v<light_rocr::runtime::ExecutableImageInfo>);
static_assert(
    !std::is_copy_assignable_v<light_rocr::runtime::ExecutableImageInfo>);
static_assert(std::is_nothrow_move_constructible_v<
              light_rocr::runtime::ExecutableImageInfo>);
static_assert(std::is_nothrow_move_assignable_v<
              light_rocr::runtime::ExecutableImageInfo>);

} // namespace

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"materializes_without_a_transport", materializes_without_a_transport},
      {"rejects_invalid_plan_before_writing",
       rejects_invalid_plan_before_writing},
      {"rejects_invalid_destination_and_mapping",
       rejects_invalid_destination_and_mapping},
      {"rejects_invalid_address_translation",
       rejects_invalid_address_translation},
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
