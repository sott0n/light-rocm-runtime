#include "light_rocr/runtime/executable_image.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace light_rocr::runtime {
namespace {

constexpr uint32_t kElfExecute = 1;
constexpr uint32_t kKnownProtectionFlags = 7;

ExecutableImageStatus failure(ExecutableImageError error,
                              const std::string &message) {
  return {error, message};
}

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t *result) {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool is_power_of_two(uint64_t value) {
  return value != 0 && (value & (value - 1U)) == 0;
}

bool image_offset(const loader::LoadPlan &plan, uint64_t virtual_address,
                  uint64_t size, uint64_t *offset) {
  if (size == 0 || virtual_address < plan.image_virtual_address) {
    return false;
  }
  const uint64_t candidate = virtual_address - plan.image_virtual_address;
  if (candidate > plan.image_size || size > plan.image_size - candidate) {
    return false;
  }
  *offset = candidate;
  return true;
}

bool file_range_is_valid(uint64_t offset, uint64_t size, size_t file_size) {
  const uint64_t bounded_file_size = static_cast<uint64_t>(file_size);
  return offset <= bounded_file_size && size <= bounded_file_size - offset;
}

template <typename Operation, typename Address, typename Size>
bool operations_are_ordered(const std::vector<Operation> &operations,
                            Address address, Size size) {
  uint64_t previous_end = 0;
  bool has_previous = false;
  for (const Operation &operation : operations) {
    const uint64_t operation_address = address(operation);
    const uint64_t operation_size = size(operation);
    uint64_t operation_end = 0;
    if (operation_size == 0 ||
        !checked_add(operation_address, operation_size, &operation_end)) {
      return false;
    }
    if (has_previous && operation_address < previous_end) {
      return false;
    }
    previous_end = operation_end;
    has_previous = true;
  }
  return true;
}

bool copy_and_zero_fill_ranges_overlap(const loader::LoadPlan &plan) {
  size_t copy_index = 0;
  size_t zero_fill_index = 0;
  while (copy_index < plan.copies.size() &&
         zero_fill_index < plan.zero_fills.size()) {
    const loader::LoadCopy &copy = plan.copies[copy_index];
    const loader::LoadZeroFill &zero_fill = plan.zero_fills[zero_fill_index];
    const uint64_t copy_end = copy.virtual_address + copy.size;
    const uint64_t zero_fill_end = zero_fill.virtual_address + zero_fill.size;
    if (copy.virtual_address < zero_fill_end &&
        zero_fill.virtual_address < copy_end) {
      return true;
    }
    if (copy_end <= zero_fill.virtual_address) {
      ++copy_index;
    } else {
      ++zero_fill_index;
    }
  }
  return false;
}

bool range_has_protection(const loader::LoadPlan &plan, uint64_t address,
                          uint64_t size, uint32_t required_flags) {
  auto protection = std::upper_bound(
      plan.protections.begin(), plan.protections.end(), address,
      [](uint64_t candidate, const loader::LoadProtection &operation) {
        return candidate < operation.virtual_address;
      });
  if (protection == plan.protections.begin()) {
    return false;
  }
  --protection;
  if ((protection->flags & required_flags) != required_flags ||
      address < protection->virtual_address) {
    return false;
  }
  const uint64_t offset = address - protection->virtual_address;
  return offset <= protection->size && size <= protection->size - offset;
}

bool range_is_file_backed(const loader::LoadPlan &plan, uint64_t address,
                          uint64_t size) {
  auto copy = std::upper_bound(
      plan.copies.begin(), plan.copies.end(), address,
      [](uint64_t candidate, const loader::LoadCopy &operation) {
        return candidate < operation.virtual_address;
      });
  if (copy == plan.copies.begin()) {
    return false;
  }
  --copy;
  if (address < copy->virtual_address) {
    return false;
  }
  const uint64_t offset = address - copy->virtual_address;
  return offset <= copy->size && size <= copy->size - offset;
}

bool add_signed_offset(uint64_t base, int64_t offset, uint64_t *result) {
  if (offset >= 0) {
    return checked_add(base, static_cast<uint64_t>(offset), result);
  }
  const uint64_t magnitude = static_cast<uint64_t>(-(offset + 1)) + 1U;
  if (base < magnitude) {
    return false;
  }
  *result = base - magnitude;
  return true;
}

ExecutableImageStatus validate_plan(const uint8_t *hsaco, size_t hsaco_size,
                                    const loader::CodeObject &code_object,
                                    uint64_t *allocation_size) {
  const loader::LoadPlan &plan = code_object.load_plan;
  if (hsaco == nullptr && hsaco_size != 0) {
    return failure(ExecutableImageError::InvalidArgument,
                   "HSACO data is null but its size is non-zero");
  }
  if (plan.image_size == 0) {
    return failure(ExecutableImageError::EmptyImage,
                   "load plan has an empty virtual image");
  }

  uint64_t image_end = 0;
  if (!checked_add(plan.image_virtual_address, plan.image_size, &image_end)) {
    return failure(ExecutableImageError::ImageSizeOverflow,
                   "load-plan virtual image range overflows uint64");
  }
  (void)image_end;
  if (!is_power_of_two(plan.alignment) ||
      plan.alignment > kExecutableImagePageSize ||
      plan.image_virtual_address % plan.alignment != 0) {
    return failure(
        ExecutableImageError::UnsupportedAlignment,
        "load-plan alignment must be a power of two no larger than 4096 "
        "bytes, and the image base must satisfy it");
  }
  if (plan.image_size >
      std::numeric_limits<uint64_t>::max() - (kExecutableImagePageSize - 1U)) {
    return failure(ExecutableImageError::ImageSizeOverflow,
                   "page-rounded executable image size overflows uint64");
  }
  *allocation_size = (plan.image_size + kExecutableImagePageSize - 1U) &
                     ~(kExecutableImagePageSize - 1U);
  if (*allocation_size > std::numeric_limits<size_t>::max()) {
    return failure(ExecutableImageError::ImageSizeOverflow,
                   "executable image does not fit in the host address space");
  }
  if (!plan.relocations.empty()) {
    return failure(
        ExecutableImageError::UnsupportedRelocations,
        "executable image materialization does not yet apply relocations");
  }

  if (!operations_are_ordered(
          plan.copies,
          [](const loader::LoadCopy &copy) { return copy.virtual_address; },
          [](const loader::LoadCopy &copy) { return copy.size; })) {
    return failure(ExecutableImageError::InvalidCopyOperation,
                   "load-plan copy operations contain empty, overlapping, "
                   "overflowing, or out-of-order ranges");
  }
  for (const loader::LoadCopy &copy : plan.copies) {
    uint64_t destination_offset = 0;
    if (!file_range_is_valid(copy.file_offset, copy.size, hsaco_size) ||
        !image_offset(plan, copy.virtual_address, copy.size,
                      &destination_offset)) {
      return failure(ExecutableImageError::InvalidCopyOperation,
                     "load-plan copy range is outside the HSACO or image");
    }
    (void)destination_offset;
  }

  if (!operations_are_ordered(
          plan.zero_fills,
          [](const loader::LoadZeroFill &zero_fill) {
            return zero_fill.virtual_address;
          },
          [](const loader::LoadZeroFill &zero_fill) {
            return zero_fill.size;
          })) {
    return failure(ExecutableImageError::InvalidZeroFillOperation,
                   "load-plan zero-fill operations contain empty, "
                   "overlapping, overflowing, or out-of-order ranges");
  }
  for (const loader::LoadZeroFill &zero_fill : plan.zero_fills) {
    uint64_t destination_offset = 0;
    if (!image_offset(plan, zero_fill.virtual_address, zero_fill.size,
                      &destination_offset)) {
      return failure(ExecutableImageError::InvalidZeroFillOperation,
                     "load-plan zero-fill range is outside the image");
    }
    (void)destination_offset;
  }
  if (copy_and_zero_fill_ranges_overlap(plan)) {
    return failure(ExecutableImageError::InvalidZeroFillOperation,
                   "load-plan copy and zero-fill ranges overlap");
  }

  if (!operations_are_ordered(
          plan.protections,
          [](const loader::LoadProtection &protection) {
            return protection.virtual_address;
          },
          [](const loader::LoadProtection &protection) {
            return protection.size;
          })) {
    return failure(ExecutableImageError::InvalidProtectionOperation,
                   "load-plan protection operations contain empty, "
                   "overlapping, overflowing, or out-of-order ranges");
  }
  for (const loader::LoadProtection &protection : plan.protections) {
    uint64_t destination_offset = 0;
    if ((protection.flags & ~kKnownProtectionFlags) != 0 ||
        !image_offset(plan, protection.virtual_address, protection.size,
                      &destination_offset)) {
      return failure(ExecutableImageError::InvalidProtectionOperation,
                     "load-plan protection range or flags are invalid");
    }
    (void)destination_offset;
  }
  for (const loader::LoadCopy &copy : plan.copies) {
    if (!range_has_protection(plan, copy.virtual_address, copy.size, 0)) {
      return failure(ExecutableImageError::InvalidProtectionOperation,
                     "load-plan copy range has no matching protection");
    }
  }
  for (const loader::LoadZeroFill &zero_fill : plan.zero_fills) {
    if (!range_has_protection(plan, zero_fill.virtual_address, zero_fill.size,
                              0)) {
      return failure(ExecutableImageError::InvalidProtectionOperation,
                     "load-plan zero-fill range has no matching protection");
    }
  }

  for (const loader::KernelInfo &kernel : code_object.kernels) {
    uint64_t descriptor_offset = 0;
    uint64_t expected_code_entry = 0;
    if (!image_offset(plan, kernel.descriptor_virtual_address,
                      kAmdhsaKernelDescriptorSize, &descriptor_offset) ||
        kernel.descriptor_virtual_address % kAmdhsaKernelDescriptorAlignment !=
            0 ||
        descriptor_offset % kAmdhsaKernelDescriptorAlignment != 0 ||
        !range_is_file_backed(plan, kernel.descriptor_virtual_address,
                              kAmdhsaKernelDescriptorSize) ||
        !add_signed_offset(kernel.descriptor_virtual_address,
                           kernel.code_entry_byte_offset,
                           &expected_code_entry) ||
        expected_code_entry != kernel.code_entry_virtual_address ||
        !range_is_file_backed(plan, kernel.code_entry_virtual_address, 1) ||
        !range_has_protection(plan, kernel.code_entry_virtual_address, 1,
                              kElfExecute)) {
      return failure(ExecutableImageError::InvalidKernelDescriptor,
                     "kernel descriptor or code entry is inconsistent with "
                     "the load plan");
    }
    (void)descriptor_offset;
  }
  return {};
}

} // namespace

ExecutableImageInfo::ExecutableImageInfo(ExecutableImageInfo &&other) noexcept
    : code_object_(std::move(other.code_object_)),
      allocation_size_(other.allocation_size_),
      gpu_address_(other.gpu_address_), kernels_(std::move(other.kernels_)),
      valid_(other.valid_) {
  other.reset();
}

ExecutableImageInfo &
ExecutableImageInfo::operator=(ExecutableImageInfo &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  code_object_ = std::move(other.code_object_);
  allocation_size_ = other.allocation_size_;
  gpu_address_ = other.gpu_address_;
  kernels_ = std::move(other.kernels_);
  valid_ = other.valid_;
  other.reset();
  return *this;
}

void ExecutableImageInfo::reset() {
  code_object_ = {};
  allocation_size_ = 0;
  gpu_address_ = 0;
  kernels_.clear();
  valid_ = false;
}

ExecutableImageRequirementsResult
executable_image_requirements(const uint8_t *hsaco, size_t hsaco_size,
                              const loader::CodeObject &code_object) {
  uint64_t allocation_size = 0;
  ExecutableImageStatus status =
      validate_plan(hsaco, hsaco_size, code_object, &allocation_size);
  return {std::move(status), allocation_size};
}

ExecutableAddressResult
translate_executable_address(const loader::LoadPlan &plan, uint64_t gpu_address,
                             uint64_t virtual_address, uint64_t size,
                             uint64_t alignment) {
  if (!is_power_of_two(alignment)) {
    return {failure(ExecutableImageError::InvalidVirtualAddress,
                    "requested GPU address alignment is not a power of two"),
            0};
  }
  uint64_t offset = 0;
  if (!image_offset(plan, virtual_address, size, &offset)) {
    return {failure(ExecutableImageError::InvalidVirtualAddress,
                    "virtual-address range is outside the executable image"),
            0};
  }
  uint64_t translated_address = 0;
  if (!checked_add(gpu_address, offset, &translated_address)) {
    return {failure(ExecutableImageError::GpuAddressOverflow,
                    "translated GPU address overflows uint64"),
            0};
  }
  if (translated_address % alignment != 0) {
    return {failure(ExecutableImageError::InvalidVirtualAddress,
                    "translated GPU address does not satisfy its alignment"),
            0};
  }
  return {{}, translated_address};
}

ExecutableAddressResult
ExecutableImageInfo::translate(uint64_t virtual_address, uint64_t size,
                               uint64_t alignment) const {
  if (!valid_) {
    return {failure(ExecutableImageError::InvalidArgument,
                    "executable image is not GPU-usable"),
            0};
  }
  return translate_executable_address(code_object_.load_plan, gpu_address_,
                                      virtual_address, size, alignment);
}

ExecutableImageMaterializationResult
materialize_executable_image(const uint8_t *hsaco, size_t hsaco_size,
                             const loader::CodeObject &code_object,
                             void *destination, uint64_t destination_size,
                             uint64_t gpu_address) {
  const ExecutableImageRequirementsResult requirements =
      executable_image_requirements(hsaco, hsaco_size, code_object);
  if (!requirements) {
    return {requirements.status, {}};
  }
  if (destination == nullptr ||
      destination_size < requirements.allocation_size) {
    return {failure(ExecutableImageError::InvalidArgument,
                    "executable image destination is null or too small"),
            {}};
  }
  if (gpu_address % kExecutableImagePageSize != 0 ||
      gpu_address % code_object.load_plan.alignment !=
          code_object.load_plan.image_virtual_address %
              code_object.load_plan.alignment) {
    return {failure(ExecutableImageError::MisalignedAllocation,
                    "executable mapping does not satisfy the load-plan "
                    "alignment"),
            {}};
  }

  uint64_t allocation_end = 0;
  if (!checked_add(gpu_address, requirements.allocation_size,
                   &allocation_end)) {
    return {failure(ExecutableImageError::GpuAddressOverflow,
                    "executable GPU-address range overflows uint64"),
            {}};
  }
  (void)allocation_end;

  std::vector<ResolvedKernel> kernels;
  kernels.reserve(code_object.kernels.size());
  for (const loader::KernelInfo &kernel : code_object.kernels) {
    const ExecutableAddressResult descriptor = translate_executable_address(
        code_object.load_plan, gpu_address, kernel.descriptor_virtual_address,
        kAmdhsaKernelDescriptorSize, kAmdhsaKernelDescriptorAlignment);
    const ExecutableAddressResult code_entry =
        translate_executable_address(code_object.load_plan, gpu_address,
                                     kernel.code_entry_virtual_address, 1, 1);
    if (!descriptor) {
      return {descriptor.status, {}};
    }
    if (!code_entry) {
      return {code_entry.status, {}};
    }
    kernels.push_back(
        ResolvedKernel{descriptor.gpu_address, code_entry.gpu_address});
  }

  auto *image = static_cast<uint8_t *>(destination);
  std::memset(image, 0, static_cast<size_t>(requirements.allocation_size));
  for (const loader::LoadCopy &copy : code_object.load_plan.copies) {
    const uint64_t destination_offset =
        copy.virtual_address - code_object.load_plan.image_virtual_address;
    std::memcpy(image + static_cast<size_t>(destination_offset),
                hsaco + static_cast<size_t>(copy.file_offset),
                static_cast<size_t>(copy.size));
  }
  for (const loader::LoadZeroFill &zero_fill :
       code_object.load_plan.zero_fills) {
    const uint64_t destination_offset =
        zero_fill.virtual_address - code_object.load_plan.image_virtual_address;
    std::memset(image + static_cast<size_t>(destination_offset), 0,
                static_cast<size_t>(zero_fill.size));
  }

  return {{},
          ExecutableImageInfo(code_object, requirements.allocation_size,
                              gpu_address, std::move(kernels))};
}

const char *executable_image_error_name(ExecutableImageError error) {
  switch (error) {
  case ExecutableImageError::None:
    return "none";
  case ExecutableImageError::InvalidArgument:
    return "invalid_argument";
  case ExecutableImageError::EmptyImage:
    return "empty_image";
  case ExecutableImageError::ImageSizeOverflow:
    return "image_size_overflow";
  case ExecutableImageError::UnsupportedAlignment:
    return "unsupported_alignment";
  case ExecutableImageError::UnsupportedRelocations:
    return "unsupported_relocations";
  case ExecutableImageError::InvalidCopyOperation:
    return "invalid_copy_operation";
  case ExecutableImageError::InvalidZeroFillOperation:
    return "invalid_zero_fill_operation";
  case ExecutableImageError::InvalidProtectionOperation:
    return "invalid_protection_operation";
  case ExecutableImageError::InvalidKernelDescriptor:
    return "invalid_kernel_descriptor";
  case ExecutableImageError::AllocationFailed:
    return "allocation_failed";
  case ExecutableImageError::MisalignedAllocation:
    return "misaligned_allocation";
  case ExecutableImageError::InvalidVirtualAddress:
    return "invalid_virtual_address";
  case ExecutableImageError::GpuAddressOverflow:
    return "gpu_address_overflow";
  }
  return "unknown";
}

} // namespace light_rocr::runtime
