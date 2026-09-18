#include "light_rocr/runtime/launch.hpp"

#include <cstring>
#include <limits>

namespace light_rocr::runtime {
namespace {

KernargBufferStatus kernarg_failure(KernargBufferError error,
                                    const char *message) {
  return {error, message};
}

bool is_power_of_two(uint64_t value) {
  return value != 0 && (value & (value - 1U)) == 0;
}

KernargBufferRequirementsResult
validate_kernarg_request(const loader::KernelInfo &kernel,
                         const void *arguments, size_t arguments_size) {
  if (!is_power_of_two(kernel.kernarg_alignment) ||
      kernel.kernarg_alignment < kAmdKernargMinimumAlignment) {
    return {kernarg_failure(KernargBufferError::InvalidKernelMetadata,
                            "kernel kernarg alignment is invalid"),
            0, 0};
  }
  if (arguments_size != 0 && arguments == nullptr) {
    return {
        kernarg_failure(KernargBufferError::InvalidArguments,
                        "kernarg bytes are null but their size is non-zero"),
        0, 0};
  }
  if (arguments_size > kernel.kernarg_size) {
    return {kernarg_failure(
                KernargBufferError::ArgumentSizeExceeded,
                "caller-provided kernarg bytes exceed the metadata-declared "
                "kernarg segment"),
            0, 0};
  }

  return {{}, kernel.kernarg_size, kernel.kernarg_alignment};
}

} // namespace

KernargBufferInfo::KernargBufferInfo(KernargBufferInfo &&other) noexcept
    : gpu_address_(other.gpu_address_), kernarg_size_(other.kernarg_size_),
      alignment_(other.alignment_), valid_(other.valid_) {
  other.reset();
}

KernargBufferInfo &
KernargBufferInfo::operator=(KernargBufferInfo &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  gpu_address_ = other.gpu_address_;
  kernarg_size_ = other.kernarg_size_;
  alignment_ = other.alignment_;
  valid_ = other.valid_;
  other.reset();
  return *this;
}

void KernargBufferInfo::reset() {
  gpu_address_ = 0;
  kernarg_size_ = 0;
  alignment_ = 0;
  valid_ = false;
}

KernargBufferRequirementsResult
kernarg_buffer_requirements(const loader::KernelInfo &kernel,
                            const void *arguments, size_t arguments_size) {
  return validate_kernarg_request(kernel, arguments, arguments_size);
}

KernargBufferMaterializationResult
materialize_kernarg_buffer(const loader::KernelInfo &kernel,
                           const void *arguments, size_t arguments_size,
                           void *destination, uint64_t destination_size,
                           uint64_t gpu_address) {
  const KernargBufferRequirementsResult requirements =
      validate_kernarg_request(kernel, arguments, arguments_size);
  if (!requirements) {
    return {requirements.status, {}};
  }

  if (requirements.storage_size == 0) {
    if (destination != nullptr || destination_size != 0 || gpu_address != 0) {
      return {kernarg_failure(
                  KernargBufferError::InvalidDestination,
                  "a zero-sized kernarg segment requires canonical null "
                  "storage"),
              {}};
    }
    return {{}, KernargBufferInfo(0, 0, requirements.alignment)};
  }
  if (destination == nullptr) {
    return {kernarg_failure(KernargBufferError::InvalidDestination,
                            "kernarg destination is null"),
            {}};
  }
  if (destination_size < requirements.storage_size) {
    return {kernarg_failure(KernargBufferError::DestinationTooSmall,
                            "kernarg destination is smaller than the "
                            "metadata-declared segment"),
            {}};
  }
  if (gpu_address == 0 ||
      gpu_address % static_cast<uint64_t>(requirements.alignment) != 0) {
    return {kernarg_failure(
                KernargBufferError::MisalignedGpuAddress,
                "kernarg GPU address does not satisfy the kernel alignment"),
            {}};
  }
  if (gpu_address >
      std::numeric_limits<uint64_t>::max() - requirements.storage_size) {
    return {kernarg_failure(KernargBufferError::GpuAddressOverflow,
                            "kernarg GPU segment range overflows uint64"),
            {}};
  }

  auto *destination_bytes = static_cast<unsigned char *>(destination);
  if (arguments_size != 0) {
    std::memmove(destination_bytes, arguments, arguments_size);
  }
  std::memset(destination_bytes + arguments_size, 0,
              static_cast<size_t>(requirements.storage_size) - arguments_size);
  return {{},
          KernargBufferInfo(gpu_address, kernel.kernarg_size,
                            requirements.alignment)};
}

const char *kernarg_buffer_error_name(KernargBufferError error) {
  switch (error) {
  case KernargBufferError::None:
    return "none";
  case KernargBufferError::InvalidKernelMetadata:
    return "invalid_kernel_metadata";
  case KernargBufferError::InvalidArguments:
    return "invalid_arguments";
  case KernargBufferError::ArgumentSizeExceeded:
    return "argument_size_exceeded";
  case KernargBufferError::InvalidDestination:
    return "invalid_destination";
  case KernargBufferError::DestinationTooSmall:
    return "destination_too_small";
  case KernargBufferError::UnsupportedAlignment:
    return "unsupported_alignment";
  case KernargBufferError::MisalignedGpuAddress:
    return "misaligned_gpu_address";
  case KernargBufferError::GpuAddressOverflow:
    return "gpu_address_overflow";
  case KernargBufferError::AllocationFailed:
    return "allocation_failed";
  }
  return "unknown";
}

} // namespace light_rocr::runtime
