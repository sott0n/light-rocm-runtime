#ifndef LIGHT_ROCR_RUNTIME_EXECUTABLE_IMAGE_HPP
#define LIGHT_ROCR_RUNTIME_EXECUTABLE_IMAGE_HPP

#include "light_rocr/loader/code_object.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace light_rocr::runtime {

inline constexpr uint64_t kExecutableImagePageSize = 4096;
inline constexpr uint64_t kAmdhsaKernelDescriptorSize = 64;
inline constexpr uint64_t kAmdhsaKernelDescriptorAlignment = 64;

enum class ExecutableImageError {
  None,
  InvalidArgument,
  EmptyImage,
  ImageSizeOverflow,
  UnsupportedAlignment,
  UnsupportedRelocations,
  InvalidCopyOperation,
  InvalidZeroFillOperation,
  InvalidProtectionOperation,
  InvalidKernelDescriptor,
  AllocationFailed,
  MisalignedAllocation,
  InvalidVirtualAddress,
  GpuAddressOverflow,
};

struct ExecutableImageStatus {
  ExecutableImageError error = ExecutableImageError::None;
  std::string message;

  explicit operator bool() const { return error == ExecutableImageError::None; }
};

struct ExecutableImageRequirementsResult {
  ExecutableImageStatus status;
  uint64_t allocation_size = 0;

  explicit operator bool() const { return static_cast<bool>(status); }
};

struct ResolvedKernel {
  uint64_t descriptor_gpu_address = 0;
  uint64_t code_entry_gpu_address = 0;
};

struct ExecutableAddressResult;
struct ExecutableImageMaterializationResult;

// Owns transport-independent image metadata, but not the backing GPU
// allocation. A transport wrapper must invalidate it when the mapping ends.
class ExecutableImageInfo {
public:
  ExecutableImageInfo() = default;
  ExecutableImageInfo(const ExecutableImageInfo &) = delete;
  ExecutableImageInfo &operator=(const ExecutableImageInfo &) = delete;
  ExecutableImageInfo(ExecutableImageInfo &&other) noexcept;
  ExecutableImageInfo &operator=(ExecutableImageInfo &&other) noexcept;

  [[nodiscard]] uint64_t gpu_address() const { return gpu_address_; }
  [[nodiscard]] uint64_t allocation_size() const { return allocation_size_; }
  [[nodiscard]] uint64_t image_virtual_address() const {
    return code_object_.load_plan.image_virtual_address;
  }
  [[nodiscard]] uint64_t image_size() const {
    return code_object_.load_plan.image_size;
  }
  [[nodiscard]] const loader::CodeObject &code_object() const {
    return code_object_;
  }
  [[nodiscard]] const std::vector<ResolvedKernel> &kernels() const {
    return kernels_;
  }
  explicit operator bool() const { return valid_; }

  [[nodiscard]] ExecutableAddressResult translate(uint64_t virtual_address,
                                                  uint64_t size,
                                                  uint64_t alignment = 1) const;

private:
  friend ExecutableImageMaterializationResult
  materialize_executable_image(const uint8_t *, size_t,
                               const loader::CodeObject &, void *, uint64_t,
                               uint64_t);

  ExecutableImageInfo(const loader::CodeObject &code_object,
                      uint64_t allocation_size, uint64_t gpu_address,
                      std::vector<ResolvedKernel> kernels)
      : code_object_(code_object), allocation_size_(allocation_size),
        gpu_address_(gpu_address), kernels_(std::move(kernels)), valid_(true) {}

  void reset();

  loader::CodeObject code_object_;
  uint64_t allocation_size_ = 0;
  uint64_t gpu_address_ = 0;
  std::vector<ResolvedKernel> kernels_;
  bool valid_ = false;
};

struct ExecutableImageMaterializationResult {
  ExecutableImageStatus status;
  ExecutableImageInfo image;

  explicit operator bool() const { return static_cast<bool>(status); }
};

struct ExecutableAddressResult {
  ExecutableImageStatus status;
  uint64_t gpu_address = 0;

  explicit operator bool() const { return static_cast<bool>(status); }
};

[[nodiscard]] ExecutableImageRequirementsResult
executable_image_requirements(const uint8_t *hsaco, size_t hsaco_size,
                              const loader::CodeObject &code_object);

// Revalidates the complete load plan before writing the destination. The
// destination must provide at least the size returned by
// executable_image_requirements().
[[nodiscard]] ExecutableImageMaterializationResult
materialize_executable_image(const uint8_t *hsaco, size_t hsaco_size,
                             const loader::CodeObject &code_object,
                             void *destination, uint64_t destination_size,
                             uint64_t gpu_address);

[[nodiscard]] ExecutableAddressResult
translate_executable_address(const loader::LoadPlan &plan, uint64_t gpu_address,
                             uint64_t virtual_address, uint64_t size,
                             uint64_t alignment = 1);

[[nodiscard]] const char *
executable_image_error_name(ExecutableImageError error);

} // namespace light_rocr::runtime

#endif
