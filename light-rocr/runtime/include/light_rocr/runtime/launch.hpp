#ifndef LIGHT_ROCR_RUNTIME_LAUNCH_HPP
#define LIGHT_ROCR_RUNTIME_LAUNCH_HPP

#include "light_rocr/loader/code_object.hpp"
#include "light_rocr/runtime/aql.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace light_rocr::runtime {

enum class KernargBufferError {
  None,
  InvalidKernelMetadata,
  InvalidArguments,
  ArgumentSizeExceeded,
  InvalidDestination,
  DestinationTooSmall,
  UnsupportedAlignment,
  MisalignedGpuAddress,
  GpuAddressOverflow,
  AllocationFailed,
};

struct KernargBufferStatus {
  KernargBufferError error = KernargBufferError::None;
  std::string message;

  explicit operator bool() const { return error == KernargBufferError::None; }
};

struct KernargBufferRequirementsResult {
  KernargBufferStatus status;
  uint64_t storage_size = 0;
  uint32_t alignment = 0;

  explicit operator bool() const { return static_cast<bool>(status); }
};

struct KernargBufferMaterializationResult;

// Owns transport-independent kernarg metadata, but not its GPU allocation. A
// transport wrapper must invalidate it as soon as the GPU mapping ends.
class KernargBufferInfo {
public:
  KernargBufferInfo() = default;
  KernargBufferInfo(const KernargBufferInfo &) = delete;
  KernargBufferInfo &operator=(const KernargBufferInfo &) = delete;
  KernargBufferInfo(KernargBufferInfo &&other) noexcept;
  KernargBufferInfo &operator=(KernargBufferInfo &&other) noexcept;

  [[nodiscard]] uint64_t gpu_address() const { return gpu_address_; }
  [[nodiscard]] uint32_t kernarg_size() const { return kernarg_size_; }
  [[nodiscard]] uint32_t alignment() const { return alignment_; }
  explicit operator bool() const { return valid_; }

private:
  friend KernargBufferMaterializationResult
  materialize_kernarg_buffer(const loader::KernelInfo &, const void *, size_t,
                             void *, uint64_t, uint64_t);

  KernargBufferInfo(uint64_t gpu_address, uint32_t kernarg_size,
                    uint32_t alignment)
      : gpu_address_(gpu_address), kernarg_size_(kernarg_size),
        alignment_(alignment), valid_(true) {}

  void reset();

  uint64_t gpu_address_ = 0;
  uint32_t kernarg_size_ = 0;
  uint32_t alignment_ = 0;
  bool valid_ = false;
};

struct KernargBufferMaterializationResult {
  KernargBufferStatus status;
  KernargBufferInfo buffer;

  explicit operator bool() const { return static_cast<bool>(status); }
};

[[nodiscard]] KernargBufferRequirementsResult
kernarg_buffer_requirements(const loader::KernelInfo &kernel,
                            const void *arguments, size_t arguments_size);

// Copies the caller-provided prefix and zero-initializes the remainder of the
// metadata-declared kernarg segment. Bytes beyond that logical segment are left
// untouched so a transport can suballocate it from a larger backing store. The
// destination must provide at least the storage size returned by
// kernarg_buffer_requirements().
[[nodiscard]] KernargBufferMaterializationResult
materialize_kernarg_buffer(const loader::KernelInfo &kernel,
                           const void *arguments, size_t arguments_size,
                           void *destination, uint64_t destination_size,
                           uint64_t gpu_address);

[[nodiscard]] const char *kernarg_buffer_error_name(KernargBufferError error);

} // namespace light_rocr::runtime

#endif
