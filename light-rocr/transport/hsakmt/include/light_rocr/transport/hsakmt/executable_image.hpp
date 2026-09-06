#ifndef LIGHT_ROCR_TRANSPORT_HSAKMT_EXECUTABLE_IMAGE_HPP
#define LIGHT_ROCR_TRANSPORT_HSAKMT_EXECUTABLE_IMAGE_HPP

#include "light_rocr/loader/code_object.hpp"
#include "light_rocr/runtime/executable_image.hpp"
#include "light_rocr/transport/hsakmt/memory.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace light_rocr::transport::hsakmt {

using ExecutableAddressResult = runtime::ExecutableAddressResult;
using ExecutableImageError = runtime::ExecutableImageError;
using ResolvedKernel = runtime::ResolvedKernel;

struct ExecutableImageStatus {
  ExecutableImageError error = ExecutableImageError::None;
  MemoryStatus memory_status;
  std::string message;

  explicit operator bool() const { return error == ExecutableImageError::None; }
};

struct ExecutableImageResult;

class ExecutableImage {
public:
  ExecutableImage() = default;
  ExecutableImage(const ExecutableImage &) = delete;
  ExecutableImage &operator=(const ExecutableImage &) = delete;
  ExecutableImage(ExecutableImage &&other) noexcept;
  ExecutableImage &operator=(ExecutableImage &&) noexcept = delete;
  ~ExecutableImage() = default;

  [[nodiscard]] void *host_address() const {
    return image_ ? allocation_.host_address() : nullptr;
  }
  [[nodiscard]] uint64_t gpu_address() const { return image_.gpu_address(); }
  [[nodiscard]] uint64_t allocation_size() const {
    return image_.allocation_size();
  }
  [[nodiscard]] uint64_t image_virtual_address() const {
    return image_.image_virtual_address();
  }
  [[nodiscard]] uint64_t image_size() const { return image_.image_size(); }
  [[nodiscard]] const loader::CodeObject &code_object() const {
    return image_.code_object();
  }
  [[nodiscard]] const std::vector<ResolvedKernel> &kernels() const {
    return image_.kernels();
  }
  // Valid only while this transport image remains GPU-usable.
  [[nodiscard]] const runtime::ExecutableImageInfo &runtime_image() const {
    return image_;
  }
  // Cleanup ownership can remain after GPU usability has been invalidated.
  [[nodiscard]] bool owns_allocation() const {
    return static_cast<bool>(allocation_);
  }
  explicit operator bool() const { return static_cast<bool>(image_); }

  [[nodiscard]] ExecutableAddressResult translate(uint64_t virtual_address,
                                                  uint64_t size,
                                                  uint64_t alignment = 1) const;
  [[nodiscard]] MemoryStatus release();

private:
  friend ExecutableImageResult
  materialize_executable_image(const KfdSession &, uint32_t, const uint8_t *,
                               size_t, const loader::CodeObject &);

  ExecutableImage(MemoryAllocation allocation,
                  runtime::ExecutableImageInfo image)
      : allocation_(std::move(allocation)), image_(std::move(image)) {}

  void invalidate();

  MemoryAllocation allocation_;
  runtime::ExecutableImageInfo image_;
};

struct ExecutableImageResult {
  ExecutableImageStatus status;
  ExecutableImage image;

  explicit operator bool() const { return static_cast<bool>(status); }
};

[[nodiscard]] ExecutableImageResult
materialize_executable_image(const KfdSession &session, uint32_t gpu_node_id,
                             const uint8_t *hsaco, size_t hsaco_size,
                             const loader::CodeObject &code_object);

[[nodiscard]] const char *
executable_image_error_name(ExecutableImageError error);

} // namespace light_rocr::transport::hsakmt

#endif
