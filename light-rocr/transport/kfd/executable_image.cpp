#include "light_rocr/transport/kfd/executable_image.hpp"

#include <string>
#include <utility>

namespace light_rocr::transport::kfd {
namespace {

static_assert(runtime::kExecutableImagePageSize == kMemoryPageSize);

ExecutableImageStatus failure(ExecutableImageError error,
                              const std::string &message) {
  return {error, {}, message};
}

ExecutableImageStatus
from_runtime_status(const runtime::ExecutableImageStatus &status) {
  return {status.error, {}, status.message};
}

} // namespace

const char *executable_image_error_name(ExecutableImageError error) {
  return runtime::executable_image_error_name(error);
}

ExecutableImage::ExecutableImage(ExecutableImage &&other) noexcept
    : allocation_(std::move(other.allocation_)),
      image_(std::move(other.image_)) {
  other.invalidate();
}

void ExecutableImage::invalidate() { image_ = runtime::ExecutableImageInfo{}; }

ExecutableAddressResult ExecutableImage::translate(uint64_t virtual_address,
                                                   uint64_t size,
                                                   uint64_t alignment) const {
  return image_.translate(virtual_address, size, alignment);
}

MemoryStatus ExecutableImage::release() {
  const MemoryStatus status = allocation_.release();
  if (status.error != MemoryError::UnmapFromGpu) {
    invalidate();
  }
  return status;
}

bool KfdSession::owns_executable_image(const ExecutableImage &image,
                                       const runtime::Node &node) const {
  return state_ != nullptr && node.is_gpu() && node.gpu_id != 0 && image &&
         image.allocation_.state_ == state_ &&
         image.allocation_.gpu_ids_.size() == 1 &&
         image.allocation_.gpu_ids_.front() == node.gpu_id;
}

ExecutableImageResult materialize_executable_image(
    const KfdSession &session, const runtime::Node &node, const uint8_t *hsaco,
    size_t hsaco_size, const loader::CodeObject &code_object,
    const std::string &dri_root) {
  const runtime::ExecutableImageRequirementsResult requirements =
      runtime::executable_image_requirements(hsaco, hsaco_size, code_object);
  if (!requirements) {
    return {from_runtime_status(requirements.status), {}};
  }

  GttAllocationResult allocated = session.allocate_executable_gtt(
      node, requirements.allocation_size, dri_root);
  if (!allocated) {
    ExecutableImageStatus allocation_status = failure(
        ExecutableImageError::AllocationFailed,
        "executable GTT allocation failed: " + allocated.status.message);
    allocation_status.memory_status = std::move(allocated.status);
    return {std::move(allocation_status),
            ExecutableImage(std::move(allocated.allocation), {})};
  }

  runtime::ExecutableImageMaterializationResult materialized =
      runtime::materialize_executable_image(
          hsaco, hsaco_size, code_object, allocated.allocation.host_address(),
          allocated.allocation.size(), allocated.allocation.gpu_address());
  if (!materialized) {
    ExecutableImageStatus status = from_runtime_status(materialized.status);
    const MemoryStatus cleanup_status = allocated.allocation.release();
    if (!cleanup_status) {
      status.memory_status = cleanup_status;
      status.message += "; executable allocation cleanup failed: ";
      status.message += cleanup_status.message;
      return {std::move(status),
              ExecutableImage(std::move(allocated.allocation), {})};
    }
    return {std::move(status), {}};
  }

  return {{},
          ExecutableImage(std::move(allocated.allocation),
                          std::move(materialized.image))};
}

} // namespace light_rocr::transport::kfd
