#include "light_rocr/transport/hsakmt/executable_image.hpp"

#include <string>
#include <utility>

namespace light_rocr::transport::hsakmt {
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

ExecutableImageResult
materialize_executable_image(const KfdSession &session, uint32_t gpu_node_id,
                             const uint8_t *hsaco, size_t hsaco_size,
                             const loader::CodeObject &code_object) {
  const runtime::ExecutableImageRequirementsResult requirements =
      runtime::executable_image_requirements(hsaco, hsaco_size, code_object);
  if (!requirements) {
    return {from_runtime_status(requirements.status), {}};
  }

  AllocationResult allocated = session.allocate_executable_gtt(
      gpu_node_id, requirements.allocation_size);
  if (!allocated) {
    ExecutableImageStatus allocation_status = failure(
        ExecutableImageError::AllocationFailed,
        "executable GTT allocation failed: " + allocated.status.message);
    allocation_status.memory_status = std::move(allocated.status);
    return {std::move(allocation_status), {}};
  }

  runtime::ExecutableImageMaterializationResult materialized =
      runtime::materialize_executable_image(
          hsaco, hsaco_size, code_object, allocated.allocation.host_address(),
          allocated.allocation.size(), allocated.allocation.gpu_address());
  if (!materialized) {
    return {from_runtime_status(materialized.status), {}};
  }

  return {{},
          ExecutableImage(std::move(allocated.allocation),
                          std::move(materialized.image))};
}

} // namespace light_rocr::transport::hsakmt
