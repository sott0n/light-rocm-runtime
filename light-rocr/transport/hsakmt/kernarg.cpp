#include "light_rocr/transport/hsakmt/kernarg.hpp"

#include <string>
#include <utility>

namespace light_rocr::transport::hsakmt {
namespace {

KernargBufferStatus
from_runtime_status(const runtime::KernargBufferStatus &status) {
  return {status.error, {}, status.message};
}

} // namespace

const char *kernarg_buffer_error_name(KernargBufferError error) {
  return runtime::kernarg_buffer_error_name(error);
}

KernargBuffer::KernargBuffer(KernargBuffer &&other) noexcept
    : allocation_(std::move(other.allocation_)),
      buffer_(std::move(other.buffer_)) {
  other.invalidate();
}

void KernargBuffer::invalidate() { buffer_ = runtime::KernargBufferInfo{}; }

MemoryStatus KernargBuffer::release() {
  const MemoryStatus status = allocation_.release();
  if (status.error != MemoryError::UnmapFromGpu) {
    invalidate();
  }
  return status;
}

KernargBufferResult create_kernarg_buffer(const KfdSession &session,
                                          uint32_t gpu_node_id,
                                          const loader::KernelInfo &kernel,
                                          const void *arguments,
                                          size_t arguments_size) {
  const runtime::KernargBufferRequirementsResult requirements =
      runtime::kernarg_buffer_requirements(kernel, arguments, arguments_size);
  if (!requirements) {
    return {from_runtime_status(requirements.status), {}};
  }

  if (requirements.storage_size == 0) {
    runtime::KernargBufferMaterializationResult materialized =
        runtime::materialize_kernarg_buffer(kernel, arguments, arguments_size,
                                            nullptr, 0, 0);
    if (!materialized) {
      return {from_runtime_status(materialized.status), {}};
    }
    return {{}, KernargBuffer({}, std::move(materialized.buffer))};
  }
  if (requirements.alignment > kMemoryPageSize) {
    return {{KernargBufferError::UnsupportedAlignment,
             {},
             "kernel kernarg alignment exceeds the 4096-byte GTT backing "
             "alignment"},
            {}};
  }

  const uint64_t allocation_size =
      (requirements.storage_size + kMemoryPageSize - 1U) &
      ~(kMemoryPageSize - 1U);

  AllocationResult allocated =
      session.allocate_gtt(gpu_node_id, allocation_size);
  if (!allocated) {
    KernargBufferStatus status{
        KernargBufferError::AllocationFailed, allocated.status,
        "kernarg GTT allocation failed: " + allocated.status.message};
    return {std::move(status), {}};
  }

  runtime::KernargBufferMaterializationResult materialized =
      runtime::materialize_kernarg_buffer(kernel, arguments, arguments_size,
                                          allocated.allocation.host_address(),
                                          allocated.allocation.size(),
                                          allocated.allocation.gpu_address());
  if (!materialized) {
    KernargBufferStatus status = from_runtime_status(materialized.status);
    const MemoryStatus cleanup_status = allocated.allocation.release();
    if (!cleanup_status) {
      status.memory_status = cleanup_status;
      status.message += "; kernarg allocation cleanup failed: ";
      status.message += cleanup_status.message;
      return {std::move(status), KernargBuffer(std::move(allocated.allocation),
                                               runtime::KernargBufferInfo{})};
    }
    return {std::move(status), {}};
  }

  return {{},
          KernargBuffer(std::move(allocated.allocation),
                        std::move(materialized.buffer))};
}

} // namespace light_rocr::transport::hsakmt
