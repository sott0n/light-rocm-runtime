#include "light_rocr/transport/hsakmt/memory.hpp"

#include "light_rocr/transport/hsakmt/status.hpp"

#include <hsakmt/hsakmt.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>

namespace light_rocr::transport::hsakmt {
namespace {

MemoryStatus failure(MemoryError error, HSAKMT_STATUS status,
                     const std::string &operation) {
  return {error, static_cast<uint32_t>(status),
          operation + " failed with " +
              hsakmt_status_name(static_cast<uint32_t>(status)) + " (" +
              std::to_string(static_cast<uint32_t>(status)) + ")"};
}

} // namespace

HostAllocationResult KfdSession::allocate_host(uint32_t gpu_node_id,
                                               uint64_t size) const {
  if (state_ == nullptr) {
    return {{MemoryError::InvalidSession, 0, "KFD session is not open"}, {}};
  }
  if (size == 0 || size > std::numeric_limits<size_t>::max()) {
    return {{MemoryError::InvalidSize, 0,
             "host allocation size must fit in the host address space"},
            {}};
  }

  void *host_address = std::malloc(static_cast<size_t>(size));
  if (host_address == nullptr) {
    return {{MemoryError::AllocateHost, 0, "host memory allocation failed"},
            {}};
  }

  HsaMemFlags registration_flags{};
  registration_flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
  registration_flags.ui32.HostAccess = 1;
  registration_flags.ui32.CachePolicy = HSA_CACHING_CACHED;
  HSAKMT_STATUS status =
      hsaKmtRegisterMemoryWithFlags(host_address, size, registration_flags);
  if (status != HSAKMT_STATUS_SUCCESS) {
    std::free(host_address);
    return {failure(MemoryError::RegisterHost, status,
                    "hsaKmtRegisterMemoryWithFlags(host)"),
            {}};
  }

  HsaMemMapFlags map_flags{};
  map_flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
  map_flags.ui32.HostAccess = 1;
  uint64_t alternate_gpu_address = 0;
  status = hsaKmtMapMemoryToGPUNodes(host_address, size, &alternate_gpu_address,
                                     map_flags, 1, &gpu_node_id);
  if (status != HSAKMT_STATUS_SUCCESS) {
    MemoryStatus result = failure(MemoryError::MapToGpu, status,
                                  "hsaKmtMapMemoryToGPUNodes(host)");
    const HSAKMT_STATUS deregister_status =
        hsaKmtDeregisterMemory(host_address);
    if (deregister_status == HSAKMT_STATUS_SUCCESS) {
      std::free(host_address);
      return {std::move(result), {}};
    }
    result.message += "; cleanup hsaKmtDeregisterMemory failed with ";
    result.message +=
        hsakmt_status_name(static_cast<uint32_t>(deregister_status));
    return {std::move(result),
            HostMemoryAllocation(state_, host_address, 0, size, true, false)};
  }

  const uint64_t gpu_address =
      alternate_gpu_address != 0
          ? alternate_gpu_address
          : static_cast<uint64_t>(reinterpret_cast<uintptr_t>(host_address));
  return {{},
          HostMemoryAllocation(state_, host_address, gpu_address, size, true,
                               true)};
}

HostMemoryAllocation::HostMemoryAllocation(
    HostMemoryAllocation &&other) noexcept
    : state_(std::move(other.state_)), host_address_(other.host_address_),
      gpu_address_(other.gpu_address_), size_(other.size_),
      registered_(other.registered_), mapped_(other.mapped_) {
  other.reset();
}

HostMemoryAllocation &
HostMemoryAllocation::operator=(HostMemoryAllocation &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  (void)release();
  state_ = std::move(other.state_);
  host_address_ = other.host_address_;
  gpu_address_ = other.gpu_address_;
  size_ = other.size_;
  registered_ = other.registered_;
  mapped_ = other.mapped_;
  other.reset();
  return *this;
}

HostMemoryAllocation::~HostMemoryAllocation() { (void)release(); }

MemoryStatus HostMemoryAllocation::release() {
  if (host_address_ == nullptr) {
    return {};
  }
  if (mapped_) {
    const HSAKMT_STATUS status = hsaKmtUnmapMemoryToGPU(host_address_);
    if (status != HSAKMT_STATUS_SUCCESS) {
      return failure(MemoryError::UnmapFromGpu, status,
                     "hsaKmtUnmapMemoryToGPU(host)");
    }
    mapped_ = false;
  }
  if (registered_) {
    const HSAKMT_STATUS status = hsaKmtDeregisterMemory(host_address_);
    if (status != HSAKMT_STATUS_SUCCESS) {
      return failure(MemoryError::DeregisterHost, status,
                     "hsaKmtDeregisterMemory(host)");
    }
    registered_ = false;
  }

  std::free(host_address_);
  reset();
  return {};
}

void HostMemoryAllocation::reset() {
  state_.reset();
  host_address_ = nullptr;
  gpu_address_ = 0;
  size_ = 0;
  registered_ = false;
  mapped_ = false;
}

} // namespace light_rocr::transport::hsakmt
