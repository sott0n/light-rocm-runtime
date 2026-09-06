#include "light_rocr/transport/hsakmt/memory.hpp"

#include "light_rocr/transport/hsakmt/status.hpp"

#include <hsakmt/hsakmt.h>

#include <cstdint>
#include <string>
#include <utility>

namespace light_rocr::transport::hsakmt {

struct KfdState {
  ~KfdState() {
    if (properties_acquired) {
      (void)hsaKmtReleaseSystemProperties();
    }
    if (active) {
      (void)hsaKmtCloseKFD();
    }
  }

  bool active = false;
  bool properties_acquired = false;
};

namespace {

MemoryStatus failure(MemoryError error, HSAKMT_STATUS status,
                     const std::string &operation) {
  return {error, static_cast<uint32_t>(status),
          operation + " failed with " +
              hsakmt_status_name(static_cast<uint32_t>(status)) + " (" +
              std::to_string(static_cast<uint32_t>(status)) + ")"};
}

} // namespace

const char *memory_error_name(MemoryError error) {
  switch (error) {
  case MemoryError::None:
    return "none";
  case MemoryError::InvalidSession:
    return "invalid_session";
  case MemoryError::InvalidSize:
    return "invalid_size";
  case MemoryError::OpenKfd:
    return "open_kfd";
  case MemoryError::AcquireSystemProperties:
    return "acquire_system_properties";
  case MemoryError::AllocateGtt:
    return "allocate_gtt";
  case MemoryError::MapToGpu:
    return "map_to_gpu";
  case MemoryError::UnmapFromGpu:
    return "unmap_from_gpu";
  case MemoryError::FreeMemory:
    return "free_memory";
  }
  return "unknown";
}

SessionResult KfdSession::open() {
  auto state = std::make_shared<KfdState>();
  const HSAKMT_STATUS status = hsaKmtOpenKFD();
  if (status != HSAKMT_STATUS_SUCCESS) {
    return {failure(MemoryError::OpenKfd, status, "hsaKmtOpenKFD"), {}};
  }
  state->active = true;

  HsaSystemProperties properties{};
  const HSAKMT_STATUS acquire_status =
      hsaKmtAcquireSystemProperties(&properties);
  if (acquire_status != HSAKMT_STATUS_SUCCESS) {
    return {failure(MemoryError::AcquireSystemProperties, acquire_status,
                    "hsaKmtAcquireSystemProperties"),
            {}};
  }
  state->properties_acquired = true;
  return {{}, KfdSession(std::move(state))};
}

AllocationResult KfdSession::allocate_gtt(uint32_t gpu_node_id,
                                          uint64_t size) const {
  if (state_ == nullptr) {
    return {{MemoryError::InvalidSession, 0, "KFD session is not open"}, {}};
  }
  if (size == 0 || size % kGttPageSize != 0) {
    return {{MemoryError::InvalidSize, 0,
             "GTT allocation size must be a non-zero multiple of 4096 bytes"},
            {}};
  }

  HsaMemFlags allocation_flags{};
  allocation_flags.ui32.NonPaged = 1;
  allocation_flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
  allocation_flags.ui32.HostAccess = 1;
  allocation_flags.ui32.NoNUMABind = 1;

  void *cpu_address = nullptr;
  HSAKMT_STATUS status =
      hsaKmtAllocMemory(0, size, allocation_flags, &cpu_address);
  if (status != HSAKMT_STATUS_SUCCESS) {
    return {failure(MemoryError::AllocateGtt, status, "hsaKmtAllocMemory"), {}};
  }

  HsaMemMapFlags map_flags{};
  map_flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
  map_flags.ui32.HostAccess = 1;
  uint64_t alternate_gpu_address = 0;
  status = hsaKmtMapMemoryToGPUNodes(cpu_address, size, &alternate_gpu_address,
                                     map_flags, 1, &gpu_node_id);
  if (status != HSAKMT_STATUS_SUCCESS) {
    const HSAKMT_STATUS free_status = hsaKmtFreeMemory(cpu_address, size);
    MemoryStatus result =
        failure(MemoryError::MapToGpu, status, "hsaKmtMapMemoryToGPUNodes");
    if (free_status != HSAKMT_STATUS_SUCCESS) {
      result.message += "; cleanup hsaKmtFreeMemory failed with ";
      result.message += hsakmt_status_name(static_cast<uint32_t>(free_status));
    }
    return {std::move(result), {}};
  }

  const uint64_t gpu_address =
      alternate_gpu_address != 0
          ? alternate_gpu_address
          : static_cast<uint64_t>(reinterpret_cast<uintptr_t>(cpu_address));
  return {{}, GttAllocation(state_, cpu_address, gpu_address, size)};
}

GttAllocation::GttAllocation(GttAllocation &&other) noexcept
    : state_(std::move(other.state_)), cpu_address_(other.cpu_address_),
      gpu_address_(other.gpu_address_), size_(other.size_),
      mapped_(other.mapped_) {
  other.cpu_address_ = nullptr;
  other.gpu_address_ = 0;
  other.size_ = 0;
  other.mapped_ = false;
}

GttAllocation &GttAllocation::operator=(GttAllocation &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  (void)release();
  state_ = std::move(other.state_);
  cpu_address_ = other.cpu_address_;
  gpu_address_ = other.gpu_address_;
  size_ = other.size_;
  mapped_ = other.mapped_;
  other.cpu_address_ = nullptr;
  other.gpu_address_ = 0;
  other.size_ = 0;
  other.mapped_ = false;
  return *this;
}

GttAllocation::~GttAllocation() { (void)release(); }

MemoryStatus GttAllocation::release() {
  if (cpu_address_ == nullptr) {
    return {};
  }
  if (mapped_) {
    const HSAKMT_STATUS status = hsaKmtUnmapMemoryToGPU(cpu_address_);
    if (status != HSAKMT_STATUS_SUCCESS) {
      return failure(MemoryError::UnmapFromGpu, status,
                     "hsaKmtUnmapMemoryToGPU");
    }
    mapped_ = false;
  }

  const HSAKMT_STATUS status = hsaKmtFreeMemory(cpu_address_, size_);
  if (status != HSAKMT_STATUS_SUCCESS) {
    return failure(MemoryError::FreeMemory, status, "hsaKmtFreeMemory");
  }
  reset();
  return {};
}

void GttAllocation::reset() {
  state_.reset();
  cpu_address_ = nullptr;
  gpu_address_ = 0;
  size_ = 0;
  mapped_ = false;
}

} // namespace light_rocr::transport::hsakmt
