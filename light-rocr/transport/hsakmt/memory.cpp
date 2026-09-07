#include "light_rocr/transport/hsakmt/memory.hpp"

#include "light_rocr/transport/hsakmt/status.hpp"

#include <hsakmt/hsakmt.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_set>
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

struct ScratchReservationRegistry {
  std::mutex mutex;
  std::unordered_set<uint32_t> gpu_node_ids;
};

ScratchReservationRegistry &scratch_reservations() {
  static ScratchReservationRegistry registry;
  return registry;
}

bool acquire_scratch_reservation(uint32_t gpu_node_id) {
  ScratchReservationRegistry &registry = scratch_reservations();
  const std::lock_guard<std::mutex> lock(registry.mutex);
  return registry.gpu_node_ids.insert(gpu_node_id).second;
}

void release_scratch_reservation(uint32_t gpu_node_id) {
  ScratchReservationRegistry &registry = scratch_reservations();
  const std::lock_guard<std::mutex> lock(registry.mutex);
  registry.gpu_node_ids.erase(gpu_node_id);
}

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
  case MemoryError::InvalidVramHeap:
    return "invalid_vram_heap";
  case MemoryError::AllocateGtt:
    return "allocate_gtt";
  case MemoryError::AllocateExecutableGtt:
    return "allocate_executable_gtt";
  case MemoryError::AllocateVram:
    return "allocate_vram";
  case MemoryError::AllocateScratch:
    return "allocate_scratch";
  case MemoryError::ScratchAlreadyReserved:
    return "scratch_already_reserved";
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
  if (size == 0 || size % kMemoryPageSize != 0) {
    return {{MemoryError::InvalidSize, 0,
             "GTT allocation size must be a non-zero multiple of 4096 bytes"},
            {}};
  }
  return allocate(0, gpu_node_id, size, MemoryKind::Gtt, true,
                  MemoryError::AllocateGtt, "GTT");
}

AllocationResult KfdSession::allocate_executable_gtt(uint32_t gpu_node_id,
                                                     uint64_t size) const {
  if (state_ == nullptr) {
    return {{MemoryError::InvalidSession, 0, "KFD session is not open"}, {}};
  }
  if (size == 0 || size % kMemoryPageSize != 0) {
    return {{MemoryError::InvalidSize, 0,
             "executable GTT allocation size must be a non-zero multiple of "
             "4096 bytes"},
            {}};
  }

  return allocate(0, gpu_node_id, size, MemoryKind::Gtt, true,
                  MemoryError::AllocateExecutableGtt, "executable GTT", true);
}

AllocationResult KfdSession::allocate_vram(uint32_t gpu_node_id,
                                           runtime::MemoryHeapType heap_type,
                                           uint64_t size) const {
  if (state_ == nullptr) {
    return {{MemoryError::InvalidSession, 0, "KFD session is not open"}, {}};
  }
  if (size == 0 || size % kMemoryPageSize != 0) {
    return {{MemoryError::InvalidSize, 0,
             "VRAM allocation size must be a non-zero multiple of 4096 bytes"},
            {}};
  }

  const bool host_accessible =
      heap_type == runtime::MemoryHeapType::FrameBufferPublic;
  if (!host_accessible &&
      heap_type != runtime::MemoryHeapType::FrameBufferPrivate) {
    return {{MemoryError::InvalidVramHeap, 0,
             "VRAM allocation requires a public or private frame-buffer heap"},
            {}};
  }

  return allocate(gpu_node_id, gpu_node_id, size, MemoryKind::Vram,
                  host_accessible, MemoryError::AllocateVram, "VRAM");
}

AllocationResult KfdSession::allocate_scratch(uint32_t gpu_node_id,
                                              uint64_t size) const {
  if (state_ == nullptr) {
    return {{MemoryError::InvalidSession, 0, "KFD session is not open"}, {}};
  }
  if (size == 0 || size % kMemoryPageSize != 0) {
    return {{MemoryError::InvalidSize, 0,
             "scratch allocation size must be a non-zero multiple of 4096 "
             "bytes"},
            {}};
  }
  if (!acquire_scratch_reservation(gpu_node_id)) {
    return {{MemoryError::ScratchAlreadyReserved, 0,
             "scratch backing is already reserved for this GPU node"},
            {}};
  }

  // Match ROCr's scratch-pool reservation contract. libhsakmt interprets
  // this allocation specially: it reserves a 64 KiB-aligned backing VA and
  // programs the process hidden-private base before any physical pages are
  // mapped.
  HsaMemFlags allocation_flags{};
  allocation_flags.ui32.Scratch = 1;
  allocation_flags.ui32.HostAccess = 1;
  void *allocation_address = nullptr;
  HSAKMT_STATUS status = hsaKmtAllocMemory(gpu_node_id, size, allocation_flags,
                                           &allocation_address);
  if (status != HSAKMT_STATUS_SUCCESS) {
    release_scratch_reservation(gpu_node_id);
    return {failure(MemoryError::AllocateScratch, status,
                    "hsaKmtAllocMemory(scratch)"),
            {}};
  }

  uint64_t unused_alternate_gpu_address = 0;
  status = hsaKmtMapMemoryToGPU(allocation_address, size,
                                &unused_alternate_gpu_address);
  if (status != HSAKMT_STATUS_SUCCESS) {
    MemoryStatus result =
        failure(MemoryError::MapToGpu, status, "hsaKmtMapMemoryToGPU(scratch)");
    const HSAKMT_STATUS free_status =
        hsaKmtFreeMemory(allocation_address, size);
    if (free_status == HSAKMT_STATUS_SUCCESS) {
      release_scratch_reservation(gpu_node_id);
      return {std::move(result), {}};
    }
    result.message += "; cleanup hsaKmtFreeMemory failed with ";
    result.message += hsakmt_status_name(static_cast<uint32_t>(free_status));
    return {
        std::move(result),
        MemoryAllocation(state_, allocation_address,
                         static_cast<uint64_t>(
                             reinterpret_cast<uintptr_t>(allocation_address)),
                         size, MemoryKind::Scratch, false, false, gpu_node_id)};
  }

  // gfx9+ queue fields carry the backing VA itself, not an offset within the
  // public scratch aperture. ROCr likewise ignores AlternateVAGPU here.
  const uint64_t gpu_address =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(allocation_address));
  return {{},
          MemoryAllocation(state_, allocation_address, gpu_address, size,
                           MemoryKind::Scratch, false, true, gpu_node_id)};
}

AllocationResult KfdSession::allocate(uint32_t preferred_node,
                                      uint32_t gpu_node_id, uint64_t size,
                                      MemoryKind kind, bool host_accessible,
                                      MemoryError allocation_error,
                                      const char *memory_name, bool executable,
                                      bool aql_queue_memory) const {
  HsaMemFlags allocation_flags{};
  allocation_flags.ui32.NonPaged = 1;
  allocation_flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
  allocation_flags.ui32.HostAccess = host_accessible ? 1U : 0U;
  allocation_flags.ui32.ExecuteAccess = executable ? 1U : 0U;
  allocation_flags.ui32.AQLQueueMemory = aql_queue_memory ? 1U : 0U;
  if (kind == MemoryKind::Gtt) {
    allocation_flags.ui32.NoNUMABind = 1;
  } else {
    allocation_flags.ui32.NoSubstitute = 1;
    allocation_flags.ui32.CoarseGrain = 1;
  }

  void *allocation_address = nullptr;
  HSAKMT_STATUS status = hsaKmtAllocMemory(
      preferred_node, size, allocation_flags, &allocation_address);
  if (status != HSAKMT_STATUS_SUCCESS) {
    return {failure(allocation_error, status,
                    std::string("hsaKmtAllocMemory(") + memory_name + ")"),
            {}};
  }

  HsaMemMapFlags map_flags{};
  map_flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
  map_flags.ui32.HostAccess = host_accessible ? 1U : 0U;
  uint64_t alternate_gpu_address = 0;
  status = hsaKmtMapMemoryToGPUNodes(allocation_address, size,
                                     &alternate_gpu_address, map_flags, 1,
                                     &gpu_node_id);
  if (status != HSAKMT_STATUS_SUCCESS) {
    const HSAKMT_STATUS free_status =
        hsaKmtFreeMemory(allocation_address, size);
    MemoryStatus result =
        failure(MemoryError::MapToGpu, status,
                std::string("hsaKmtMapMemoryToGPUNodes(") + memory_name + ")");
    if (free_status != HSAKMT_STATUS_SUCCESS) {
      result.message += "; cleanup hsaKmtFreeMemory failed with ";
      result.message += hsakmt_status_name(static_cast<uint32_t>(free_status));
    }
    return {std::move(result), {}};
  }

  const uint64_t gpu_address =
      alternate_gpu_address != 0
          ? alternate_gpu_address
          : static_cast<uint64_t>(
                reinterpret_cast<uintptr_t>(allocation_address));
  return {{},
          MemoryAllocation(state_, allocation_address, gpu_address, size, kind,
                           host_accessible)};
}

MemoryAllocation::MemoryAllocation(MemoryAllocation &&other) noexcept
    : state_(std::move(other.state_)),
      allocation_address_(other.allocation_address_),
      gpu_address_(other.gpu_address_), size_(other.size_), kind_(other.kind_),
      scratch_gpu_node_id_(other.scratch_gpu_node_id_),
      host_accessible_(other.host_accessible_), mapped_(other.mapped_) {
  other.allocation_address_ = nullptr;
  other.gpu_address_ = 0;
  other.size_ = 0;
  other.kind_ = MemoryKind::Gtt;
  other.scratch_gpu_node_id_ = 0;
  other.host_accessible_ = false;
  other.mapped_ = false;
}

MemoryAllocation &
MemoryAllocation::operator=(MemoryAllocation &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  (void)release();
  state_ = std::move(other.state_);
  allocation_address_ = other.allocation_address_;
  gpu_address_ = other.gpu_address_;
  size_ = other.size_;
  kind_ = other.kind_;
  scratch_gpu_node_id_ = other.scratch_gpu_node_id_;
  host_accessible_ = other.host_accessible_;
  mapped_ = other.mapped_;
  other.allocation_address_ = nullptr;
  other.gpu_address_ = 0;
  other.size_ = 0;
  other.kind_ = MemoryKind::Gtt;
  other.scratch_gpu_node_id_ = 0;
  other.host_accessible_ = false;
  other.mapped_ = false;
  return *this;
}

MemoryAllocation::~MemoryAllocation() { (void)release(); }

MemoryStatus MemoryAllocation::release() {
  if (allocation_address_ == nullptr) {
    return {};
  }
  if (mapped_) {
    const HSAKMT_STATUS status = hsaKmtUnmapMemoryToGPU(allocation_address_);
    if (status != HSAKMT_STATUS_SUCCESS) {
      return failure(MemoryError::UnmapFromGpu, status,
                     "hsaKmtUnmapMemoryToGPU");
    }
    mapped_ = false;
  }

  const HSAKMT_STATUS status = hsaKmtFreeMemory(allocation_address_, size_);
  if (status != HSAKMT_STATUS_SUCCESS) {
    return failure(MemoryError::FreeMemory, status, "hsaKmtFreeMemory");
  }
  if (kind_ == MemoryKind::Scratch) {
    release_scratch_reservation(scratch_gpu_node_id_);
  }
  reset();
  return {};
}

void MemoryAllocation::reset() {
  state_.reset();
  allocation_address_ = nullptr;
  gpu_address_ = 0;
  size_ = 0;
  kind_ = MemoryKind::Gtt;
  scratch_gpu_node_id_ = 0;
  host_accessible_ = false;
  mapped_ = false;
}

} // namespace light_rocr::transport::hsakmt
