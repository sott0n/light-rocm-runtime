#include "light_rocr/transport/hsakmt/memory.hpp"

#include "light_rocr/transport/hsakmt/status.hpp"

#include <hsakmt/hsakmt.h>

#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace light_rocr::transport::hsakmt {

struct ScratchRange {
  uint64_t size = 0;
  bool free = true;
};

struct ScratchPoolState {
  ScratchPoolState(MemoryAllocation &&reserved, uint64_t capacity)
      : reservation(std::move(reserved)), capacity(capacity) {}

  std::mutex mutex;
  MemoryAllocation reservation;
  uint64_t capacity = 0;
  std::map<uint64_t, ScratchRange> ranges;
  size_t active_leases = 0;
  bool retired = false;
};

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
  std::mutex scratch_pool_mutex;
  std::unordered_map<uint32_t, std::weak_ptr<ScratchPoolState>> scratch_pools;
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

bool add_overflows(uint64_t left, uint64_t right) {
  return left > std::numeric_limits<uint64_t>::max() - right;
}

bool aligned_address(uint64_t base, uint64_t offset, uint64_t alignment,
                     uint64_t *aligned_offset) {
  if (add_overflows(base, offset)) {
    return false;
  }
  const uint64_t address = base + offset;
  const uint64_t remainder = address % alignment;
  const uint64_t padding = remainder == 0 ? 0 : alignment - remainder;
  if (add_overflows(offset, padding)) {
    return false;
  }
  *aligned_offset = offset + padding;
  return true;
}

bool carve_scratch_range(ScratchPoolState *pool,
                         std::map<uint64_t, ScratchRange>::iterator range,
                         uint64_t offset, uint64_t size) {
  const uint64_t range_offset = range->first;
  const uint64_t range_size = range->second.size;
  const uint64_t lease_end = offset + size;
  const uint64_t range_end = range_offset + range_size;
  bool inserted_lease = false;
  bool inserted_suffix = false;
  try {
    if (offset != range_offset) {
      pool->ranges.emplace(offset, ScratchRange{size, false});
      inserted_lease = true;
    }
    if (lease_end != range_end) {
      pool->ranges.emplace(lease_end,
                           ScratchRange{range_end - lease_end, true});
      inserted_suffix = true;
    }
  } catch (const std::bad_alloc &) {
    if (inserted_suffix) {
      pool->ranges.erase(lease_end);
    }
    if (inserted_lease) {
      pool->ranges.erase(offset);
    }
    return false;
  }

  if (offset == range_offset) {
    range->second = ScratchRange{size, false};
  } else {
    range->second.size = offset - range_offset;
  }
  return true;
}

void return_scratch_range(ScratchPoolState *pool, uint64_t offset) {
  auto current = pool->ranges.find(offset);
  assert(current != pool->ranges.end() && !current->second.free);
  current->second.free = true;

  if (current != pool->ranges.begin()) {
    auto previous = std::prev(current);
    if (previous->second.free &&
        previous->first + previous->second.size == current->first) {
      previous->second.size += current->second.size;
      pool->ranges.erase(current);
      current = previous;
    }
  }
  auto next = std::next(current);
  if (next != pool->ranges.end() && next->second.free &&
      current->first + current->second.size == next->first) {
    current->second.size += next->second.size;
    pool->ranges.erase(next);
  }
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
  case MemoryError::AllocateHost:
    return "allocate_host";
  case MemoryError::RegisterHost:
    return "register_host";
  case MemoryError::DeregisterHost:
    return "deregister_host";
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

ScratchLeaseResult KfdSession::acquire_scratch_lease(uint32_t gpu_node_id,
                                                     uint64_t pool_size,
                                                     uint64_t size,
                                                     uint64_t alignment) const {
  if (state_ == nullptr) {
    return {{MemoryError::InvalidSession, 0, "KFD session is not open"}, {}};
  }
  if (pool_size == 0 || size == 0 || size > pool_size ||
      pool_size % kMemoryPageSize != 0 || size % kMemoryPageSize != 0 ||
      alignment == 0 || (alignment & (alignment - 1U)) != 0) {
    return {{MemoryError::InvalidSize, 0,
             "scratch pool, lease, and alignment must be valid page-backed "
             "ranges"},
            {}};
  }

  std::shared_ptr<ScratchPoolState> pool;
  std::unique_lock<std::mutex> pool_lock;
  {
    const std::lock_guard<std::mutex> state_lock(state_->scratch_pool_mutex);
    auto existing = state_->scratch_pools.find(gpu_node_id);
    if (existing != state_->scratch_pools.end()) {
      pool = existing->second.lock();
      if (pool != nullptr) {
        pool_lock = std::unique_lock<std::mutex>(pool->mutex);
        if (pool->retired || !pool->reservation) {
          state_->scratch_pools.erase(existing);
          pool_lock.unlock();
          pool.reset();
        }
      }
    }
    if (pool != nullptr && pool->capacity != pool_size) {
      return {{MemoryError::InvalidSize, 0,
               "scratch pool size changed for an active GPU node"},
              {}};
    }
    if (pool == nullptr) {
      if (!acquire_scratch_reservation(gpu_node_id)) {
        return {{MemoryError::ScratchAlreadyReserved, 0,
                 "scratch backing is already reserved for this GPU node"},
                {}};
      }

      HsaMemFlags allocation_flags{};
      allocation_flags.ui32.Scratch = 1;
      allocation_flags.ui32.HostAccess = 1;
      void *allocation_address = nullptr;
      const HSAKMT_STATUS status = hsaKmtAllocMemory(
          gpu_node_id, pool_size, allocation_flags, &allocation_address);
      if (status != HSAKMT_STATUS_SUCCESS) {
        release_scratch_reservation(gpu_node_id);
        return {failure(MemoryError::AllocateScratch, status,
                        "hsaKmtAllocMemory(scratch pool)"),
                {}};
      }

      MemoryAllocation reservation(
          state_, allocation_address,
          static_cast<uint64_t>(
              reinterpret_cast<uintptr_t>(allocation_address)),
          pool_size, MemoryKind::Scratch, false, false, gpu_node_id);
      try {
        pool = std::make_shared<ScratchPoolState>(std::move(reservation),
                                                  pool_size);
        pool->ranges.emplace(0, ScratchRange{pool_size, true});
        state_->scratch_pools[gpu_node_id] = pool;
        pool_lock = std::unique_lock<std::mutex>(pool->mutex);
      } catch (const std::bad_alloc &) {
        const MemoryStatus cleanup = pool != nullptr
                                         ? pool->reservation.release()
                                         : reservation.release();
        MemoryStatus result{MemoryError::AllocateScratch, 0,
                            "scratch pool host bookkeeping allocation failed"};
        if (!cleanup) {
          result.hsakmt_status = cleanup.hsakmt_status;
          result.message +=
              "; scratch reservation cleanup failed: " + cleanup.message;
        }
        return {std::move(result), {}};
      }
    }
  }

  const uint64_t pool_base = pool->reservation.gpu_address();
  auto selected = pool->ranges.end();
  uint64_t lease_offset = 0;
  for (auto range = pool->ranges.begin(); range != pool->ranges.end();
       ++range) {
    if (!range->second.free) {
      continue;
    }
    uint64_t candidate = 0;
    if (!aligned_address(pool_base, range->first, alignment, &candidate) ||
        candidate < range->first ||
        candidate - range->first > range->second.size ||
        size > range->second.size - (candidate - range->first)) {
      continue;
    }
    selected = range;
    lease_offset = candidate;
    break;
  }
  if (selected == pool->ranges.end() ||
      add_overflows(pool_base, lease_offset) ||
      pool_base + lease_offset > std::numeric_limits<uintptr_t>::max()) {
    return {{MemoryError::AllocateScratch, 0,
             "scratch pool has no aligned range large enough for the queue"},
            {}};
  }

  if (!carve_scratch_range(pool.get(), selected, lease_offset, size)) {
    return {{MemoryError::AllocateScratch, 0,
             "scratch lease host bookkeeping allocation failed"},
            {}};
  }

  void *lease_address = reinterpret_cast<void *>(
      static_cast<uintptr_t>(pool_base + lease_offset));
  uint64_t unused_alternate_gpu_address = 0;
  const HSAKMT_STATUS map_status =
      hsaKmtMapMemoryToGPU(lease_address, size, &unused_alternate_gpu_address);
  if (map_status != HSAKMT_STATUS_SUCCESS) {
    return_scratch_range(pool.get(), lease_offset);
    MemoryStatus result = failure(MemoryError::MapToGpu, map_status,
                                  "hsaKmtMapMemoryToGPU(scratch lease)");
    if (pool->active_leases == 0) {
      const MemoryStatus cleanup = pool->reservation.release();
      if (cleanup) {
        pool->retired = true;
      } else {
        result.message += "; scratch pool cleanup failed: " + cleanup.message;
        return {std::move(result),
                ScratchLease(std::move(pool), 0, size, false, true)};
      }
    }
    return {std::move(result), {}};
  }

  ++pool->active_leases;
  return {{}, ScratchLease(std::move(pool), lease_offset, size, true, false)};
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

ScratchLease::ScratchLease(ScratchLease &&other) noexcept
    : pool_(std::move(other.pool_)), offset_(other.offset_), size_(other.size_),
      mapped_(other.mapped_), range_returned_(other.range_returned_) {
  other.reset();
}

ScratchLease &ScratchLease::operator=(ScratchLease &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  (void)release();
  pool_ = std::move(other.pool_);
  offset_ = other.offset_;
  size_ = other.size_;
  mapped_ = other.mapped_;
  range_returned_ = other.range_returned_;
  other.reset();
  return *this;
}

ScratchLease::~ScratchLease() { (void)release(); }

uint64_t ScratchLease::gpu_address() const {
  if (!mapped_ || pool_ == nullptr || !pool_->reservation) {
    return 0;
  }
  return pool_->reservation.gpu_address() + offset_;
}

MemoryStatus ScratchLease::release() {
  if (pool_ == nullptr) {
    return {};
  }
  std::shared_ptr<ScratchPoolState> pool = pool_;
  {
    const std::lock_guard<std::mutex> lock(pool->mutex);
    if (mapped_) {
      void *lease_address = reinterpret_cast<void *>(
          static_cast<uintptr_t>(pool->reservation.gpu_address() + offset_));
      const HSAKMT_STATUS status = hsaKmtUnmapMemoryToGPU(lease_address);
      if (status != HSAKMT_STATUS_SUCCESS) {
        return failure(MemoryError::UnmapFromGpu, status,
                       "hsaKmtUnmapMemoryToGPU(scratch lease)");
      }
      mapped_ = false;
    }
    if (!range_returned_) {
      return_scratch_range(pool.get(), offset_);
      range_returned_ = true;
      --pool->active_leases;
    }
    if (pool->active_leases == 0 && pool->reservation) {
      const MemoryStatus status = pool->reservation.release();
      if (!status) {
        return status;
      }
      pool->retired = true;
    }
  }
  reset();
  return {};
}

void ScratchLease::reset() {
  pool_.reset();
  offset_ = 0;
  size_ = 0;
  mapped_ = false;
  range_returned_ = false;
}

} // namespace light_rocr::transport::hsakmt
