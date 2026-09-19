#ifndef LIGHT_ROCR_TRANSPORT_HSAKMT_MEMORY_HPP
#define LIGHT_ROCR_TRANSPORT_HSAKMT_MEMORY_HPP

#include "light_rocr/runtime/topology.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace light_rocr::transport::hsakmt {

inline constexpr uint64_t kMemoryPageSize = 4096;

enum class MemoryKind {
  Gtt,
  Vram,
  Scratch,
};

enum class MemoryError {
  None,
  InvalidSession,
  InvalidSize,
  AllocateHost,
  RegisterHost,
  DeregisterHost,
  OpenKfd,
  AcquireSystemProperties,
  InvalidVramHeap,
  AllocateGtt,
  AllocateExecutableGtt,
  AllocateVram,
  AllocateScratch,
  ScratchAlreadyReserved,
  MapToGpu,
  UnmapFromGpu,
  FreeMemory,
};

struct MemoryStatus {
  MemoryError error = MemoryError::None;
  uint32_t hsakmt_status = 0;
  std::string message;

  explicit operator bool() const { return error == MemoryError::None; }
};

struct KfdState;
struct SessionResult;
struct AllocationResult;
struct HostAllocationResult;
struct ScratchLeaseResult;
struct AqlQueueResult;
struct UserSignalResult;

class KfdSession {
public:
  KfdSession() = default;
  KfdSession(const KfdSession &) = delete;
  KfdSession &operator=(const KfdSession &) = delete;
  KfdSession(KfdSession &&) noexcept = default;
  KfdSession &operator=(KfdSession &&) noexcept = default;
  ~KfdSession() = default;

  [[nodiscard]] static SessionResult open();
  [[nodiscard]] AllocationResult allocate_gtt(uint32_t gpu_node_id,
                                              uint64_t size) const;
  [[nodiscard]] HostAllocationResult allocate_host(uint32_t gpu_node_id,
                                                   uint64_t size) const;
  [[nodiscard]] AllocationResult allocate_executable_gtt(uint32_t gpu_node_id,
                                                         uint64_t size) const;
  [[nodiscard]] AllocationResult
  allocate_vram(uint32_t gpu_node_id, runtime::MemoryHeapType heap_type,
                uint64_t size) const;
  [[nodiscard]] AllocationResult allocate_scratch(uint32_t gpu_node_id,
                                                  uint64_t size) const;
  [[nodiscard]] ScratchLeaseResult
  acquire_scratch_lease(uint32_t gpu_node_id, uint64_t pool_size, uint64_t size,
                        uint64_t alignment) const;
  [[nodiscard]] AqlQueueResult
  create_aql_queue(const runtime::Node &node, uint64_t ring_size,
                   uint32_t private_segment_size = 0) const;
  [[nodiscard]] UserSignalResult
  create_user_signal(uint32_t gpu_node_id, int64_t initial_value) const;
  explicit operator bool() const { return state_ != nullptr; }

private:
  explicit KfdSession(std::shared_ptr<KfdState> state)
      : state_(std::move(state)) {}
  [[nodiscard]] AllocationResult
  allocate(uint32_t preferred_node, uint32_t gpu_node_id, uint64_t size,
           MemoryKind kind, bool host_accessible, MemoryError allocation_error,
           const char *memory_name, bool executable = false,
           bool aql_queue_memory = false) const;

  std::shared_ptr<KfdState> state_;
};

struct ScratchPoolState;

class HostMemoryAllocation {
public:
  HostMemoryAllocation() = default;
  HostMemoryAllocation(const HostMemoryAllocation &) = delete;
  HostMemoryAllocation &operator=(const HostMemoryAllocation &) = delete;
  HostMemoryAllocation(HostMemoryAllocation &&other) noexcept;
  HostMemoryAllocation &operator=(HostMemoryAllocation &&other) noexcept;
  ~HostMemoryAllocation();

  [[nodiscard]] void *host_address() const { return host_address_; }
  [[nodiscard]] uint64_t gpu_address() const { return gpu_address_; }
  [[nodiscard]] uint64_t size() const { return size_; }
  explicit operator bool() const { return host_address_ != nullptr; }

  [[nodiscard]] MemoryStatus release();

private:
  friend class KfdSession;
  HostMemoryAllocation(std::shared_ptr<KfdState> state, void *host_address,
                       uint64_t gpu_address, uint64_t size, bool registered,
                       bool mapped)
      : state_(std::move(state)), host_address_(host_address),
        gpu_address_(gpu_address), size_(size), registered_(registered),
        mapped_(mapped) {}
  void reset();

  std::shared_ptr<KfdState> state_;
  void *host_address_ = nullptr;
  uint64_t gpu_address_ = 0;
  uint64_t size_ = 0;
  bool registered_ = false;
  bool mapped_ = false;
};

class ScratchLease {
public:
  ScratchLease() = default;
  ScratchLease(const ScratchLease &) = delete;
  ScratchLease &operator=(const ScratchLease &) = delete;
  ScratchLease(ScratchLease &&other) noexcept;
  ScratchLease &operator=(ScratchLease &&other) noexcept;
  ~ScratchLease();

  [[nodiscard]] uint64_t gpu_address() const;
  [[nodiscard]] uint64_t size() const { return size_; }
  explicit operator bool() const { return mapped_; }

  [[nodiscard]] MemoryStatus release();

private:
  friend class KfdSession;
  ScratchLease(std::shared_ptr<ScratchPoolState> pool, uint64_t offset,
               uint64_t size, bool mapped, bool range_returned)
      : pool_(std::move(pool)), offset_(offset), size_(size), mapped_(mapped),
        range_returned_(range_returned) {}
  void reset();

  std::shared_ptr<ScratchPoolState> pool_;
  uint64_t offset_ = 0;
  uint64_t size_ = 0;
  bool mapped_ = false;
  bool range_returned_ = false;
};

class MemoryAllocation {
public:
  MemoryAllocation() = default;
  MemoryAllocation(const MemoryAllocation &) = delete;
  MemoryAllocation &operator=(const MemoryAllocation &) = delete;
  MemoryAllocation(MemoryAllocation &&other) noexcept;
  MemoryAllocation &operator=(MemoryAllocation &&other) noexcept;
  ~MemoryAllocation();

  [[nodiscard]] MemoryKind kind() const { return kind_; }
  [[nodiscard]] bool host_accessible() const { return host_accessible_; }
  [[nodiscard]] void *host_address() const {
    return host_accessible_ ? allocation_address_ : nullptr;
  }
  [[nodiscard]] uint64_t gpu_address() const { return gpu_address_; }
  [[nodiscard]] uint64_t size() const { return size_; }
  explicit operator bool() const { return allocation_address_ != nullptr; }

  [[nodiscard]] MemoryStatus release();

private:
  friend class KfdSession;
  MemoryAllocation(std::shared_ptr<KfdState> state, void *allocation_address,
                   uint64_t gpu_address, uint64_t size, MemoryKind kind,
                   bool host_accessible, bool mapped = true,
                   uint32_t scratch_gpu_node_id = 0)
      : state_(std::move(state)), allocation_address_(allocation_address),
        gpu_address_(gpu_address), size_(size), kind_(kind),
        scratch_gpu_node_id_(scratch_gpu_node_id),
        host_accessible_(host_accessible), mapped_(mapped) {}
  void reset();

  std::shared_ptr<KfdState> state_;
  void *allocation_address_ = nullptr;
  uint64_t gpu_address_ = 0;
  uint64_t size_ = 0;
  MemoryKind kind_ = MemoryKind::Gtt;
  uint32_t scratch_gpu_node_id_ = 0;
  bool host_accessible_ = false;
  bool mapped_ = false;
};

struct SessionResult {
  MemoryStatus status;
  KfdSession session;

  explicit operator bool() const { return static_cast<bool>(status); }
};

struct AllocationResult {
  MemoryStatus status;
  MemoryAllocation allocation;

  explicit operator bool() const { return static_cast<bool>(status); }
};

struct HostAllocationResult {
  MemoryStatus status;
  HostMemoryAllocation allocation;

  explicit operator bool() const { return static_cast<bool>(status); }
};

struct ScratchLeaseResult {
  MemoryStatus status;
  ScratchLease lease;

  explicit operator bool() const { return static_cast<bool>(status); }
};

[[nodiscard]] const char *memory_error_name(MemoryError error);

} // namespace light_rocr::transport::hsakmt

#endif
