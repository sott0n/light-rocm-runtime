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
};

enum class MemoryError {
  None,
  InvalidSession,
  InvalidSize,
  OpenKfd,
  AcquireSystemProperties,
  InvalidVramHeap,
  AllocateGtt,
  AllocateExecutableGtt,
  AllocateVram,
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
  [[nodiscard]] AllocationResult allocate_executable_gtt(uint32_t gpu_node_id,
                                                         uint64_t size) const;
  [[nodiscard]] AllocationResult
  allocate_vram(uint32_t gpu_node_id, runtime::MemoryHeapType heap_type,
                uint64_t size) const;
  [[nodiscard]] AqlQueueResult create_aql_queue(uint32_t gpu_node_id,
                                                uint64_t ring_size) const;
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
                   bool host_accessible)
      : state_(std::move(state)), allocation_address_(allocation_address),
        gpu_address_(gpu_address), size_(size), kind_(kind),
        host_accessible_(host_accessible), mapped_(true) {}
  void reset();

  std::shared_ptr<KfdState> state_;
  void *allocation_address_ = nullptr;
  uint64_t gpu_address_ = 0;
  uint64_t size_ = 0;
  MemoryKind kind_ = MemoryKind::Gtt;
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

[[nodiscard]] const char *memory_error_name(MemoryError error);

} // namespace light_rocr::transport::hsakmt

#endif
