#ifndef LIGHT_ROCR_TRANSPORT_KFD_MEMORY_TYPES_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_MEMORY_TYPES_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace light_rocr::transport::kfd {

inline constexpr uint64_t kMemoryPageSize = 4096;

enum class MemoryError {
  None,
  InvalidSession,
  InvalidNode,
  InvalidSize,
  AllocateState,
  AcquireVm,
  ReserveVa,
  SetScratchBacking,
  AllocateGtt,
  AllocateScratch,
  ScratchAlreadyReserved,
  MapHost,
  AdviseDontFork,
  MapToGpu,
  UnmapFromGpu,
  UnmapHost,
  FreeGtt,
};

struct MemoryStatus {
  MemoryError error = MemoryError::None;
  int system_error = 0;
  std::string message;

  explicit operator bool() const { return error == MemoryError::None; }
};

struct KfdState;

class GttAllocation {
public:
  GttAllocation() = default;
  GttAllocation(const GttAllocation &) = delete;
  GttAllocation &operator=(const GttAllocation &) = delete;
  GttAllocation(GttAllocation &&other) noexcept;
  GttAllocation &operator=(GttAllocation &&other) noexcept = delete;
  ~GttAllocation();

  [[nodiscard]] void *host_address() const { return host_address_; }
  [[nodiscard]] uint64_t gpu_address() const {
    return gpu_mapped() ? static_cast<uint64_t>(
                              reinterpret_cast<uintptr_t>(host_address_))
                        : 0;
  }
  [[nodiscard]] bool gpu_mapped() const {
    return map_complete_ && !gpu_ids_.empty() &&
           mapped_device_count_ == gpu_ids_.size() &&
           unmapped_device_count_ == 0;
  }
  [[nodiscard]] uint64_t size() const { return size_; }
  explicit operator bool() const {
    return reservation_address_ != nullptr || handle_ != 0;
  }

  [[nodiscard]] MemoryStatus release();

private:
  friend class KfdSession;
  GttAllocation(std::shared_ptr<KfdState> state, void *reservation_address,
                uint64_t reservation_size, void *host_address, uint64_t size,
                uint64_t handle, std::vector<uint32_t> gpu_ids,
                uint32_t mapped_device_count, uint32_t unmapped_device_count,
                bool map_complete)
      : state_(std::move(state)), reservation_address_(reservation_address),
        reservation_size_(reservation_size), host_address_(host_address),
        size_(size), handle_(handle), gpu_ids_(std::move(gpu_ids)),
        mapped_device_count_(mapped_device_count),
        unmapped_device_count_(unmapped_device_count),
        map_complete_(map_complete) {}
  void reset();

  std::shared_ptr<KfdState> state_;
  void *reservation_address_ = nullptr;
  uint64_t reservation_size_ = 0;
  void *host_address_ = nullptr;
  uint64_t size_ = 0;
  uint64_t handle_ = 0;
  std::vector<uint32_t> gpu_ids_;
  uint32_t mapped_device_count_ = 0;
  uint32_t unmapped_device_count_ = 0;
  bool map_complete_ = false;
};

struct GttAllocationResult {
  MemoryStatus status;
  GttAllocation allocation;

  explicit operator bool() const { return static_cast<bool>(status); }
};

class ScratchAllocation {
public:
  ScratchAllocation() = default;
  ScratchAllocation(const ScratchAllocation &) = delete;
  ScratchAllocation &operator=(const ScratchAllocation &) = delete;
  ScratchAllocation(ScratchAllocation &&other) noexcept;
  ScratchAllocation &operator=(ScratchAllocation &&other) noexcept = delete;
  ~ScratchAllocation();

  [[nodiscard]] uint64_t gpu_address() const {
    return gpu_mapped_ ? gpu_address_ : 0;
  }
  [[nodiscard]] bool gpu_mapped() const { return gpu_mapped_; }
  [[nodiscard]] uint64_t size() const { return size_; }
  explicit operator bool() const {
    return reservation_address_ != nullptr || handle_ != 0;
  }

  [[nodiscard]] MemoryStatus release();

private:
  friend class KfdSession;
  ScratchAllocation(std::shared_ptr<KfdState> state, void *reservation_address,
                    uint64_t reservation_size, uint64_t gpu_address,
                    uint64_t size, uint64_t handle,
                    std::vector<uint32_t> gpu_ids, uint32_t mapped_device_count,
                    uint32_t unmapped_device_count, bool map_complete,
                    uint32_t gpu_id, bool integrated, bool gpu_mapped)
      : state_(std::move(state)), reservation_address_(reservation_address),
        reservation_size_(reservation_size), gpu_address_(gpu_address),
        size_(size), handle_(handle), gpu_ids_(std::move(gpu_ids)),
        mapped_device_count_(mapped_device_count),
        unmapped_device_count_(unmapped_device_count),
        map_complete_(map_complete), gpu_id_(gpu_id), integrated_(integrated),
        gpu_mapped_(gpu_mapped) {}
  void reset();

  std::shared_ptr<KfdState> state_;
  void *reservation_address_ = nullptr;
  uint64_t reservation_size_ = 0;
  uint64_t gpu_address_ = 0;
  uint64_t size_ = 0;
  uint64_t handle_ = 0;
  std::vector<uint32_t> gpu_ids_;
  uint32_t mapped_device_count_ = 0;
  uint32_t unmapped_device_count_ = 0;
  bool map_complete_ = false;
  uint32_t gpu_id_ = 0;
  bool integrated_ = false;
  bool gpu_mapped_ = false;
};

struct ScratchAllocationResult {
  MemoryStatus status;
  // Failed setup may retain a reservation or KFD handle for explicit cleanup.
  ScratchAllocation allocation;

  explicit operator bool() const { return static_cast<bool>(status); }
};

} // namespace light_rocr::transport::kfd

#endif
