#ifndef LIGHT_ROCR_TRANSPORT_KFD_MEMORY_TYPES_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_MEMORY_TYPES_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace light_rocr::transport::kfd {

inline constexpr uint64_t kMemoryPageSize = 4096;

enum class MemoryError {
  None,
  InvalidSession,
  InvalidNode,
  InvalidSize,
  AcquireVm,
  ReserveVa,
  AllocateGtt,
  MapHost,
  AdviseDontFork,
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
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(host_address_));
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
                uint64_t handle)
      : state_(std::move(state)), reservation_address_(reservation_address),
        reservation_size_(reservation_size), host_address_(host_address),
        size_(size), handle_(handle) {}
  void reset();

  std::shared_ptr<KfdState> state_;
  void *reservation_address_ = nullptr;
  uint64_t reservation_size_ = 0;
  void *host_address_ = nullptr;
  uint64_t size_ = 0;
  uint64_t handle_ = 0;
};

struct GttAllocationResult {
  MemoryStatus status;
  GttAllocation allocation;

  explicit operator bool() const { return static_cast<bool>(status); }
};

} // namespace light_rocr::transport::kfd

#endif
