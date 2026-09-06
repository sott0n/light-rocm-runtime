#ifndef LIGHT_ROCR_TRANSPORT_HSAKMT_MEMORY_HPP
#define LIGHT_ROCR_TRANSPORT_HSAKMT_MEMORY_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace light_rocr::transport::hsakmt {

inline constexpr uint64_t kGttPageSize = 4096;

enum class MemoryError {
  None,
  InvalidSession,
  InvalidSize,
  OpenKfd,
  AcquireSystemProperties,
  AllocateGtt,
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
  explicit operator bool() const { return state_ != nullptr; }

private:
  explicit KfdSession(std::shared_ptr<KfdState> state)
      : state_(std::move(state)) {}

  std::shared_ptr<KfdState> state_;
};

class GttAllocation {
public:
  GttAllocation() = default;
  GttAllocation(const GttAllocation &) = delete;
  GttAllocation &operator=(const GttAllocation &) = delete;
  GttAllocation(GttAllocation &&other) noexcept;
  GttAllocation &operator=(GttAllocation &&other) noexcept;
  ~GttAllocation();

  [[nodiscard]] void *cpu_address() const { return cpu_address_; }
  [[nodiscard]] uint64_t gpu_address() const { return gpu_address_; }
  [[nodiscard]] uint64_t size() const { return size_; }
  explicit operator bool() const { return cpu_address_ != nullptr; }

  [[nodiscard]] MemoryStatus release();

private:
  friend class KfdSession;
  GttAllocation(std::shared_ptr<KfdState> state, void *cpu_address,
                uint64_t gpu_address, uint64_t size)
      : state_(std::move(state)), cpu_address_(cpu_address),
        gpu_address_(gpu_address), size_(size), mapped_(true) {}
  void reset();

  std::shared_ptr<KfdState> state_;
  void *cpu_address_ = nullptr;
  uint64_t gpu_address_ = 0;
  uint64_t size_ = 0;
  bool mapped_ = false;
};

struct SessionResult {
  MemoryStatus status;
  KfdSession session;

  explicit operator bool() const { return static_cast<bool>(status); }
};

struct AllocationResult {
  MemoryStatus status;
  GttAllocation allocation;

  explicit operator bool() const { return static_cast<bool>(status); }
};

[[nodiscard]] const char *memory_error_name(MemoryError error);

} // namespace light_rocr::transport::hsakmt

#endif
