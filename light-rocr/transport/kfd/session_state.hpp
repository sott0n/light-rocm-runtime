#ifndef LIGHT_ROCR_TRANSPORT_KFD_SESSION_STATE_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_SESSION_STATE_HPP

#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/kfd/memory_types.hpp"
#include "light_rocr/transport/kfd/vm_types.hpp"

#include <atomic>
#include <link.h>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace light_rocr::transport::kfd {

struct SignalPage;

struct QueueEventPageState {
  QueueEventPageState(MemoryAllocation &&page_allocation, uint64_t page_handle)
      : allocation(std::move(page_allocation)), handle(page_handle) {}

  MemoryAllocation allocation;
  uint64_t handle = 0;
  bool registered = false;
};

struct ScratchPoolState {
  void *reservation_address = nullptr;
  uint64_t reservation_size = 0;
  uint64_t gpu_address = 0;
  uint64_t size = 0;
  bool integrated = false;
  bool configured = false;
  std::map<uint64_t, uint64_t> free_ranges;
};

struct DeviceVmState {
  int render_fd = -1;
  int32_t drm_render_minor = -1;
  bool acquired = false;
  bool memory_policy_configured = false;
  bool aperture_valid = false;
  ProcessAperture aperture;
  std::unique_ptr<ScratchPoolState> scratch_pool;
  std::optional<MemoryAllocation> trap_handler;
  bool trap_handler_configured = false;
};

struct KfdState {
  KfdState(int opened_fd, runtime::KfdVersion queried_version);
  KfdState(const KfdState &) = delete;
  KfdState &operator=(const KfdState &) = delete;
  ~KfdState();

  int fd = -1;
  runtime::KfdVersion version;
  r_debug runtime_debug{11, nullptr, 0, r_debug::RT_CONSISTENT, 0};
  bool exception_debugging_enabled = false;
  uint32_t memory_fault_event_id = 0;
  bool memory_fault_event_active = false;
  std::atomic<bool> memory_faulted{false};
  std::atomic<bool> stop_memory_fault_monitor{false};
  std::thread memory_fault_monitor;
  std::mutex mutex;
  std::unordered_map<uint32_t, DeviceVmState> device_vms;
  std::vector<std::weak_ptr<SignalPage>> signal_pages;
  std::unique_ptr<QueueEventPageState> queue_event_page;
};

} // namespace light_rocr::transport::kfd

namespace light_rocr::transport::kfd::detail {

[[nodiscard]] bool memory_fault_acquire(const std::shared_ptr<KfdState> &state);

} // namespace light_rocr::transport::kfd::detail

#endif
