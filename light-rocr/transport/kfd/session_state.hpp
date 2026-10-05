#ifndef LIGHT_ROCR_TRANSPORT_KFD_SESSION_STATE_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_SESSION_STATE_HPP

#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/kfd/vm_types.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace light_rocr::transport::kfd {

struct SignalPage;

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
};

struct KfdState {
  KfdState(int opened_fd, runtime::KfdVersion queried_version);
  KfdState(const KfdState &) = delete;
  KfdState &operator=(const KfdState &) = delete;
  ~KfdState();

  int fd = -1;
  runtime::KfdVersion version;
  std::mutex mutex;
  std::unordered_map<uint32_t, DeviceVmState> device_vms;
  std::vector<std::weak_ptr<SignalPage>> signal_pages;
};

} // namespace light_rocr::transport::kfd

#endif
