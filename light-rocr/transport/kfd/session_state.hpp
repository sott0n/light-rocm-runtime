#ifndef LIGHT_ROCR_TRANSPORT_KFD_SESSION_STATE_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_SESSION_STATE_HPP

#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/kfd/vm_types.hpp"

#include <mutex>
#include <unordered_map>

namespace light_rocr::transport::kfd {

struct DeviceVmState {
  int render_fd = -1;
  int32_t drm_render_minor = -1;
  bool acquired = false;
  bool aperture_valid = false;
  bool scratch_reserved = false;
  ProcessAperture aperture;
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
};

} // namespace light_rocr::transport::kfd

#endif
