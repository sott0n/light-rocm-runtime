#ifndef LIGHT_ROCR_TRANSPORT_KFD_SESSION_STATE_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_SESSION_STATE_HPP

#include "light_rocr/runtime/topology.hpp"

namespace light_rocr::transport::kfd {

struct KfdState {
  KfdState(int opened_fd, runtime::KfdVersion queried_version);
  KfdState(const KfdState &) = delete;
  KfdState &operator=(const KfdState &) = delete;
  ~KfdState();

  int fd = -1;
  runtime::KfdVersion version;
};

} // namespace light_rocr::transport::kfd

#endif
