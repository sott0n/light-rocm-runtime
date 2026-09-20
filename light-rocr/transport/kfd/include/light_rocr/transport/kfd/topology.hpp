#ifndef LIGHT_ROCR_TRANSPORT_KFD_TOPOLOGY_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_TOPOLOGY_HPP

#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/kfd/session.hpp"

#include <string>

namespace light_rocr::transport::kfd {

enum class DiscoveryError {
  None,
  InvalidSession,
  ReadGeneration,
  EnumerateNodes,
  ReadNode,
  InvalidNode,
  ReadMemoryBank,
  InvalidMemoryBank,
  TopologyChanged,
};

struct DiscoveryResult {
  DiscoveryError error = DiscoveryError::None;
  int system_error = 0;
  runtime::Topology topology;
  std::string message;

  explicit operator bool() const { return error == DiscoveryError::None; }
};

[[nodiscard]] const char *discovery_error_name(DiscoveryError error);

[[nodiscard]] DiscoveryResult
read_topology_sysfs(const std::string &topology_path,
                    runtime::KfdVersion version);

[[nodiscard]] DiscoveryResult discover_topology(
    const KfdSession &session,
    const std::string &topology_path = "/sys/class/kfd/kfd/topology");

} // namespace light_rocr::transport::kfd

#endif
