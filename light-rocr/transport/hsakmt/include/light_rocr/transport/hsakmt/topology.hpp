#ifndef LIGHT_ROCR_TRANSPORT_HSAKMT_TOPOLOGY_HPP
#define LIGHT_ROCR_TRANSPORT_HSAKMT_TOPOLOGY_HPP

#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/hsakmt/status.hpp"

#include <cstdint>
#include <string>

namespace light_rocr::transport::hsakmt {

enum class DiscoveryError {
  None,
  OpenKfd,
  QueryKfdVersion,
  AcquireSystemProperties,
  QueryNodeProperties,
  QueryMemoryProperties,
  ReleaseSystemProperties,
  CloseKfd,
};

struct DiscoveryResult {
  DiscoveryError error = DiscoveryError::None;
  uint32_t hsakmt_status = 0;
  runtime::Topology topology;
  std::string message;

  explicit operator bool() const { return error == DiscoveryError::None; }
};

[[nodiscard]] const char *discovery_error_name(DiscoveryError error);
[[nodiscard]] DiscoveryResult discover_topology();

} // namespace light_rocr::transport::hsakmt

#endif
