#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/kfd/session.hpp"
#include "light_rocr/transport/kfd/topology.hpp"

#include <iostream>

int main() {
  auto opened = light_rocr::transport::kfd::KfdSession::open();
  if (!opened) {
    std::cerr << opened.status.message << '\n';
    return 1;
  }

  const auto discovered =
      light_rocr::transport::kfd::discover_topology(opened.session);
  if (!discovered) {
    std::cerr << discovered.message << '\n';
    return 1;
  }

  const auto selected =
      light_rocr::runtime::select_unique_gpu(discovered.topology, "gfx1101");
  if (!selected) {
    std::cerr << selected.message << '\n';
    return 1;
  }

  const auto &node = discovered.topology.nodes[selected.node_index];
  if (node.gpu_id == 0 || node.compute_unit_count() == 0 ||
      node.memory_banks.empty()) {
    std::cerr << "selected gfx1101 node is incomplete\n";
    return 1;
  }

  std::cout << "KFD " << discovered.topology.kfd_version.major << '.'
            << discovered.topology.kfd_version.minor << ", node "
            << node.node_id << ", gpu_id " << node.gpu_id << '\n';
  return 0;
}
