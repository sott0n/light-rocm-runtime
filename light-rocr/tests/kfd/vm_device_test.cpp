#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/kfd/session.hpp"
#include "light_rocr/transport/kfd/topology.hpp"
#include "light_rocr/transport/kfd/vm.hpp"

#include <algorithm>
#include <iostream>

namespace {

bool same_aperture(const light_rocr::transport::kfd::ProcessAperture &left,
                   const light_rocr::transport::kfd::ProcessAperture &right) {
  return left.gpu_id == right.gpu_id && left.lds_base == right.lds_base &&
         left.lds_limit == right.lds_limit &&
         left.scratch_base == right.scratch_base &&
         left.scratch_limit == right.scratch_limit &&
         left.gpuvm_base == right.gpuvm_base &&
         left.gpuvm_limit == right.gpuvm_limit;
}

} // namespace

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
  const auto cpu = std::find_if(
      discovered.topology.nodes.begin(), discovered.topology.nodes.end(),
      [](const auto &candidate) { return !candidate.is_gpu(); });
  if (cpu == discovered.topology.nodes.end()) {
    std::cerr << "no CPU node available for validation\n";
    return 1;
  }
  const auto invalid_node = opened.session.acquire_vm(*cpu);
  if (invalid_node || invalid_node.status.error !=
                          light_rocr::transport::kfd::VmError::InvalidNode) {
    std::cerr << "CPU node was accepted for VM acquisition\n";
    return 1;
  }
  const auto missing_render =
      opened.session.acquire_vm(node, "/definitely/missing/light-rocr-dri");
  if (missing_render ||
      missing_render.status.error !=
          light_rocr::transport::kfd::VmError::OpenRenderNode) {
    std::cerr << "missing render node was not reported\n";
    return 1;
  }

  const auto acquired = opened.session.acquire_vm(node);
  if (!acquired) {
    std::cerr << acquired.status.message << '\n';
    return 1;
  }
  const auto &aperture = acquired.aperture;
  if (aperture.gpu_id != node.gpu_id || aperture.lds_base == 0 ||
      aperture.lds_limit < aperture.lds_base || aperture.scratch_base == 0 ||
      aperture.scratch_limit < aperture.scratch_base ||
      aperture.gpuvm_limit <= aperture.gpuvm_base) {
    std::cerr << "acquired gfx1101 VM has invalid process apertures\n";
    return 1;
  }

  auto mismatched_node = node;
  ++mismatched_node.drm_render_minor;
  const auto mismatched = opened.session.acquire_vm(mismatched_node);
  if (mismatched || mismatched.status.error !=
                        light_rocr::transport::kfd::VmError::InvalidNode) {
    std::cerr << "cached VM accepted a different DRM render node\n";
    return 1;
  }

  const auto repeated = opened.session.acquire_vm(node);
  if (!repeated || !same_aperture(aperture, repeated.aperture)) {
    std::cerr << "repeated VM acquisition was not idempotent\n";
    return 1;
  }

  auto second_session = light_rocr::transport::kfd::KfdSession::open();
  if (!second_session) {
    std::cerr << second_session.status.message << '\n';
    return 1;
  }
  const auto shared = second_session.session.acquire_vm(node);
  if (!shared || !same_aperture(aperture, shared.aperture)) {
    std::cerr << "a second KFD session did not share the process VM\n";
    if (!shared) {
      std::cerr << shared.status.message << '\n';
    }
    return 1;
  }

  std::cout << "acquired VM for gpu_id " << aperture.gpu_id << ", gpuvm 0x"
            << std::hex << aperture.gpuvm_base << "-0x" << aperture.gpuvm_limit
            << '\n';
  return 0;
}
