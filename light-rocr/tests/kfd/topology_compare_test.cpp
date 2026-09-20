#include "light_rocr/transport/hsakmt/topology.hpp"
#include "light_rocr/transport/kfd/session.hpp"
#include "light_rocr/transport/kfd/topology.hpp"

#include <algorithm>
#include <iostream>

namespace {

bool same_architecture(const light_rocr::runtime::GpuArchitecture &left,
                       const light_rocr::runtime::GpuArchitecture &right) {
  return left.major == right.major && left.minor == right.minor &&
         left.stepping == right.stepping;
}

const light_rocr::runtime::Node *
find_matching_node(const light_rocr::runtime::Topology &topology,
                   const light_rocr::runtime::Node &direct_node) {
  const auto found = std::find_if(
      topology.nodes.begin(), topology.nodes.end(), [&](const auto &node) {
        if (direct_node.is_gpu()) {
          return node.is_gpu() && node.gpu_id == direct_node.gpu_id;
        }
        return !node.is_gpu() &&
               node.cpu_core_count == direct_node.cpu_core_count;
      });
  return found == topology.nodes.end() ? nullptr : &*found;
}

bool compare_node(const light_rocr::runtime::Node &direct,
                  const light_rocr::runtime::Node &kmt) {
  bool matches = true;
  const auto expect = [&](bool condition, const char *field) {
    if (!condition) {
      std::cerr << "mismatch node " << direct.node_id << ": " << field << '\n';
      matches = false;
    }
  };

  expect(direct.gpu_id == kmt.gpu_id, "gpu_id");
  expect(direct.cpu_core_count == kmt.cpu_core_count, "cpu_core_count");
  expect(direct.simd_count == kmt.simd_count, "simd_count");
  expect(direct.simd_per_compute_unit == kmt.simd_per_compute_unit,
         "simd_per_compute_unit");
  expect(direct.wavefront_size == kmt.wavefront_size, "wavefront_size");
  expect(direct.maximum_waves_per_simd == kmt.maximum_waves_per_simd,
         "maximum_waves_per_simd");
  expect(direct.shader_engine_count == kmt.shader_engine_count,
         "shader_engine_count");
  expect(direct.maximum_scratch_waves_per_compute_unit ==
             kmt.maximum_scratch_waves_per_compute_unit,
         "maximum_scratch_waves_per_compute_unit");
  expect(direct.xcc_count == kmt.xcc_count, "xcc_count");
  expect(direct.vendor_id == kmt.vendor_id, "vendor_id");
  expect(direct.device_id == kmt.device_id, "device_id");
  expect(direct.domain == kmt.domain, "domain");
  expect(direct.location_id == kmt.location_id, "location_id");
  expect(direct.drm_render_minor == kmt.drm_render_minor, "drm_render_minor");
  expect(direct.local_memory_size == kmt.local_memory_size,
         "local_memory_size");
  expect(direct.integrated == kmt.integrated, "integrated");
  expect(same_architecture(direct.architecture, kmt.architecture),
         "architecture");
  if (direct.is_gpu()) {
    expect(direct.name == kmt.name, "name");
  }

  for (const auto &direct_bank : direct.memory_banks) {
    const auto found = std::find_if(
        kmt.memory_banks.begin(), kmt.memory_banks.end(),
        [&](const auto &bank) {
          return bank.heap_type == direct_bank.heap_type &&
                 bank.size == direct_bank.size &&
                 bank.virtual_base_address ==
                     direct_bank.virtual_base_address &&
                 bank.width == direct_bank.width &&
                 bank.maximum_clock_mhz == direct_bank.maximum_clock_mhz &&
                 bank.hot_pluggable == direct_bank.hot_pluggable &&
                 bank.non_volatile == direct_bank.non_volatile;
        });
    expect(found != kmt.memory_banks.end(), "physical memory bank");
  }
  return matches;
}

} // namespace

int main() {
  auto opened = light_rocr::transport::kfd::KfdSession::open();
  if (!opened) {
    std::cerr << opened.status.message << '\n';
    return 1;
  }
  const auto direct =
      light_rocr::transport::kfd::discover_topology(opened.session);
  if (!direct) {
    std::cerr << direct.message << '\n';
    return 1;
  }

  const auto kmt = light_rocr::transport::hsakmt::discover_topology();
  if (!kmt) {
    std::cerr << kmt.message << '\n';
    return 1;
  }

  if (direct.topology.kfd_version.major != kmt.topology.kfd_version.major ||
      direct.topology.kfd_version.minor != kmt.topology.kfd_version.minor) {
    std::cerr << "KFD version mismatch\n";
    return 1;
  }
  if (direct.topology.nodes.size() != kmt.topology.nodes.size()) {
    std::cerr << "node count mismatch\n";
    return 1;
  }

  bool matches = true;
  for (const auto &node : direct.topology.nodes) {
    const auto *kmt_node = find_matching_node(kmt.topology, node);
    if (kmt_node == nullptr) {
      std::cerr << "no KMT match for direct node " << node.node_id << '\n';
      matches = false;
      continue;
    }
    matches = compare_node(node, *kmt_node) && matches;
  }
  if (!matches) {
    return 1;
  }

  std::cout << "direct KFD topology matches libhsakmt\n";
  return 0;
}
