#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/hsakmt/topology.hpp"

#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  std::string target = "gfx1101";
  if (argc == 3 && std::string(argv[1]) == "--target") {
    target = argv[2];
  } else if (argc != 1) {
    std::cerr << "Usage: " << argv[0] << " [--target TARGET_ISA]\n";
    return 2;
  }

  const light_rocr::transport::hsakmt::DiscoveryResult discovered =
      light_rocr::transport::hsakmt::discover_topology();
  if (!discovered) {
    std::cerr << "discovery_error="
              << light_rocr::transport::hsakmt::discovery_error_name(
                     discovered.error)
              << '\n';
    std::cerr << "hsakmt_status="
              << light_rocr::transport::hsakmt::hsakmt_status_name(
                     discovered.hsakmt_status)
              << '\n';
    std::cerr << "message=" << discovered.message << '\n';
    return 1;
  }

  const light_rocr::runtime::Topology &topology = discovered.topology;
  std::cout << "kfd.version=" << topology.kfd_version.major << '.'
            << topology.kfd_version.minor << '\n';
  std::cout << "node_count=" << topology.nodes.size() << '\n';
  for (size_t index = 0; index < topology.nodes.size(); ++index) {
    const light_rocr::runtime::Node &node = topology.nodes[index];
    const std::string prefix = "node." + std::to_string(index) + '.';
    std::cout << prefix << "id=" << node.node_id << '\n';
    std::cout << prefix << "kind=" << (node.is_gpu() ? "gpu" : "cpu") << '\n';
    std::cout << prefix << "gpu_id=" << node.gpu_id << '\n';
    std::cout << prefix << "name=" << std::quoted(node.name) << '\n';
    std::cout << prefix << "target="
              << light_rocr::runtime::gfx_target_name(node.architecture)
              << '\n';
    std::cout << prefix << "cpu_core_count=" << node.cpu_core_count << '\n';
    std::cout << prefix << "simd_count=" << node.simd_count << '\n';
    std::cout << prefix << "compute_unit_count=" << node.compute_unit_count()
              << '\n';
    std::cout << prefix << "wavefront_size=" << node.wavefront_size << '\n';
    std::cout << prefix << "local_memory_size=" << node.local_memory_size
              << '\n';
    std::cout << prefix << "integrated=" << (node.integrated ? 1 : 0) << '\n';
    std::cout << prefix << "vendor_id=0x" << std::hex << node.vendor_id << '\n';
    std::cout << prefix << "device_id=0x" << node.device_id << std::dec << '\n';
    std::cout << prefix << "domain=" << node.domain << '\n';
    std::cout << prefix << "location_id=0x" << std::hex << node.location_id
              << std::dec << '\n';
    std::cout << prefix << "drm_render_minor=" << node.drm_render_minor << '\n';
    std::cout << prefix << "memory_bank_count=" << node.memory_banks.size()
              << '\n';
    for (size_t bank_index = 0; bank_index < node.memory_banks.size();
         ++bank_index) {
      const light_rocr::runtime::MemoryBank &bank =
          node.memory_banks[bank_index];
      const std::string bank_prefix =
          prefix + "memory_bank." + std::to_string(bank_index) + '.';
      std::cout << bank_prefix << "heap_type="
                << light_rocr::runtime::memory_heap_type_name(bank.heap_type)
                << '\n';
      std::cout << bank_prefix << "size=" << bank.size << '\n';
      std::cout << bank_prefix << "virtual_base_address=0x" << std::hex
                << bank.virtual_base_address << std::dec << '\n';
      std::cout << bank_prefix << "width=" << bank.width << '\n';
      std::cout << bank_prefix << "maximum_clock_mhz=" << bank.maximum_clock_mhz
                << '\n';
    }
  }

  const light_rocr::runtime::GpuSelectionResult selected =
      light_rocr::runtime::select_unique_gpu(topology, target);
  if (!selected) {
    std::cerr << "selection_error="
              << light_rocr::runtime::gpu_selection_error_name(selected.error)
              << '\n';
    std::cerr << "message=" << selected.message << '\n';
    return 1;
  }

  const light_rocr::runtime::Node &node = topology.nodes[selected.node_index];
  std::cout << "selected.node_index=" << selected.node_index << '\n';
  std::cout << "selected.node_id=" << node.node_id << '\n';
  std::cout << "selected.gpu_id=" << node.gpu_id << '\n';
  std::cout << "selected.target="
            << light_rocr::runtime::gfx_target_name(node.architecture) << '\n';
  return 0;
}
