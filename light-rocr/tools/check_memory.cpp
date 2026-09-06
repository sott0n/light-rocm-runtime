#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/hsakmt/memory.hpp"
#include "light_rocr/transport/hsakmt/topology.hpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

int fail(const light_rocr::transport::hsakmt::MemoryStatus &status) {
  std::cerr << "memory_error="
            << light_rocr::transport::hsakmt::memory_error_name(status.error)
            << '\n';
  std::cerr << "hsakmt_status="
            << light_rocr::transport::hsakmt::hsakmt_status_name(
                   status.hsakmt_status)
            << '\n';
  std::cerr << "message=" << status.message << '\n';
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  const std::string target = argc > 1 ? argv[1] : "gfx1101";
  const auto discovered = light_rocr::transport::hsakmt::discover_topology();
  if (!discovered) {
    std::cerr << "discovery_error="
              << light_rocr::transport::hsakmt::discovery_error_name(
                     discovered.error)
              << '\n';
    std::cerr << "message=" << discovered.message << '\n';
    return 1;
  }

  const auto selected =
      light_rocr::runtime::select_unique_gpu(discovered.topology, target);
  if (!selected) {
    std::cerr << "selection_error="
              << light_rocr::runtime::gpu_selection_error_name(selected.error)
              << '\n';
    std::cerr << "message=" << selected.message << '\n';
    return 1;
  }
  const auto &node = discovered.topology.nodes[selected.node_index];

  auto opened = light_rocr::transport::hsakmt::KfdSession::open();
  if (!opened) {
    return fail(opened.status);
  }
  auto allocated = opened.session.allocate_gtt(
      node.node_id, light_rocr::transport::hsakmt::kGttPageSize);
  if (!allocated) {
    return fail(allocated.status);
  }

  auto *bytes = static_cast<uint8_t *>(allocated.allocation.cpu_address());
  std::fill_n(bytes, allocated.allocation.size(), uint8_t{0xa5});
  const bool host_access_ok =
      std::all_of(bytes, bytes + allocated.allocation.size(),
                  [](uint8_t value) { return value == uint8_t{0xa5}; });

  std::cout << "node_id=" << node.node_id << '\n';
  std::cout << "target="
            << light_rocr::runtime::gfx_target_name(node.architecture) << '\n';
  std::cout << "size=" << allocated.allocation.size() << '\n';
  std::cout << "cpu_address=0x" << std::hex
            << reinterpret_cast<uintptr_t>(allocated.allocation.cpu_address())
            << '\n';
  std::cout << "gpu_address=0x" << allocated.allocation.gpu_address() << '\n';
  std::cout << "host_access=" << (host_access_ok ? "ok" : "failed") << '\n';

  const auto released = allocated.allocation.release();
  if (!released) {
    return fail(released);
  }
  std::cout << "cleanup=ok\n";
  return host_access_ok ? 0 : 1;
}
