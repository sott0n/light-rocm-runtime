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
  const auto vram_bank = std::find_if(
      node.memory_banks.begin(), node.memory_banks.end(),
      [](const light_rocr::runtime::MemoryBank &bank) {
        return bank.heap_type ==
                   light_rocr::runtime::MemoryHeapType::FrameBufferPublic ||
               bank.heap_type ==
                   light_rocr::runtime::MemoryHeapType::FrameBufferPrivate;
      });
  if (vram_bank == node.memory_banks.end()) {
    std::cerr << "message=selected GPU has no frame-buffer heap\n";
    return 1;
  }

  auto opened = light_rocr::transport::hsakmt::KfdSession::open();
  if (!opened) {
    return fail(opened.status);
  }
  auto allocated = opened.session.allocate_gtt(
      node.node_id, light_rocr::transport::hsakmt::kMemoryPageSize);
  if (!allocated) {
    return fail(allocated.status);
  }

  auto *bytes = static_cast<uint8_t *>(allocated.allocation.host_address());
  std::fill_n(bytes, allocated.allocation.size(), uint8_t{0xa5});
  const bool host_access_ok =
      std::all_of(bytes, bytes + allocated.allocation.size(),
                  [](uint8_t value) { return value == uint8_t{0xa5}; });

  std::cout << "node_id=" << node.node_id << '\n';
  std::cout << "target="
            << light_rocr::runtime::gfx_target_name(node.architecture) << '\n';
  std::cout << "gtt.size=" << allocated.allocation.size() << '\n';
  std::cout << "gtt.host_address=0x" << std::hex
            << reinterpret_cast<uintptr_t>(allocated.allocation.host_address())
            << '\n';
  std::cout << "gtt.gpu_address=0x" << allocated.allocation.gpu_address()
            << '\n';
  std::cout << "gtt.host_access=" << (host_access_ok ? "ok" : "failed") << '\n';

  const auto released = allocated.allocation.release();
  if (!released) {
    return fail(released);
  }
  std::cout << "gtt.cleanup=ok\n";
  if (!host_access_ok) {
    return 1;
  }

  auto vram = opened.session.allocate_vram(
      node.node_id, vram_bank->heap_type,
      light_rocr::transport::hsakmt::kMemoryPageSize);
  if (!vram) {
    return fail(vram.status);
  }

  bool vram_host_access_ok = true;
  if (vram.allocation.host_accessible()) {
    auto *vram_bytes = static_cast<uint8_t *>(vram.allocation.host_address());
    std::fill_n(vram_bytes, vram.allocation.size(), uint8_t{0x5a});
    vram_host_access_ok =
        std::all_of(vram_bytes, vram_bytes + vram.allocation.size(),
                    [](uint8_t value) { return value == uint8_t{0x5a}; });
  }

  std::cout << "vram.heap="
            << light_rocr::runtime::memory_heap_type_name(vram_bank->heap_type)
            << '\n';
  std::cout << "vram.size=" << std::dec << vram.allocation.size() << '\n';
  std::cout << "vram.gpu_address=0x" << std::hex
            << vram.allocation.gpu_address() << '\n';
  std::cout << "vram.host_access="
            << (vram.allocation.host_accessible()
                    ? (vram_host_access_ok ? "ok" : "failed")
                    : "not_available")
            << '\n';

  const auto vram_released = vram.allocation.release();
  if (!vram_released) {
    return fail(vram_released);
  }
  std::cout << "vram.cleanup=ok\n";
  return vram_host_access_ok ? 0 : 1;
}
