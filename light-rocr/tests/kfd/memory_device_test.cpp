#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/kfd/memory.hpp"
#include "light_rocr/transport/kfd/session.hpp"
#include "light_rocr/transport/kfd/topology.hpp"

#include <cstdint>
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
  auto allocated = opened.session.allocate_gtt(node, 8192);
  if (!allocated) {
    std::cerr << allocated.status.message << '\n';
    return 1;
  }
  if (allocated.allocation.host_address() == nullptr ||
      !allocated.allocation.gpu_mapped() ||
      allocated.allocation.gpu_address() !=
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
              allocated.allocation.host_address())) ||
      allocated.allocation.size() != 8192) {
    std::cerr << "GTT allocation metadata is invalid\n";
    return 1;
  }

  auto *words = static_cast<uint32_t *>(allocated.allocation.host_address());
  for (uint32_t index = 0; index < 2048; ++index) {
    words[index] = index ^ 0xa5a5a5a5U;
  }
  for (uint32_t index = 0; index < 2048; ++index) {
    if (words[index] != (index ^ 0xa5a5a5a5U)) {
      std::cerr << "GTT CPU mapping did not retain written data\n";
      return 1;
    }
  }

  const uint64_t gpu_address = allocated.allocation.gpu_address();
  const auto released = allocated.allocation.release();
  if (!released || allocated.allocation || allocated.allocation.gpu_mapped() ||
      allocated.allocation.gpu_address() != 0) {
    std::cerr << (released ? "released GTT allocation retained state"
                           : released.message)
              << '\n';
    return 1;
  }

  std::cout << "mapped and freed 8192-byte GTT buffer at GPU VA 0x" << std::hex
            << gpu_address << '\n';
  return 0;
}
