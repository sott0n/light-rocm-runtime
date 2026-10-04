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

  auto vram = opened.session.allocate_vram(node, 8192);
  if (!vram || vram.allocation.host_address() == nullptr ||
      !vram.allocation.gpu_mapped() ||
      vram.allocation.gpu_address() !=
          static_cast<uint64_t>(
              reinterpret_cast<uintptr_t>(vram.allocation.host_address()))) {
    std::cerr << (vram ? "host-mapped VRAM allocation metadata is invalid"
                       : vram.status.message)
              << '\n';
    return 1;
  }
  auto *vram_words = static_cast<uint32_t *>(vram.allocation.host_address());
  for (uint32_t index = 0; index < 2048; ++index) {
    vram_words[index] = index ^ 0x5a5a5a5aU;
  }
  for (uint32_t index = 0; index < 2048; ++index) {
    if (vram_words[index] != (index ^ 0x5a5a5a5aU)) {
      std::cerr << "host-mapped VRAM did not retain CPU-written data\n";
      return 1;
    }
  }
  const auto vram_released = vram.allocation.release();
  if (!vram_released || vram.allocation) {
    std::cerr << (vram_released
                      ? "released host-mapped VRAM allocation retained state"
                      : vram_released.message)
              << '\n';
    return 1;
  }

  auto device_vram = opened.session.allocate_device_vram(node, 8192);
  if (!device_vram || device_vram.allocation.host_address() != nullptr ||
      !device_vram.allocation.gpu_mapped() ||
      device_vram.allocation.gpu_address() == 0) {
    std::cerr << (device_vram ? "device VRAM allocation metadata is invalid"
                              : device_vram.status.message)
              << '\n';
    return 1;
  }
  const auto device_vram_released = device_vram.allocation.release();
  if (!device_vram_released || device_vram.allocation) {
    std::cerr << (device_vram_released
                      ? "released device VRAM allocation retained state"
                      : device_vram_released.message)
              << '\n';
    return 1;
  }

  auto scratch = opened.session.allocate_scratch(node, 8192);
  if (!scratch) {
    std::cerr << scratch.status.message << '\n';
    return 1;
  }
  if (!scratch.allocation.gpu_mapped() ||
      scratch.allocation.gpu_address() == 0 ||
      scratch.allocation.gpu_address() % (64 * 1024) != 0 ||
      scratch.allocation.size() != 64 * 1024) {
    std::cerr << "scratch allocation metadata is invalid\n";
    return 1;
  }

  auto second_opened = light_rocr::transport::kfd::KfdSession::open();
  if (!second_opened) {
    std::cerr << second_opened.status.message << '\n';
    return 1;
  }
  const auto duplicate = second_opened.session.allocate_scratch(node, 4096);
  if (duplicate ||
      duplicate.status.error !=
          light_rocr::transport::kfd::MemoryError::ScratchAlreadyReserved ||
      duplicate.allocation) {
    std::cerr << "duplicate scratch reservation was accepted\n";
    return 1;
  }

  const uint64_t scratch_address = scratch.allocation.gpu_address();
  const auto scratch_released = scratch.allocation.release();
  if (!scratch_released || scratch.allocation ||
      scratch.allocation.gpu_mapped()) {
    std::cerr << (scratch_released
                      ? "released scratch allocation retained state"
                      : scratch_released.message)
              << '\n';
    return 1;
  }

  auto scratch_retry = opened.session.allocate_scratch(node, 4096);
  if (!scratch_retry) {
    std::cerr << "scratch reservation could not be reused: "
              << scratch_retry.status.message << '\n';
    return 1;
  }
  const auto retry_released = scratch_retry.allocation.release();
  if (!retry_released) {
    std::cerr << retry_released.message << '\n';
    return 1;
  }

  std::cout << "mapped and freed 8192-byte GTT buffer at GPU VA 0x" << std::hex
            << gpu_address << "; reserved scratch at 0x" << scratch_address
            << '\n';
  return 0;
}
