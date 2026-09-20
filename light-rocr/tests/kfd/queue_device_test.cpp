#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/kfd/queue.hpp"
#include "light_rocr/transport/kfd/session.hpp"
#include "light_rocr/transport/kfd/topology.hpp"

#include <cstdint>
#include <cstring>
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
  auto created = opened.session.create_aql_queue(
      node, light_rocr::transport::kfd::kAqlRingDefaultSize);
  if (!created) {
    std::cerr << created.status.message << '\n';
    return 1;
  }
  if (!created.queue || created.queue.doorbell_address() == 0 ||
      created.queue.ring_host_address() == nullptr ||
      created.queue.ring_gpu_address() == 0 ||
      created.queue.ring_size() !=
          light_rocr::transport::kfd::kAqlRingDefaultSize ||
      created.queue.packet_count() != 1024 ||
      created.queue.read_index_acquire() != 0 ||
      created.queue.write_index_relaxed() != 0) {
    std::cerr << "direct KFD AQL queue metadata is invalid\n";
    return 1;
  }
  const auto *ring =
      static_cast<const uint8_t *>(created.queue.ring_host_address());
  for (uint64_t packet = 0; packet < created.queue.packet_count(); ++packet) {
    uint16_t header = 0;
    std::memcpy(&header, ring + packet * 64, sizeof(header));
    if (header != 1) {
      std::cerr << "AQL ring contains a valid packet before publication\n";
      return 1;
    }
  }

  const uint32_t queue_id = created.queue.queue_id();
  const auto released = created.queue.release();
  if (!released || created.queue) {
    std::cerr << (released ? "released AQL queue retained state"
                           : released.message)
              << '\n';
    return 1;
  }

  std::cout << "created and destroyed direct KFD AQL queue " << queue_id
            << '\n';
  return 0;
}
