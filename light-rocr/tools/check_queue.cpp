#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/hsakmt/memory.hpp"
#include "light_rocr/transport/hsakmt/queue.hpp"
#include "light_rocr/transport/hsakmt/status.hpp"
#include "light_rocr/transport/hsakmt/topology.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

int fail(const light_rocr::transport::hsakmt::AqlQueueStatus &status) {
  std::cerr << "queue_error="
            << light_rocr::transport::hsakmt::aql_queue_error_name(status.error)
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
    std::cerr << "memory_error="
              << light_rocr::transport::hsakmt::memory_error_name(
                     opened.status.error)
              << '\n';
    std::cerr << "message=" << opened.status.message << '\n';
    return 1;
  }
  auto created = opened.session.create_aql_queue(
      node, light_rocr::transport::hsakmt::kAqlRingDefaultSize, 272);
  if (!created) {
    return fail(created.status);
  }

  std::cout << "node_id=" << node.node_id << '\n';
  std::cout << "target="
            << light_rocr::runtime::gfx_target_name(node.architecture) << '\n';
  std::cout << "queue.id=0x" << std::hex << created.queue.queue_id() << '\n';
  std::cout << "queue.ring_size=" << std::dec << created.queue.ring_size()
            << '\n';
  std::cout << "queue.ring_host_address=0x" << std::hex
            << reinterpret_cast<uintptr_t>(created.queue.ring_host_address())
            << '\n';
  std::cout << "queue.ring_gpu_address=0x" << created.queue.ring_gpu_address()
            << '\n';
  std::cout << "queue.doorbell_address=0x" << created.queue.doorbell_address()
            << '\n';
  std::cout << "queue.scratch_private_segment_size=" << std::dec
            << created.queue.scratch_private_segment_size() << '\n';
  std::cout << "queue.scratch_size=" << created.queue.scratch_size() << '\n';
  std::cout << "queue.scratch_gpu_address=0x" << std::hex
            << created.queue.scratch_gpu_address() << '\n';

  const auto released = created.queue.release();
  if (!released) {
    return fail(released);
  }
  std::cout << "queue.cleanup=ok\n";
  return 0;
}
