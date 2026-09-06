#include "light_rocr/runtime/signal.hpp"
#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/hsakmt/memory.hpp"
#include "light_rocr/transport/hsakmt/signal.hpp"
#include "light_rocr/transport/hsakmt/status.hpp"
#include "light_rocr/transport/hsakmt/topology.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

int fail(const light_rocr::transport::hsakmt::UserSignalStatus &status) {
  std::cerr << "signal_error="
            << light_rocr::transport::hsakmt::user_signal_error_name(
                   status.error)
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
  auto created = opened.session.create_user_signal(node.node_id, 1);
  if (!created) {
    return fail(created.status);
  }

  const int64_t initial_value = created.signal.load_relaxed();
  created.signal.store_release(0);
  const auto wait_result = created.signal.wait_until_equal(
      0, std::chrono::steady_clock::now() + std::chrono::seconds(1));

  std::cout << "node_id=" << node.node_id << '\n';
  std::cout << "target="
            << light_rocr::runtime::gfx_target_name(node.architecture) << '\n';
  std::cout << "signal.size=" << sizeof(light_rocr::runtime::AmdSignal) << '\n';
  std::cout << "signal.alignment=" << alignof(light_rocr::runtime::AmdSignal)
            << '\n';
  std::cout << "signal.host_address=0x" << std::hex
            << reinterpret_cast<uintptr_t>(created.signal.host_address())
            << '\n';
  std::cout << "signal.gpu_handle=0x" << created.signal.gpu_handle() << '\n';
  std::cout << "signal.kind=" << std::dec << created.signal.host_address()->kind
            << '\n';
  std::cout << "signal.initial_value=" << initial_value << '\n';
  std::cout << "signal.cpu_wait=" << (wait_result ? "ok" : "timed_out") << '\n';

  const auto released = created.signal.release();
  if (!released) {
    return fail(released);
  }
  std::cout << "signal.cleanup=ok\n";
  return wait_result && initial_value == 1 ? 0 : 1;
}
