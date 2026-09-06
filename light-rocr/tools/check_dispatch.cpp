#include "fixed_gfx1101_store.hpp"

#include "light_rocr/runtime/aql.hpp"
#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/hsakmt/memory.hpp"
#include "light_rocr/transport/hsakmt/queue.hpp"
#include "light_rocr/transport/hsakmt/signal.hpp"
#include "light_rocr/transport/hsakmt/status.hpp"
#include "light_rocr/transport/hsakmt/topology.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <new>
#include <string>
#include <thread>

namespace {

using light_rocr::transport::hsakmt::AqlQueue;
using light_rocr::transport::hsakmt::MemoryAllocation;
using light_rocr::transport::hsakmt::UserSignal;

constexpr uint64_t kOutputOffset = 0;
constexpr uint64_t kKernargOffset = 64;

int fail_memory(const light_rocr::transport::hsakmt::MemoryStatus &status) {
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

int fail_queue(const light_rocr::transport::hsakmt::AqlQueueStatus &status) {
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

int fail_signal(const light_rocr::transport::hsakmt::UserSignalStatus &status) {
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

int cleanup(AqlQueue &queue, UserSignal &signal, MemoryAllocation &data,
            MemoryAllocation &executable) {
  const auto queue_status = queue.release();
  if (!queue_status) {
    fail_queue(queue_status);
    std::cerr.flush();
    std::cout.flush();
    // Do not run destructors for memory still referenced by a live queue.
    std::_Exit(1);
  }
  const auto signal_status = signal.release();
  if (!signal_status) {
    return fail_signal(signal_status);
  }
  const auto data_status = data.release();
  if (!data_status) {
    return fail_memory(data_status);
  }
  const auto executable_status = executable.release();
  if (!executable_status) {
    return fail_memory(executable_status);
  }
  std::cout << "dispatch.cleanup=ok\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  namespace fixed = light_rocr::tools::fixed_gfx1101_store;
  const std::string target = argc > 1 ? argv[1] : "gfx1101";
  if (target != "gfx1101") {
    std::cerr << "message=fixed dispatch smoke only supports gfx1101\n";
    return 1;
  }

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
    return fail_memory(opened.status);
  }
  auto executable = opened.session.allocate_executable_gtt(
      node.node_id, light_rocr::transport::hsakmt::kMemoryPageSize);
  if (!executable) {
    return fail_memory(executable.status);
  }
  if (!fixed::write_image(executable.allocation.host_address(),
                          static_cast<size_t>(executable.allocation.size()))) {
    std::cerr
        << "message=executable allocation cannot hold fixed kernel image\n";
    return 1;
  }

  auto data = opened.session.allocate_gtt(
      node.node_id, light_rocr::transport::hsakmt::kMemoryPageSize);
  if (!data) {
    return fail_memory(data.status);
  }
  std::memset(data.allocation.host_address(), 0,
              static_cast<size_t>(data.allocation.size()));
  auto *data_bytes = static_cast<uint8_t *>(data.allocation.host_address());
  auto *output =
      ::new (static_cast<void *>(data_bytes + kOutputOffset)) uint32_t{0};
  auto *kernarg =
      ::new (static_cast<void *>(data_bytes + kKernargOffset)) fixed::Kernarg;
  kernarg->output_address = data.allocation.gpu_address() + kOutputOffset;
  kernarg->value = fixed::kExpectedValue;

  auto signal = opened.session.create_user_signal(node.node_id, 1);
  if (!signal) {
    return fail_signal(signal.status);
  }
  auto queue = opened.session.create_aql_queue(
      node.node_id, light_rocr::transport::hsakmt::kAqlRingDefaultSize);
  if (!queue) {
    return fail_queue(queue.status);
  }

  light_rocr::runtime::KernelDispatchSpec spec;
  spec.kernel_object = executable.allocation.gpu_address();
  spec.kernarg_address = data.allocation.gpu_address() + kKernargOffset;
  spec.completion_signal = signal.signal.gpu_handle();
  const auto packet = light_rocr::runtime::make_kernel_dispatch_packet(spec);
  if (!packet) {
    std::cerr << "packet_error="
              << light_rocr::runtime::aql_packet_error_name(packet.status.error)
              << '\n';
    std::cerr << "message=" << packet.status.message << '\n';
    return 1;
  }

  std::cout << "node_id=" << node.node_id << '\n';
  std::cout << "target="
            << light_rocr::runtime::gfx_target_name(node.architecture) << '\n';
  std::cout << "kernel.descriptor_gpu_address=0x" << std::hex
            << spec.kernel_object << '\n';
  std::cout << "kernel.entry_gpu_address=0x"
            << spec.kernel_object + fixed::kCodeOffset << '\n';
  std::cout << "kernarg.gpu_address=0x" << spec.kernarg_address << '\n';
  std::cout << "output.gpu_address=0x" << kernarg->output_address << '\n';
  std::cout << "signal.gpu_handle=0x" << spec.completion_signal << '\n';
  std::cout << "queue.id=0x" << queue.queue.queue_id() << '\n';
  std::cout << "queue.doorbell_address=0x" << queue.queue.doorbell_address()
            << '\n';
  std::cout << "packet.header=0x" << packet.packet.header << '\n';
  std::cout << "packet.setup=0x" << packet.packet.setup << std::dec << '\n';
  std::cout.flush();

  const auto submitted = queue.queue.submit_kernel_dispatch(packet.packet);
  if (!submitted) {
    std::cerr << "submit_error="
              << light_rocr::transport::hsakmt::aql_submit_error_name(
                     submitted.error)
              << '\n';
    std::cerr << "message=" << submitted.message << '\n';
    return 1;
  }

  const auto waited = signal.signal.wait_until_equal(
      0, std::chrono::steady_clock::now() + std::chrono::seconds(2));
  const uint32_t observed_output = __atomic_load_n(output, __ATOMIC_ACQUIRE);
  const uint64_t observed_write_index = queue.queue.write_index_relaxed();
  uint64_t observed_read_index = queue.queue.read_index_acquire();
  const auto read_index_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  while (observed_read_index != observed_write_index &&
         std::chrono::steady_clock::now() < read_index_deadline) {
    std::this_thread::yield();
    observed_read_index = queue.queue.read_index_acquire();
  }

  std::cout << "packet.id=" << submitted.packet_id << '\n';
  std::cout << "queue.read_index=" << observed_read_index << '\n';
  std::cout << "queue.write_index=" << observed_write_index << '\n';
  std::cout << "signal.value=" << waited.observed_value << '\n';
  std::cout << "signal.wait=" << (waited ? "satisfied" : "timed_out") << '\n';
  std::cout << "output.expected=0x" << std::hex << fixed::kExpectedValue
            << '\n';
  std::cout << "output.observed=0x" << observed_output << std::dec << '\n';

  const bool correct = waited && observed_output == fixed::kExpectedValue &&
                       observed_read_index == submitted.write_index;
  if (!correct) {
    std::cerr << "message=fixed AQL dispatch did not complete correctly\n";
  }
  const int cleanup_status = cleanup(queue.queue, signal.signal,
                                     data.allocation, executable.allocation);
  return correct && cleanup_status == 0 ? 0 : 1;
}
