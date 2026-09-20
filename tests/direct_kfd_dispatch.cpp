#include "aql_producer.hpp"
#include "fixed_gfx1101_store.hpp"
#include "light_rocr_kfd_aql_queue.hpp"

#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/kfd/memory.hpp"
#include "light_rocr/transport/kfd/queue.hpp"
#include "light_rocr/transport/kfd/session.hpp"
#include "light_rocr/transport/kfd/signal.hpp"
#include "light_rocr/transport/kfd/topology.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <thread>

namespace {

using light_rocr::transport::kfd::AqlQueue;
using light_rocr::transport::kfd::GttAllocation;
using light_rocr::transport::kfd::UserSignal;

constexpr uint64_t kOutputOffset = 0;
constexpr uint64_t kKernargOffset = 64;

int fail_memory(const light_rocr::transport::kfd::MemoryStatus &status) {
  std::cerr << "memory_error="
            << light_rocr::transport::kfd::memory_error_name(status.error)
            << '\n';
  std::cerr << "system_error=" << status.system_error << '\n';
  std::cerr << "message=" << status.message << '\n';
  return 1;
}

int fail_queue(const light_rocr::transport::kfd::AqlQueueStatus &status) {
  std::cerr << "queue_error="
            << light_rocr::transport::kfd::aql_queue_error_name(status.error)
            << '\n';
  std::cerr << "system_error=" << status.system_error << '\n';
  std::cerr << "message=" << status.message << '\n';
  return 1;
}

int fail_signal(const light_rocr::transport::kfd::UserSignalStatus &status) {
  std::cerr << "signal_error="
            << light_rocr::transport::kfd::user_signal_error_name(status.error)
            << '\n';
  std::cerr << "system_error=" << status.system_error << '\n';
  std::cerr << "message=" << status.message << '\n';
  return 1;
}

int cleanup(AqlQueue &queue, UserSignal &signal, GttAllocation &data,
            GttAllocation &executable) {
  const auto queue_status = queue.release();
  if (!queue_status) {
    fail_queue(queue_status);
    std::cerr.flush();
    std::cout.flush();
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
  return 0;
}

} // namespace

int main() {
  namespace fixed = light_rocr::tools::fixed_gfx1101_store;

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

  auto executable = opened.session.allocate_executable_gtt(
      node, light_rocr::transport::kfd::kMemoryPageSize);
  if (!executable) {
    return fail_memory(executable.status);
  }
  if (!fixed::write_image(executable.allocation.host_address(),
                          static_cast<size_t>(executable.allocation.size()))) {
    std::cerr << "fixed kernel image does not fit executable GTT\n";
    return 1;
  }

  auto data = opened.session.allocate_gtt(
      node, light_rocr::transport::kfd::kMemoryPageSize);
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

  auto signal = opened.session.create_user_signal(node, 1);
  if (!signal) {
    return fail_signal(signal.status);
  }
  auto queue = opened.session.create_aql_queue(
      node, light_rocr::transport::kfd::kAqlRingDefaultSize);
  if (!queue) {
    return fail_queue(queue.status);
  }

  lrrt_internal::AqlKernelDispatchParameters parameters;
  parameters.grid_size_x = 1;
  parameters.grid_size_y = 1;
  parameters.grid_size_z = 1;
  parameters.workgroup_size_x = 1;
  parameters.workgroup_size_y = 1;
  parameters.workgroup_size_z = 1;
  parameters.kernel_object = executable.allocation.gpu_address();
  parameters.kernarg_address = data.allocation.gpu_address() + kKernargOffset;
  parameters.completion_signal = signal.signal.gpu_handle();

  const auto submitted = lrrt_internal::submit_aql_kernel_dispatch(
      lrrt_internal::light_rocr_kfd_producer_ops(&queue.queue), parameters, {});
  if (!submitted) {
    std::cerr << "AQL submission failed with error "
              << static_cast<int>(submitted.error) << '\n';
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

  const bool correct = waited && observed_output == fixed::kExpectedValue &&
                       observed_write_index == submitted.packet_id + 1 &&
                       observed_read_index == observed_write_index;
  if (!correct) {
    std::cerr << "direct KFD dispatch did not complete correctly\n";
    std::cerr << "packet_id=" << submitted.packet_id << '\n';
    std::cerr << "read_index=" << observed_read_index << '\n';
    std::cerr << "write_index=" << observed_write_index << '\n';
    std::cerr << "signal_value=" << waited.observed_value << '\n';
    std::cerr << "output=0x" << std::hex << observed_output << '\n';
  }

  const int cleanup_status = cleanup(queue.queue, signal.signal,
                                     data.allocation, executable.allocation);
  if (correct && cleanup_status == 0) {
    std::cout << "direct KFD fixed dispatch: ok\n";
  }
  return correct && cleanup_status == 0 ? 0 : 1;
}
