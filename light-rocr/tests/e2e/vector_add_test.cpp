#include "light_rocr/loader/code_object.hpp"
#include "light_rocr/runtime/launch.hpp"
#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/hsakmt/executable_image.hpp"
#include "light_rocr/transport/hsakmt/kernarg.hpp"
#include "light_rocr/transport/hsakmt/memory.hpp"
#include "light_rocr/transport/hsakmt/queue.hpp"
#include "light_rocr/transport/hsakmt/signal.hpp"
#include "light_rocr/transport/hsakmt/status.hpp"
#include "light_rocr/transport/hsakmt/topology.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

using light_rocr::transport::hsakmt::AqlQueue;
using light_rocr::transport::hsakmt::ExecutableImage;
using light_rocr::transport::hsakmt::KernargBuffer;
using light_rocr::transport::hsakmt::MemoryAllocation;
using light_rocr::transport::hsakmt::UserSignal;

constexpr size_t kElementCount = 64;
constexpr size_t kArraySize = kElementCount * sizeof(float);
constexpr uint64_t kInputAOffset = 0;
constexpr uint64_t kInputBOffset = kInputAOffset + kArraySize;
constexpr uint64_t kOutputOffset = kInputBOffset + kArraySize;
constexpr uint64_t kDataSize = kOutputOffset + kArraySize;

static_assert(kDataSize <= light_rocr::transport::hsakmt::kMemoryPageSize);

struct VectorAddArguments {
  uint64_t input_a_address = 0;
  uint64_t input_b_address = 0;
  uint64_t output_address = 0;
  int32_t element_count = 0;
  uint32_t padding = 0;
};

static_assert(sizeof(VectorAddArguments) == 32);
static_assert(offsetof(VectorAddArguments, input_a_address) == 0);
static_assert(offsetof(VectorAddArguments, input_b_address) == 8);
static_assert(offsetof(VectorAddArguments, output_address) == 16);
static_assert(offsetof(VectorAddArguments, element_count) == 24);

bool read_file(const std::string &path, std::vector<uint8_t> *bytes) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    std::cerr << "message=failed to open " << std::quoted(path) << '\n';
    return false;
  }
  const std::streampos end = input.tellg();
  if (end <= 0 ||
      static_cast<uintmax_t>(end) > std::numeric_limits<size_t>::max()) {
    std::cerr << "message=invalid file size for " << std::quoted(path) << '\n';
    return false;
  }
  bytes->resize(static_cast<size_t>(end));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char *>(bytes->data()),
             static_cast<std::streamsize>(bytes->size()));
  if (!input) {
    std::cerr << "message=failed to read " << std::quoted(path) << '\n';
    return false;
  }
  return true;
}

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

int fail_image(
    const light_rocr::transport::hsakmt::ExecutableImageStatus &status) {
  std::cerr << "image_error="
            << light_rocr::transport::hsakmt::executable_image_error_name(
                   status.error)
            << '\n';
  std::cerr << "message=" << status.message << '\n';
  return 1;
}

int fail_kernarg(
    const light_rocr::transport::hsakmt::KernargBufferStatus &status) {
  std::cerr << "kernarg_error="
            << light_rocr::transport::hsakmt::kernarg_buffer_error_name(
                   status.error)
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

int cleanup(AqlQueue &queue, UserSignal &signal, KernargBuffer &kernarg,
            MemoryAllocation &data, ExecutableImage &image) {
  const auto queue_status = queue.release();
  if (!queue_status) {
    fail_queue(queue_status);
    std::cerr.flush();
    std::cout.flush();
    // Do not release memory that may still be referenced by a live queue.
    std::_Exit(1);
  }
  const auto signal_status = signal.release();
  if (!signal_status) {
    return fail_signal(signal_status);
  }
  const auto kernarg_status = kernarg.release();
  if (!kernarg_status) {
    return fail_memory(kernarg_status);
  }
  const auto data_status = data.release();
  if (!data_status) {
    return fail_memory(data_status);
  }
  const auto image_status = image.release();
  if (!image_status) {
    return fail_memory(image_status);
  }
  std::cout << "vector_add_e2e.cleanup=ok\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0] << " PATH_TO_VECTOR_ADD_HSACO [TARGET]\n";
    return 2;
  }
  const std::string target = argc == 3 ? argv[2] : "gfx1101";

  std::vector<uint8_t> bytes;
  if (!read_file(argv[1], &bytes)) {
    return 1;
  }
  const auto parsed =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  if (!parsed) {
    std::cerr << "parse_error="
              << light_rocr::loader::parse_error_code_name(parsed.error.code)
              << '\n';
    std::cerr << "error_offset=0x" << std::hex << parsed.error.offset
              << std::dec << '\n';
    std::cerr << "message=" << parsed.error.message << '\n';
    return 1;
  }

  size_t kernel_index = parsed.code_object.kernels.size();
  for (size_t index = 0; index < parsed.code_object.kernels.size(); ++index) {
    if (parsed.code_object.kernels[index].name == "vector_add") {
      kernel_index = index;
      break;
    }
  }
  if (kernel_index == parsed.code_object.kernels.size()) {
    std::cerr << "message=HSACO does not contain the vector_add kernel\n";
    return 1;
  }
  const auto &kernel = parsed.code_object.kernels[kernel_index];
  if (kernel.private_segment_size == 0) {
    std::cerr << "message=vector_add kernel does not exercise private-segment "
                 "scratch\n";
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
  auto loaded = light_rocr::transport::hsakmt::materialize_executable_image(
      opened.session, node.node_id, bytes.data(), bytes.size(),
      parsed.code_object);
  if (!loaded) {
    return fail_image(loaded.status);
  }

  auto data = opened.session.allocate_gtt(
      node.node_id, light_rocr::transport::hsakmt::kMemoryPageSize);
  if (!data) {
    return fail_memory(data.status);
  }
  if (data.allocation.gpu_address() >
      std::numeric_limits<uint64_t>::max() - (kDataSize - 1U)) {
    std::cerr << "message=data GPU address range overflows uint64\n";
    return 1;
  }

  std::memset(data.allocation.host_address(), 0,
              static_cast<size_t>(data.allocation.size()));
  auto *data_bytes = static_cast<uint8_t *>(data.allocation.host_address());
  auto *input_a = reinterpret_cast<float *>(data_bytes + kInputAOffset);
  auto *input_b = reinterpret_cast<float *>(data_bytes + kInputBOffset);
  auto *output = reinterpret_cast<float *>(data_bytes + kOutputOffset);
  for (size_t index = 0; index < kElementCount; ++index) {
    input_a[index] = static_cast<float>(index);
    input_b[index] = static_cast<float>(index * 2);
  }

  VectorAddArguments arguments;
  arguments.input_a_address = data.allocation.gpu_address() + kInputAOffset;
  arguments.input_b_address = data.allocation.gpu_address() + kInputBOffset;
  arguments.output_address = data.allocation.gpu_address() + kOutputOffset;
  arguments.element_count = static_cast<int32_t>(kElementCount);
  auto kernarg = light_rocr::transport::hsakmt::create_kernarg_buffer(
      opened.session, node.node_id, kernel, &arguments, sizeof(arguments));
  if (!kernarg) {
    return fail_kernarg(kernarg.status);
  }

  auto signal = opened.session.create_user_signal(node.node_id, 1);
  if (!signal) {
    return fail_signal(signal.status);
  }

  light_rocr::runtime::KernelLaunchConfiguration configuration;
  configuration.workgroup_size_x = static_cast<uint16_t>(kElementCount);
  configuration.grid_size_x = static_cast<uint32_t>(kElementCount);
  const auto packet = light_rocr::runtime::make_kernel_launch_packet(
      loaded.image.runtime_image(), kernel_index,
      kernarg.buffer.runtime_buffer(), configuration,
      signal.signal.gpu_handle());
  if (!packet) {
    std::cerr << "launch_error="
              << light_rocr::runtime::kernel_launch_error_name(
                     packet.status.error)
              << '\n';
    if (!packet.status.aql_status) {
      std::cerr << "packet_error="
                << light_rocr::runtime::aql_packet_error_name(
                       packet.status.aql_status.error)
                << '\n';
    }
    std::cerr << "message=" << packet.status.message << '\n';
    return 1;
  }

  auto queue = opened.session.create_aql_queue(
      node, light_rocr::transport::hsakmt::kAqlRingDefaultSize,
      kernel.private_segment_size);
  if (!queue) {
    return fail_queue(queue.status);
  }

  std::cout << "node_id=" << node.node_id << '\n';
  std::cout << "target="
            << light_rocr::runtime::gfx_target_name(node.architecture) << '\n';
  std::cout << "kernel.name=" << std::quoted(kernel.name) << '\n';
  std::cout << "kernel.descriptor_gpu_address=0x" << std::hex
            << loaded.image.kernels()[kernel_index].descriptor_gpu_address
            << '\n';
  std::cout << "kernel.entry_gpu_address=0x"
            << loaded.image.kernels()[kernel_index].code_entry_gpu_address
            << std::dec << '\n';
  std::cout << "kernel.private_segment_size=" << kernel.private_segment_size
            << '\n';
  std::cout << "kernarg.gpu_address=0x" << std::hex
            << kernarg.buffer.gpu_address() << '\n';
  std::cout << "signal.gpu_handle=0x" << signal.signal.gpu_handle() << '\n';
  std::cout << "queue.scratch_gpu_address=0x"
            << queue.queue.scratch_gpu_address() << std::dec << '\n';
  std::cout << "queue.scratch_size=" << queue.queue.scratch_size() << '\n';
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
  const uint64_t observed_write_index = queue.queue.write_index_relaxed();
  uint64_t observed_read_index = queue.queue.read_index_acquire();
  const auto read_index_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  while (observed_read_index != observed_write_index &&
         std::chrono::steady_clock::now() < read_index_deadline) {
    std::this_thread::yield();
    observed_read_index = queue.queue.read_index_acquire();
  }

  bool output_matches = static_cast<bool>(waited);
  size_t mismatch_index = 0;
  float expected_value = 0.0F;
  float observed_value = 0.0F;
  if (waited) {
    for (size_t index = 0; index < kElementCount; ++index) {
      const float expected = input_a[index] + input_b[index];
      if (output[index] != expected) {
        output_matches = false;
        mismatch_index = index;
        expected_value = expected;
        observed_value = output[index];
        break;
      }
    }
  }

  std::cout << "packet.id=" << submitted.packet_id << '\n';
  std::cout << "queue.read_index=" << observed_read_index << '\n';
  std::cout << "queue.write_index=" << observed_write_index << '\n';
  std::cout << "signal.value=" << waited.observed_value << '\n';
  std::cout << "signal.wait=" << (waited ? "satisfied" : "timed_out") << '\n';
  std::cout << "vector_add.result=";
  if (!waited) {
    std::cout << "not_checked\n";
  } else {
    std::cout << (output_matches ? "correct" : "mismatch") << '\n';
  }
  if (waited && !output_matches) {
    std::cout << "vector_add.mismatch_index=" << mismatch_index << '\n';
    std::cout << "vector_add.expected=" << expected_value << '\n';
    std::cout << "vector_add.observed=" << observed_value << '\n';
  }

  const bool correct =
      waited && output_matches && observed_read_index == submitted.write_index;
  if (!correct) {
    std::cerr << "message=vector_add dispatch did not complete correctly\n";
  }
  const int cleanup_status = cleanup(queue.queue, signal.signal, kernarg.buffer,
                                     data.allocation, loaded.image);
  return correct && cleanup_status == 0 ? 0 : 1;
}
