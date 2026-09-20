#include "aql_producer.hpp"
#include "light_rocr_kfd_aql_queue.hpp"

#include "light_rocr/loader/code_object.hpp"
#include "light_rocr/runtime/launch.hpp"
#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/kfd/code_cache.hpp"
#include "light_rocr/transport/kfd/executable_image.hpp"
#include "light_rocr/transport/kfd/memory.hpp"
#include "light_rocr/transport/kfd/queue.hpp"
#include "light_rocr/transport/kfd/session.hpp"
#include "light_rocr/transport/kfd/signal.hpp"
#include "light_rocr/transport/kfd/topology.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#ifndef DIRECT_KFD_VECTOR_ADD_HSACO
#error "DIRECT_KFD_VECTOR_ADD_HSACO must name the test code object"
#endif

namespace {

using light_rocr::transport::kfd::AqlQueue;
using light_rocr::transport::kfd::ExecutableImage;
using light_rocr::transport::kfd::GttAllocation;
using light_rocr::transport::kfd::UserSignal;

constexpr size_t kElementCount = 64;
constexpr size_t kArraySize = kElementCount * sizeof(float);
constexpr uint64_t kInputAOffset = 0;
constexpr uint64_t kInputBOffset = kInputAOffset + kArraySize;
constexpr uint64_t kOutputOffset = kInputBOffset + kArraySize;
constexpr uint64_t kKernargOffset = 1024;

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
    std::cerr << "failed to open " << path << '\n';
    return false;
  }
  const std::streampos end = input.tellg();
  if (end <= 0 ||
      static_cast<uintmax_t>(end) > std::numeric_limits<size_t>::max()) {
    std::cerr << "invalid HSACO file size\n";
    return false;
  }
  bytes->resize(static_cast<size_t>(end));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char *>(bytes->data()),
             static_cast<std::streamsize>(bytes->size()));
  return static_cast<bool>(input);
}

int fail_memory(const light_rocr::transport::kfd::MemoryStatus &status) {
  std::cerr << "memory_error="
            << light_rocr::transport::kfd::memory_error_name(status.error)
            << '\n';
  std::cerr << "system_error=" << status.system_error << '\n';
  std::cerr << "message=" << status.message << '\n';
  return 1;
}

int fail_image(
    const light_rocr::transport::kfd::ExecutableImageStatus &status) {
  std::cerr << "image_error="
            << light_rocr::transport::kfd::executable_image_error_name(
                   status.error)
            << '\n';
  std::cerr << "message=" << status.message << '\n';
  return 1;
}

int fail_cache(const light_rocr::transport::kfd::CodeCacheStatus &status) {
  std::cerr << "code_cache_error="
            << light_rocr::transport::kfd::code_cache_error_name(status.error)
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
            ExecutableImage &image) {
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
  const auto image_status = image.release();
  if (!image_status) {
    return fail_memory(image_status);
  }
  return 0;
}

} // namespace

int main() {
  std::vector<uint8_t> hsaco;
  if (!read_file(DIRECT_KFD_VECTOR_ADD_HSACO, &hsaco)) {
    return 1;
  }
  const auto parsed =
      light_rocr::loader::parse_code_object(hsaco.data(), hsaco.size());
  if (!parsed) {
    std::cerr << "parse_error="
              << light_rocr::loader::parse_error_code_name(parsed.error.code)
              << '\n';
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
    std::cerr << "HSACO does not contain vector_add\n";
    return 1;
  }
  const auto &kernel = parsed.code_object.kernels[kernel_index];
  if (kernel.private_segment_size != 0) {
    std::cerr << "direct KFD HSACO fixture unexpectedly requires scratch\n";
    return 1;
  }

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

  auto loaded = light_rocr::transport::kfd::materialize_executable_image(
      opened.session, node, hsaco.data(), hsaco.size(), parsed.code_object);
  if (!loaded) {
    if (loaded.image.owns_allocation()) {
      const auto released = loaded.image.release();
      if (!released) {
        return fail_memory(released);
      }
    }
    return fail_image(loaded.status);
  }

  auto data = opened.session.allocate_gtt(
      node, light_rocr::transport::kfd::kMemoryPageSize);
  if (!data) {
    return fail_memory(data.status);
  }
  if (kernel.kernarg_size >
      light_rocr::transport::kfd::kMemoryPageSize - kKernargOffset) {
    std::cerr << "kernarg fixture does not fit its GTT allocation\n";
    return 1;
  }

  auto *bytes = static_cast<uint8_t *>(data.allocation.host_address());
  auto *input_a = reinterpret_cast<float *>(bytes + kInputAOffset);
  auto *input_b = reinterpret_cast<float *>(bytes + kInputBOffset);
  auto *output = reinterpret_cast<float *>(bytes + kOutputOffset);
  for (size_t index = 0; index < kElementCount; ++index) {
    input_a[index] = static_cast<float>(index);
    input_b[index] = static_cast<float>(index * 2U);
    output[index] = 0.0F;
  }

  VectorAddArguments arguments;
  arguments.input_a_address = data.allocation.gpu_address() + kInputAOffset;
  arguments.input_b_address = data.allocation.gpu_address() + kInputBOffset;
  arguments.output_address = data.allocation.gpu_address() + kOutputOffset;
  arguments.element_count = static_cast<int32_t>(kElementCount);
  const auto kernarg = light_rocr::runtime::materialize_kernarg_buffer(
      kernel, &arguments, sizeof(arguments), bytes + kKernargOffset,
      data.allocation.size() - kKernargOffset,
      data.allocation.gpu_address() + kKernargOffset);
  if (!kernarg) {
    std::cerr << "kernarg_error="
              << light_rocr::runtime::kernarg_buffer_error_name(
                     kernarg.status.error)
              << '\n';
    std::cerr << "message=" << kernarg.status.message << '\n';
    return 1;
  }

  auto blocker = opened.session.create_user_signal(node, 1);
  if (!blocker) {
    return fail_signal(blocker.status);
  }
  auto timeout_queue = opened.session.create_aql_queue(
      node, light_rocr::transport::kfd::kAqlRingDefaultSize);
  if (!timeout_queue) {
    return fail_queue(timeout_queue.status);
  }
  lrrt_internal::AqlBarrierAndParameters blocker_parameters;
  blocker_parameters.dependency_signals[0] = blocker.signal.gpu_handle();
  const auto blocker_submitted = lrrt_internal::submit_aql_barrier_and(
      lrrt_internal::light_rocr_kfd_producer_ops(&timeout_queue.queue),
      blocker_parameters);
  if (!blocker_submitted) {
    std::cerr << "failed to submit cache-timeout blocker\n";
    return 1;
  }
  auto timed_out = light_rocr::transport::kfd::freeze_executable_image(
      opened.session, node, timeout_queue.queue, loaded.image,
      std::chrono::steady_clock::now() + std::chrono::milliseconds(10));
  if (timed_out.operation) {
    const auto released = timed_out.operation.release(timeout_queue.queue);
    if (!released) {
      return fail_cache(released);
    }
  }
  if (timed_out.status.error !=
          light_rocr::transport::kfd::CodeCacheError::WaitForCompletion ||
      timeout_queue.queue) {
    std::cerr << "cache timeout did not destroy its blocked queue\n";
    return 1;
  }
  const auto blocker_status = blocker.signal.release();
  if (!blocker_status) {
    return fail_signal(blocker_status);
  }

  auto signal = opened.session.create_user_signal(node, 1);
  if (!signal) {
    return fail_signal(signal.status);
  }
  auto queue = opened.session.create_aql_queue(
      node, light_rocr::transport::kfd::kAqlRingDefaultSize);
  if (!queue) {
    return fail_queue(queue.status);
  }

  auto mismatched_node = node;
  ++mismatched_node.gpu_id;
  const auto mismatched = light_rocr::transport::kfd::freeze_executable_image(
      opened.session, mismatched_node, queue.queue, loaded.image,
      std::chrono::steady_clock::now());
  if (mismatched.status.error !=
          light_rocr::transport::kfd::CodeCacheError::InvalidArgument ||
      mismatched.operation) {
    std::cerr << "cross-node code-cache resources were accepted\n";
    return 1;
  }

  auto frozen = light_rocr::transport::kfd::freeze_executable_image(
      opened.session, node, queue.queue, loaded.image,
      std::chrono::steady_clock::now() + std::chrono::seconds(2));
  if (!frozen) {
    if (frozen.operation) {
      const auto released = frozen.operation.release(queue.queue);
      if (!released) {
        return fail_cache(released);
      }
    }
    return fail_cache(frozen.status);
  }
  if (frozen.operation) {
    std::cerr << "successful code-cache invalidation retained resources\n";
    return 1;
  }

  lrrt_internal::AqlKernelDispatchParameters parameters;
  parameters.grid_size_x = static_cast<uint32_t>(kElementCount);
  parameters.grid_size_y = 1;
  parameters.grid_size_z = 1;
  parameters.workgroup_size_x = static_cast<uint16_t>(kElementCount);
  parameters.workgroup_size_y = 1;
  parameters.workgroup_size_z = 1;
  parameters.group_segment_size = kernel.group_segment_size;
  parameters.private_segment_size = kernel.private_segment_size;
  parameters.kernel_object =
      loaded.image.kernels()[kernel_index].descriptor_gpu_address;
  parameters.kernarg_address = kernarg.buffer.gpu_address();
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
  for (size_t index = 0; waited && index < kElementCount; ++index) {
    if (output[index] != input_a[index] + input_b[index]) {
      output_matches = false;
      std::cerr << "vector_add mismatch at index " << index << '\n';
      break;
    }
  }
  const bool correct = waited && output_matches &&
                       observed_write_index == submitted.packet_id + 1 &&
                       observed_read_index == observed_write_index;
  if (!correct) {
    std::cerr << "direct KFD HSACO dispatch did not complete correctly\n";
    std::cerr << "packet_id=" << submitted.packet_id << '\n';
    std::cerr << "read_index=" << observed_read_index << '\n';
    std::cerr << "write_index=" << observed_write_index << '\n';
    std::cerr << "signal_value=" << waited.observed_value << '\n';
  }

  const int cleanup_status =
      cleanup(queue.queue, signal.signal, data.allocation, loaded.image);
  if (correct && cleanup_status == 0) {
    std::cout << "direct KFD HSACO vector add: ok\n";
  }
  return correct && cleanup_status == 0 ? 0 : 1;
}
