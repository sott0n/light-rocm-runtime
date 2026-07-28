#include "lrrt/lrrt.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef LRRT_DOUBLE_BUFFER_PIPELINE_HSACO
#define LRRT_DOUBLE_BUFFER_PIPELINE_HSACO "double_buffer_pipeline_kernel.hsaco"
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kMiB = 1024 * 1024;

struct Options {
  uint32_t chunks = 50;
  uint32_t warmup_chunks = 4;
  uint32_t chunk_size_mib = 4;
  uint32_t compute_rounds = 64;
};

struct KernelArgs {
  uint32_t *data;
  uint32_t count;
  uint32_t rounds;
};

struct Measurement {
  double total_ms;
  double prepare_ms;
};

struct StageProfile {
  double wall_ms = 0.0;
  double prepare_ms = 0.0;
  double host_wait_ms = 0.0;
  double copy_api_ms = 0.0;
  double queue_api_ms = 0.0;
  double h2d_device_ms = 0.0;
  double gpu_stage_ms = 0.0;
};

uint32_t parse_uint32(const char *text, const char *option) {
  char *end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (!end || *end != '\0' || value == 0 ||
      value > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(std::string(option) +
                                " must be a positive uint32");
  }
  return static_cast<uint32_t>(value);
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    auto require_value = [&](const char *option) {
      if (++i >= argc) {
        throw std::invalid_argument(std::string(option) + " requires a value");
      }
      return argv[i];
    };

    if (argument == "--chunks") {
      options.chunks = parse_uint32(require_value("--chunks"), "--chunks");
    } else if (argument == "--warmup-chunks") {
      options.warmup_chunks =
          parse_uint32(require_value("--warmup-chunks"), "--warmup-chunks");
    } else if (argument == "--chunk-size-mib") {
      options.chunk_size_mib =
          parse_uint32(require_value("--chunk-size-mib"), "--chunk-size-mib");
      if (options.chunk_size_mib > 1024) {
        throw std::invalid_argument("--chunk-size-mib must not exceed 1024");
      }
    } else if (argument == "--compute-rounds") {
      options.compute_rounds =
          parse_uint32(require_value("--compute-rounds"), "--compute-rounds");
    } else if (argument == "--help") {
      std::printf("usage: lrrt_double_buffer_pipeline_benchmark "
                  "[--chunks N] [--warmup-chunks N] [--chunk-size-mib N] "
                  "[--compute-rounds N]\n");
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  return options;
}

std::vector<unsigned char> read_file(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("failed to open double-buffer benchmark HSACO");
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("double-buffer benchmark HSACO is empty");
  }
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error("failed to read double-buffer benchmark HSACO");
  }
  return data;
}

double milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

void prepare(void *host, size_t element_count, uint32_t seed) {
  auto *values = static_cast<uint32_t *>(host);
  std::fill(values, values + element_count, seed);
}

uint32_t expected_value(uint32_t seed, uint32_t rounds) {
  uint32_t value = seed;
  for (uint32_t round = 0; round < rounds; ++round) {
    value = value * 1664525u + 1013904223u;
  }
  return value;
}

void verify(lrrt::Device device, const lrrt::DeviceBuffer &buffer,
            uint32_t seed, uint32_t rounds) {
  uint32_t actual = 0;
  lrrt::check(lr_memcpy(device.get(), &actual, buffer.data(), sizeof(actual),
                        LR_MEMCPY_DEVICE_TO_HOST),
              "verify pipeline output");
  const uint32_t expected = expected_value(seed, rounds);
  if (actual != expected) {
    char message[160] = {};
    std::snprintf(message, sizeof(message),
                  "double-buffer benchmark validation failed: seed=%u "
                  "rounds=%u actual=%u expected=%u",
                  seed, rounds, actual, expected);
    throw std::runtime_error(message);
  }
}

Measurement run_sequential(lrrt::Device device, const lrrt::Queue &queue,
                           const lrrt::Kernel &kernel,
                           const lr_launch_config_t &config, size_t bytes,
                           uint32_t chunks, uint32_t rounds,
                           uint32_t first_seed, bool validate) {
  lrrt::PinnedHostBuffer host(device, bytes);
  lrrt::DeviceBuffer device_buffer(device, bytes);
  lrrt::Event copy_complete(device);
  lrrt::Event work_complete(device);
  const size_t element_count = bytes / sizeof(uint32_t);
  double prepare_ms = 0.0;

  const auto begin = Clock::now();
  for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
    const uint32_t seed = first_seed + chunk;
    const auto prepare_begin = Clock::now();
    prepare(host.data(), element_count, seed);
    prepare_ms += milliseconds(prepare_begin, Clock::now());

    lrrt::copy_to_device_async(device_buffer, host.data(), bytes, copy_complete,
                               {});
    const KernelArgs args = {static_cast<uint32_t *>(device_buffer.data()),
                             static_cast<uint32_t>(element_count), rounds};
    lrrt::launch(queue, kernel, config, args, {&copy_complete});
    work_complete.record(queue);
    work_complete.synchronize();
  }
  const double total_ms = milliseconds(begin, Clock::now());

  if (validate) {
    verify(device, device_buffer, first_seed + chunks - 1, rounds);
  }
  return {total_ms, prepare_ms};
}

Measurement run_double_buffered(lrrt::Device device, const lrrt::Queue &queue,
                                const lrrt::Kernel &kernel,
                                const lr_launch_config_t &config, size_t bytes,
                                uint32_t chunks, uint32_t rounds,
                                uint32_t first_seed, bool validate) {
  lrrt::PinnedHostDoubleBuffer pipeline(device, bytes);
  const size_t element_count = bytes / sizeof(uint32_t);
  std::unordered_map<const lrrt::PinnedHostDoubleBuffer::Slot *, uint32_t>
      final_seeds;
  double prepare_ms = 0.0;

  const auto begin = Clock::now();
  for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
    auto &slot = pipeline.acquire();
    const uint32_t seed = first_seed + chunk;
    const auto prepare_begin = Clock::now();
    prepare(slot.host_data(), element_count, seed);
    prepare_ms += milliseconds(prepare_begin, Clock::now());

    pipeline.copy_to_device_async(slot, bytes);
    const KernelArgs args = {
        static_cast<uint32_t *>(slot.device_buffer().data()),
        static_cast<uint32_t>(element_count), rounds};
    lrrt::launch(queue, kernel, config, args, {&slot.copy_complete()});
    pipeline.mark_work_submitted(slot, queue);
    final_seeds[&slot] = seed;
  }
  pipeline.finish();
  const double total_ms = milliseconds(begin, Clock::now());

  if (validate) {
    for (const auto &entry : final_seeds) {
      verify(device, entry.first->device_buffer(), entry.second, rounds);
    }
  }
  return {total_ms, prepare_ms};
}

StageProfile profile_sequential(lrrt::Device device, const lrrt::Queue &queue,
                                const lrrt::Kernel &kernel,
                                const lr_launch_config_t &config, size_t bytes,
                                uint32_t chunks, uint32_t rounds,
                                uint32_t first_seed) {
  lrrt::PinnedHostBuffer host(device, bytes);
  lrrt::DeviceBuffer device_buffer(device, bytes);
  lrrt::Event copy_complete(device);
  lrrt::Event gpu_start(device);
  lrrt::Event work_complete(device);
  const size_t element_count = bytes / sizeof(uint32_t);
  StageProfile profile;

  const auto begin = Clock::now();
  for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
    const auto prepare_begin = Clock::now();
    prepare(host.data(), element_count, first_seed + chunk);
    profile.prepare_ms += milliseconds(prepare_begin, Clock::now());

    const auto copy_api_begin = Clock::now();
    lrrt::copy_to_device_async(device_buffer, host.data(), bytes, copy_complete,
                               {});
    profile.copy_api_ms += milliseconds(copy_api_begin, Clock::now());

    const auto queue_api_begin = Clock::now();
    gpu_start.record(queue);
    const KernelArgs args = {static_cast<uint32_t *>(device_buffer.data()),
                             static_cast<uint32_t>(element_count), rounds};
    lrrt::launch(queue, kernel, config, args, {&copy_complete});
    work_complete.record(queue);
    profile.queue_api_ms += milliseconds(queue_api_begin, Clock::now());

    const auto wait_begin = Clock::now();
    work_complete.synchronize();
    profile.host_wait_ms += milliseconds(wait_begin, Clock::now());
    copy_complete.synchronize();
    gpu_start.synchronize();
    profile.h2d_device_ms +=
        static_cast<double>(lrrt::duration_ns(copy_complete)) / 1.0e6;
    profile.gpu_stage_ms +=
        static_cast<double>(lrrt::elapsed_time_ns(gpu_start, work_complete)) /
        1.0e6;
  }
  profile.wall_ms = milliseconds(begin, Clock::now());
  return profile;
}

StageProfile profile_double_buffered(lrrt::Device device,
                                     const lrrt::Queue &queue,
                                     const lrrt::Kernel &kernel,
                                     const lr_launch_config_t &config,
                                     size_t bytes, uint32_t chunks,
                                     uint32_t rounds, uint32_t first_seed) {
  using Slot = lrrt::PinnedHostDoubleBuffer::Slot;

  lrrt::PinnedHostDoubleBuffer pipeline(device, bytes);
  std::array<lrrt::Event, 2> gpu_starts = {lrrt::Event(device),
                                           lrrt::Event(device)};
  std::unordered_map<const Slot *, size_t> slot_indices;
  std::array<bool, 2> sample_pending = {false, false};
  const size_t element_count = bytes / sizeof(uint32_t);
  StageProfile profile;

  auto slot_index = [&](const Slot &slot) {
    const auto found = slot_indices.find(&slot);
    if (found != slot_indices.end()) {
      return found->second;
    }
    const size_t index = slot_indices.size();
    if (index >= gpu_starts.size()) {
      throw std::runtime_error("double-buffer returned more than two slots");
    }
    slot_indices.emplace(&slot, index);
    return index;
  };

  auto collect = [&](const Slot &slot, size_t index) {
    if (!sample_pending[index]) {
      return;
    }
    slot.copy_complete().synchronize();
    gpu_starts[index].synchronize();
    profile.h2d_device_ms +=
        static_cast<double>(lrrt::duration_ns(slot.copy_complete())) / 1.0e6;
    profile.gpu_stage_ms += static_cast<double>(lrrt::elapsed_time_ns(
                                gpu_starts[index], slot.work_complete())) /
                            1.0e6;
    sample_pending[index] = false;
  };

  const auto begin = Clock::now();
  for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
    const auto wait_begin = Clock::now();
    Slot &slot = pipeline.acquire();
    profile.host_wait_ms += milliseconds(wait_begin, Clock::now());
    const size_t index = slot_index(slot);
    collect(slot, index);

    const auto prepare_begin = Clock::now();
    prepare(slot.host_data(), element_count, first_seed + chunk);
    profile.prepare_ms += milliseconds(prepare_begin, Clock::now());

    const auto copy_api_begin = Clock::now();
    pipeline.copy_to_device_async(slot, bytes);
    profile.copy_api_ms += milliseconds(copy_api_begin, Clock::now());

    const auto queue_api_begin = Clock::now();
    gpu_starts[index].record(queue);
    const KernelArgs args = {
        static_cast<uint32_t *>(slot.device_buffer().data()),
        static_cast<uint32_t>(element_count), rounds};
    lrrt::launch(queue, kernel, config, args, {&slot.copy_complete()});
    pipeline.mark_work_submitted(slot, queue);
    profile.queue_api_ms += milliseconds(queue_api_begin, Clock::now());
    sample_pending[index] = true;
  }

  const auto wait_begin = Clock::now();
  pipeline.finish();
  profile.host_wait_ms += milliseconds(wait_begin, Clock::now());
  for (const auto &entry : slot_indices) {
    collect(*entry.first, entry.second);
  }
  profile.wall_ms = milliseconds(begin, Clock::now());
  return profile;
}

void print_measurement(const char *mode, const Measurement &measurement,
                       size_t bytes, uint32_t chunks) {
  const double gib =
      static_cast<double>(bytes) * chunks / (1024.0 * 1024.0 * 1024.0);
  const double seconds = measurement.total_ms / 1000.0;
  std::printf("%-16s %12.3f %12.3f %12.1f %12.3f\n", mode, measurement.total_ms,
              measurement.prepare_ms, chunks / seconds, gib / seconds);
}

void print_profile(const char *mode, const StageProfile &profile) {
  std::printf("%-16s %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n", mode,
              profile.wall_ms, profile.prepare_ms, profile.host_wait_ms,
              profile.copy_api_ms, profile.queue_api_ms, profile.h2d_device_ms,
              profile.gpu_stage_ms);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      std::fprintf(stderr,
                   "double_buffer_pipeline_benchmark: no GPU devices\n");
      return 1;
    }
    lrrt::Device device = runtime.open_device(0);
    lrrt::Queue queue(device);
    std::vector<unsigned char> hsaco =
        read_file(LRRT_DOUBLE_BUFFER_PIPELINE_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel("double_buffer_pipeline_kernel");

    const size_t bytes = static_cast<size_t>(options.chunk_size_mib) * kMiB;
    const size_t element_count = bytes / sizeof(uint32_t);
    if (element_count > std::numeric_limits<uint32_t>::max()) {
      throw std::invalid_argument("chunk contains too many elements");
    }
    constexpr uint32_t block_size = 256;
    const uint32_t grid_size = static_cast<uint32_t>(
        ((element_count + block_size - 1) / block_size) * block_size);
    const lr_launch_config_t config = {
        {grid_size, 1, 1}, {block_size, 1, 1}, 0};

    run_sequential(device, queue, kernel, config, bytes, options.warmup_chunks,
                   options.compute_rounds, 1, false);
    run_double_buffered(device, queue, kernel, config, bytes,
                        options.warmup_chunks, options.compute_rounds, 1001,
                        false);

    const Measurement sequential =
        run_sequential(device, queue, kernel, config, bytes, options.chunks,
                       options.compute_rounds, 2001, true);
    const Measurement double_buffered =
        run_double_buffered(device, queue, kernel, config, bytes,
                            options.chunks, options.compute_rounds, 3001, true);
    const StageProfile sequential_profile =
        profile_sequential(device, queue, kernel, config, bytes, options.chunks,
                           options.compute_rounds, 4001);
    const StageProfile double_buffered_profile =
        profile_double_buffered(device, queue, kernel, config, bytes,
                                options.chunks, options.compute_rounds, 5001);

    std::printf("\nLRRT Pinned Host Double-buffer Pipeline Benchmark\n");
    std::printf("================================================\n");
    std::printf("Device:          %s\n", device.name().c_str());
    std::printf("Chunks:          %u\n", options.chunks);
    std::printf("Chunk size:      %u MiB\n", options.chunk_size_mib);
    std::printf("Compute rounds:  %u\n", options.compute_rounds);
    std::printf("Warm-up chunks:  %u per mode\n\n", options.warmup_chunks);
    std::printf("%-16s %12s %12s %12s %12s\n", "Mode", "Total ms", "Prepare ms",
                "Chunks/s", "Input GiB/s");
    std::printf("%-16s %12s %12s %12s %12s\n", "----------------",
                "------------", "------------", "------------", "------------");
    print_measurement("sequential", sequential, bytes, options.chunks);
    print_measurement("double-buffered", double_buffered, bytes,
                      options.chunks);
    std::printf("\nEnd-to-end speedup: %.3fx\n",
                sequential.total_ms / double_buffered.total_ms);
    const double saved_ms = sequential.total_ms - double_buffered.total_ms;
    std::printf("Observed wall-time saving: %.3f ms (%.1f%%)\n", saved_ms,
                saved_ms * 100.0 / sequential.total_ms);
    std::printf(
        "Total includes CPU preparation, H2D transfer, GPU work, and final "
        "drain; allocation and validation are excluded.\n");

    std::printf("\nInstrumented stage profile (sum across all chunks)\n");
    std::printf("--------------------------------------------------\n");
    std::printf("%-16s %10s %10s %10s %10s %10s %10s %10s\n", "Mode", "Wall ms",
                "Prepare", "Host wait", "Copy API", "Queue API", "H2D dev",
                "GPU stage");
    std::printf("%-16s %10s %10s %10s %10s %10s %10s %10s\n",
                "----------------", "----------", "----------", "----------",
                "----------", "----------", "----------", "----------");
    print_profile("sequential", sequential_profile);
    print_profile("double-buffered", double_buffered_profile);
    std::printf(
        "H2D dev is copy-engine duration. GPU stage spans the queue marker "
        "before the copy dependency through kernel completion, so it includes "
        "dependency wait plus kernel time.\n");
    std::printf(
        "Host columns are wall-clock sums. Device-stage totals can overlap "
        "each other and CPU work, so all columns need not add up to Wall ms. "
        "The throughput table above is measured separately without profiling "
        "queries.\n");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
