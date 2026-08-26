#include "lrrt/lrrt.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>

#ifndef LRRT_LAUNCH_BENCHMARK_HSACO
#define LRRT_LAUNCH_BENCHMARK_HSACO "launch_benchmark_kernel.hsaco"
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t kDefaultLaunches = 512;
constexpr uint32_t kWarmupLaunches = 512;
constexpr std::array<uint32_t, 4> kThreadCounts = {1, 2, 4, 8};

struct LaunchArgs {
  int32_t token;
};

enum class QueueLayout {
  Shared,
  PerThread,
};

struct Measurement {
  QueueLayout layout;
  uint32_t thread_count;
  uint32_t launch_count;
  uint64_t wall_ns;
  uint64_t average_ns;
  uint64_t p50_ns;
  uint64_t p95_ns;
  uint64_t p99_ns;
};

std::vector<unsigned char> read_file(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("failed to open parallel-launch HSACO");
  }
  std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("parallel-launch HSACO is empty");
  }
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error("failed to read parallel-launch HSACO");
  }
  return data;
}

uint32_t parse_launch_count(int argc, char **argv) {
  if (argc > 2) {
    throw std::invalid_argument(
        "usage: lrrt_parallel_launch_benchmark [total-launches]");
  }
  if (argc == 1) {
    return kDefaultLaunches;
  }

  char *end = nullptr;
  unsigned long value = strtoul(argv[1], &end, 10);
  if (!end || *end != '\0' || value == 0 ||
      value > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("total launches must be a positive uint32");
  }
  return static_cast<uint32_t>(value);
}

uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count());
}

std::array<uint64_t, 3> calculate_percentiles(std::vector<uint64_t> *samples) {
  std::sort(samples->begin(), samples->end());
  const auto percentile = [samples](size_t percentage) {
    const size_t rank = (samples->size() * percentage + 99) / 100;
    return (*samples)[rank - 1];
  };
  return {percentile(50), percentile(95), percentile(99)};
}

Measurement measure_parallel_launches(lrrt::Device device,
                                      const lrrt::Kernel &kernel,
                                      const lr_launch_config_t &config,
                                      const LaunchArgs &args,
                                      QueueLayout layout, uint32_t thread_count,
                                      uint32_t launch_count) {
  const uint32_t queue_count = layout == QueueLayout::Shared ? 1 : thread_count;
  std::vector<lrrt::Queue> queues;
  queues.reserve(queue_count);
  for (uint32_t i = 0; i < queue_count; ++i) {
    queues.emplace_back(device);
  }

  std::atomic<uint32_t> ready{0};
  std::atomic<bool> start{false};
  std::atomic<lr_status_t> launch_status{LR_SUCCESS};
  std::vector<std::vector<uint64_t>> thread_samples(thread_count);
  std::vector<std::thread> workers;
  workers.reserve(thread_count);

  const uint32_t launches_per_thread = launch_count / thread_count;
  const uint32_t remainder = launch_count % thread_count;
  for (uint32_t thread_index = 0; thread_index < thread_count; ++thread_index) {
    const uint32_t local_launches =
        launches_per_thread + (thread_index < remainder ? 1 : 0);
    thread_samples[thread_index].reserve(local_launches);
    lr_queue_t *queue =
        queues[layout == QueueLayout::Shared ? 0 : thread_index].get();
    workers.emplace_back([&, thread_index, local_launches, queue] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (uint32_t i = 0; i < local_launches; ++i) {
        const auto begin = Clock::now();
        lr_status_t status = lr_launch_on_queue(queue, kernel.get(), &config,
                                                &args, sizeof(args));
        thread_samples[thread_index].push_back(elapsed_ns(begin, Clock::now()));
        if (status != LR_SUCCESS) {
          lr_status_t expected = LR_SUCCESS;
          launch_status.compare_exchange_strong(expected, status);
          break;
        }
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != thread_count) {
    std::this_thread::yield();
  }
  const auto begin = Clock::now();
  start.store(true, std::memory_order_release);
  for (std::thread &worker : workers) {
    worker.join();
  }
  const uint64_t wall_ns = elapsed_ns(begin, Clock::now());

  for (const lrrt::Queue &queue : queues) {
    queue.synchronize();
  }
  lrrt::check(launch_status.load(std::memory_order_relaxed),
              "lr_launch_on_queue parallel");

  std::vector<uint64_t> samples;
  samples.reserve(launch_count);
  uint64_t total_ns = 0;
  for (const std::vector<uint64_t> &local_samples : thread_samples) {
    for (uint64_t sample : local_samples) {
      samples.push_back(sample);
      total_ns += sample;
    }
  }
  if (samples.size() != launch_count) {
    throw std::runtime_error("parallel-launch sample count mismatch");
  }
  const std::array<uint64_t, 3> percentiles = calculate_percentiles(&samples);
  return {layout,
          thread_count,
          launch_count,
          wall_ns,
          total_ns / launch_count,
          percentiles[0],
          percentiles[1],
          percentiles[2]};
}

const char *layout_name(QueueLayout layout) {
  return layout == QueueLayout::Shared ? "Shared" : "Per-thread";
}

void print_measurements(const std::vector<Measurement> &measurements) {
  std::printf("%-11s %7s %9s %12s %8s %8s %10s %10s %10s %10s\n", "Queues",
              "Threads", "Launches", "Launches/s", "Speedup", "Eff.", "Avg us",
              "p50 us", "p95 us", "p99 us");
  std::printf("%-11s %7s %9s %12s %8s %8s %10s %10s %10s %10s\n", "-----------",
              "-------", "---------", "------------", "--------", "--------",
              "----------", "----------", "----------", "----------");

  std::array<double, 2> baseline_throughput{};
  for (const Measurement &measurement : measurements) {
    const size_t layout_index =
        measurement.layout == QueueLayout::Shared ? 0 : 1;
    const double throughput = static_cast<double>(measurement.launch_count) *
                              1.0e9 / static_cast<double>(measurement.wall_ns);
    if (measurement.thread_count == 1) {
      baseline_throughput[layout_index] = throughput;
    }
    const double speedup = throughput / baseline_throughput[layout_index];
    const double efficiency =
        speedup / static_cast<double>(measurement.thread_count) * 100.0;
    std::printf("%-11s %7u %9u %12.0f %7.2fx %7.1f%% %10.3f %10.3f "
                "%10.3f %10.3f\n",
                layout_name(measurement.layout), measurement.thread_count,
                measurement.launch_count, throughput, speedup, efficiency,
                static_cast<double>(measurement.average_ns) / 1.0e3,
                static_cast<double>(measurement.p50_ns) / 1.0e3,
                static_cast<double>(measurement.p95_ns) / 1.0e3,
                static_cast<double>(measurement.p99_ns) / 1.0e3);
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const uint32_t launch_count = parse_launch_count(argc, argv);
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      std::fprintf(stderr, "parallel_launch_benchmark: no GPU devices\n");
      return 1;
    }
    lrrt::Device device = runtime.open_device(0);
    std::vector<unsigned char> hsaco = read_file(LRRT_LAUNCH_BENCHMARK_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel("launch_benchmark_kernel");
    const lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
    const LaunchArgs args = {0};

    {
      lrrt::Queue warmup_queue(device);
      for (uint32_t i = 0; i < kWarmupLaunches; ++i) {
        lrrt::launch(warmup_queue, kernel, config, args);
      }
      warmup_queue.synchronize();
    }

    std::vector<Measurement> measurements;
    for (QueueLayout layout : {QueueLayout::Shared, QueueLayout::PerThread}) {
      for (uint32_t thread_count : kThreadCounts) {
        if (thread_count > launch_count) {
          continue;
        }
        measurements.push_back(measure_parallel_launches(
            device, kernel, config, args, layout, thread_count, launch_count));
      }
    }

    std::printf("\nLRRT Parallel Launch Scalability Benchmark\n");
    std::printf("==========================================\n");
    std::printf("Device index:          %u\n", device.index());
    std::printf("Device name:           %s\n", device.name().c_str());
    std::printf("Launches per row:      %u\n", launch_count);
    std::printf("Synchronization:       excluded from timed region\n\n");
    print_measurements(measurements);
    std::printf(
        "\nSpeedup is relative to the one-thread result for each queue "
        "layout.\nLow scaling with per-thread queues indicates shared runtime "
        "submission\ncontention rather than contention on one queue.\n");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
