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

#ifndef LRRT_MUTEX_CONTENTION_HSACO
#define LRRT_MUTEX_CONTENTION_HSACO "mutex_contention_kernel.hsaco"
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t kBaselineProbes = 10000;
constexpr uint32_t kMaximumBackpressureLaunches = 2048;

struct WaitArgs {
  unsigned long long iterations;
  unsigned long long *output;
};

struct Measurement {
  const char *name;
  uint64_t wait_ns;
  uint64_t probe_call_ns;
  uint64_t probe_window_ns;
  uint64_t maximum_probe_ns;
  uint64_t probe_count;
};

std::vector<unsigned char> read_file(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("failed to open mutex-contention HSACO");
  }
  std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("mutex-contention HSACO is empty");
  }
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error("failed to read mutex-contention HSACO");
  }
  return data;
}

uint64_t parse_wait_iterations(int argc, char **argv) {
  if (argc > 2) {
    throw std::invalid_argument(
        "usage: lrrt_mutex_contention_benchmark [wait-iterations]");
  }
  if (argc == 1) {
    return 50000;
  }

  char *end = nullptr;
  unsigned long long value = strtoull(argv[1], &end, 10);
  if (!end || *end != '\0' || value == 0) {
    throw std::invalid_argument("wait iterations must be a positive uint64");
  }
  return value;
}

uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count());
}

template <typename Predicate>
Measurement probe_while(lrrt::Device device, const char *name, uint64_t wait_ns,
                        Predicate active) {
  Measurement measurement{name, wait_ns, 0, 0, 0, 0};
  const auto probe_begin = Clock::now();
  while (active()) {
    lr_memory_stats_t stats{};
    const auto begin = Clock::now();
    lrrt::check(lr_get_memory_stats(device.get(), &stats),
                "lr_get_memory_stats probe");
    const uint64_t duration = elapsed_ns(begin, Clock::now());
    measurement.probe_call_ns += duration;
    measurement.maximum_probe_ns =
        std::max(measurement.maximum_probe_ns, duration);
    ++measurement.probe_count;
  }
  measurement.probe_window_ns = elapsed_ns(probe_begin, Clock::now());
  return measurement;
}

template <typename Operation>
Measurement measure_wait(lrrt::Device device, const char *name,
                         Operation operation) {
  std::atomic<bool> started{false};
  std::atomic<bool> completed{false};
  std::atomic<lr_status_t> status{LR_ERROR_RUNTIME};
  std::atomic<uint64_t> wait_ns{0};

  std::thread worker([&] {
    started.store(true, std::memory_order_release);
    const auto begin = Clock::now();
    status.store(operation(), std::memory_order_relaxed);
    wait_ns.store(elapsed_ns(begin, Clock::now()), std::memory_order_relaxed);
    completed.store(true, std::memory_order_release);
  });

  while (!started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  Measurement measurement = probe_while(device, name, 0, [&] {
    return !completed.load(std::memory_order_acquire);
  });
  worker.join();
  lrrt::check(status.load(std::memory_order_relaxed), name);
  measurement.wait_ns = wait_ns.load(std::memory_order_relaxed);
  return measurement;
}

Measurement measure_backpressure(lrrt::Device device, lrrt::Queue &queue,
                                 const lrrt::Kernel &noop_kernel,
                                 const lr_launch_config_t &config) {
  std::atomic<uint32_t> started{0};
  std::atomic<uint32_t> completed{0};
  std::atomic<bool> stop{false};
  std::atomic<lr_status_t> status{LR_SUCCESS};
  std::array<uint64_t, kMaximumBackpressureLaunches + 1> call_ns{};
  const int32_t noop_args = 0;

  std::thread submitter([&] {
    for (uint32_t i = 1; i <= kMaximumBackpressureLaunches; ++i) {
      if (stop.load(std::memory_order_acquire)) {
        break;
      }
      started.store(i, std::memory_order_release);
      const auto begin = Clock::now();
      lr_status_t launch_status =
          lr_launch_on_queue(queue.get(), noop_kernel.get(), &config,
                             &noop_args, sizeof(noop_args));
      call_ns[i] = elapsed_ns(begin, Clock::now());
      status.store(launch_status, std::memory_order_relaxed);
      completed.store(i, std::memory_order_release);
      if (launch_status != LR_SUCCESS) {
        break;
      }
    }
  });

  uint32_t blocked_call = 0;
  while (blocked_call == 0) {
    const uint32_t current_started = started.load(std::memory_order_acquire);
    const uint32_t current_completed =
        completed.load(std::memory_order_acquire);
    if (current_started > current_completed) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if (started.load(std::memory_order_acquire) == current_started &&
          completed.load(std::memory_order_acquire) == current_completed) {
        blocked_call = current_started;
      }
    } else if (current_completed == kMaximumBackpressureLaunches) {
      break;
    } else {
      std::this_thread::yield();
    }
  }

  if (blocked_call == 0) {
    stop.store(true, std::memory_order_release);
    submitter.join();
    throw std::runtime_error(
        "queue backpressure was not observed; increase wait iterations");
  }

  Measurement measurement = probe_while(device, "Queue backpressure", 0, [&] {
    return completed.load(std::memory_order_acquire) < blocked_call;
  });
  measurement.wait_ns = call_ns[blocked_call];
  stop.store(true, std::memory_order_release);
  submitter.join();
  lrrt::check(status.load(std::memory_order_relaxed),
              "queue backpressure launch");
  queue.synchronize();
  return measurement;
}

void print_measurement(const Measurement &measurement) {
  const double average_probe_ns =
      measurement.probe_count == 0
          ? 0.0
          : static_cast<double>(measurement.probe_call_ns) /
                static_cast<double>(measurement.probe_count);
  const double probe_rate =
      measurement.probe_window_ns == 0
          ? 0.0
          : static_cast<double>(measurement.probe_count) * 1.0e9 /
                static_cast<double>(measurement.probe_window_ns);
  std::printf("%-22s %10.3f %10llu %12.3f %12.3f %12.0f\n", measurement.name,
              static_cast<double>(measurement.wait_ns) / 1.0e3,
              static_cast<unsigned long long>(measurement.probe_count),
              average_probe_ns / 1.0e3,
              static_cast<double>(measurement.maximum_probe_ns) / 1.0e3,
              probe_rate);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const uint64_t wait_iterations = parse_wait_iterations(argc, argv);
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      std::fprintf(stderr, "mutex_contention_benchmark: no GPU devices\n");
      return 1;
    }
    lrrt::Device device = runtime.open_device(0);
    std::vector<unsigned char> hsaco = read_file(LRRT_MUTEX_CONTENTION_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel wait_kernel = module.kernel("mutex_contention_wait_kernel");
    lrrt::Kernel noop_kernel = module.kernel("mutex_contention_noop_kernel");
    const lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
    lrrt::DeviceBuffer wait_output(device, sizeof(unsigned long long));
    lrrt::DeviceBuffer copy_source(device, sizeof(unsigned long long));
    lrrt::DeviceBuffer copy_destination(device, sizeof(unsigned long long));
    const WaitArgs wait_args = {
        wait_iterations,
        static_cast<unsigned long long *>(wait_output.data()),
    };

    uint64_t baseline_total_ns = 0;
    uint64_t baseline_maximum_ns = 0;
    for (uint32_t i = 0; i < kBaselineProbes; ++i) {
      lr_memory_stats_t stats{};
      const auto begin = Clock::now();
      lrrt::check(lr_get_memory_stats(device.get(), &stats),
                  "lr_get_memory_stats baseline");
      const uint64_t duration = elapsed_ns(begin, Clock::now());
      baseline_total_ns += duration;
      baseline_maximum_ns = std::max(baseline_maximum_ns, duration);
    }

    std::vector<Measurement> measurements;

    {
      lrrt::Queue queue(device);
      lrrt::launch(queue, wait_kernel, config, wait_args);
      measurements.push_back(measure_wait(device, "Queue synchronize", [&] {
        return lr_queue_synchronize(queue.get());
      }));
    }

    {
      lr_queue_t *queue = nullptr;
      lrrt::check(lr_queue_create(device.get(), &queue), "lr_queue_create");
      lrrt::check(lr_launch_on_queue(queue, wait_kernel.get(), &config,
                                     &wait_args, sizeof(wait_args)),
                  "lr_launch_on_queue");
      measurements.push_back(measure_wait(
          device, "Queue destroy", [&] { return lr_queue_destroy(queue); }));
    }

    {
      lr_module_t *destroyed_module = nullptr;
      lrrt::check(lr_module_load_hsaco(device.get(), hsaco.data(), hsaco.size(),
                                       &destroyed_module),
                  "lr_module_load_hsaco");
      lr_kernel_t *destroyed_module_kernel = nullptr;
      lrrt::check(lr_kernel_get(destroyed_module,
                                "mutex_contention_wait_kernel",
                                &destroyed_module_kernel),
                  "lr_kernel_get");
      lrrt::Queue queue(device);
      lrrt::check(lr_launch_on_queue(queue.get(), destroyed_module_kernel,
                                     &config, &wait_args, sizeof(wait_args)),
                  "lr_launch_on_queue");
      measurements.push_back(measure_wait(device, "Module destroy", [&] {
        return lr_module_destroy(destroyed_module);
      }));
    }

    {
      lrrt::Queue queue(device);
      lrrt::Event event(device);
      lrrt::launch(queue, wait_kernel, config, wait_args);
      event.record(queue);
      measurements.push_back(measure_wait(device, "Event synchronize", [&] {
        return lr_event_synchronize(event.get());
      }));
    }

    {
      lrrt::Queue queue(device);
      lrrt::launch(queue, wait_kernel, config, wait_args);
      measurements.push_back(
          measure_backpressure(device, queue, noop_kernel, config));
    }

    {
      lrrt::Queue queue(device);
      lrrt::launch(queue, wait_kernel, config, wait_args);
      measurements.push_back(measure_wait(device, "Synchronous D2D copy", [&] {
        return lr_memcpy(device.get(), copy_destination.data(),
                         copy_source.data(), sizeof(unsigned long long),
                         LR_MEMCPY_DEVICE_TO_DEVICE);
      }));
    }

    std::printf("\nLRRT Global Mutex Contention Benchmark\n");
    std::printf("======================================\n");
    std::printf("Device index:          %u\n", device.index());
    std::printf("Device name:           %s\n", device.name().c_str());
    std::printf("GPU wait iterations:   %llu\n",
                static_cast<unsigned long long>(wait_iterations));
    std::printf("Probe API:             lr_get_memory_stats\n");
    std::printf("Baseline probe:        %.3f us average, %.3f us maximum\n\n",
                static_cast<double>(baseline_total_ns) /
                    static_cast<double>(kBaselineProbes) / 1.0e3,
                static_cast<double>(baseline_maximum_ns) / 1.0e3);
    std::printf("%-22s %10s %10s %12s %12s %12s\n", "Concurrent wait",
                "Wait us", "Probes", "Avg probe us", "Max probe us",
                "Probes/s");
    std::printf("%-22s %10s %10s %12s %12s %12s\n", "----------------------",
                "----------", "----------", "------------", "------------",
                "------------");
    for (const Measurement &measurement : measurements) {
      print_measurement(measurement);
    }
    std::printf(
        "\nA maximum probe time far below Wait us indicates that the runtime "
        "mutex\nremained available while the other thread waited for GPU "
        "progress.\n");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
