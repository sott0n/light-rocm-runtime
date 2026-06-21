#include "lrrt/lrrt.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>

#ifndef LRRT_LAUNCH_BENCHMARK_HSACO
#define LRRT_LAUNCH_BENCHMARK_HSACO "launch_benchmark_kernel.hsaco"
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct LaunchArgs {
  int32_t token;
};

struct Colors {
  const char *title;
  const char *label;
  const char *time;
  const char *throughput;
  const char *reset;
};

Colors output_colors() {
  const char *term = getenv("TERM");
  bool enabled = isatty(fileno(stdout)) && getenv("NO_COLOR") == nullptr &&
                 (!term || strcmp(term, "dumb") != 0);
  if (!enabled) {
    return {"", "", "", "", ""};
  }
  return {"\033[1;36m", "\033[1m", "\033[36m", "\033[1;32m", "\033[0m"};
}

std::vector<unsigned char> read_file(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("failed to open benchmark HSACO");
  }
  std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("benchmark HSACO is empty");
  }
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error("failed to read benchmark HSACO");
  }
  return data;
}

uint32_t parse_iterations(int argc, char **argv) {
  if (argc > 2) {
    throw std::invalid_argument(
        "usage: lrrt_launch_overhead_benchmark [count]");
  }
  if (argc == 1) {
    return 10000;
  }
  char *end = nullptr;
  unsigned long value = strtoul(argv[1], &end, 10);
  if (!end || *end != '\0' || value == 0 ||
      value > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("benchmark count must be a positive uint32");
  }
  return static_cast<uint32_t>(value);
}

double elapsed_ns(Clock::time_point begin, Clock::time_point end) {
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count());
}

} // namespace

int main(int argc, char **argv) {
  try {
    const uint32_t iterations = parse_iterations(argc, argv);
    const uint32_t burst_size = std::min(iterations, 512u);
    const uint32_t round_trip_iterations = std::min(iterations, 1000u);
    const uint32_t warmup_iterations = 1024;

    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      fprintf(stderr, "launch_overhead: no GPU devices\n");
      return 1;
    }
    lrrt::Device device = runtime.open_device(0);
    std::vector<unsigned char> hsaco = read_file(LRRT_LAUNCH_BENCHMARK_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel("launch_benchmark_kernel");
    const lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
    const LaunchArgs args = {0};

    for (uint32_t i = 0; i < warmup_iterations; ++i) {
      lrrt::launch(kernel, config, args);
    }
    device.synchronize();

    auto begin = Clock::now();
    for (uint32_t i = 0; i < round_trip_iterations; ++i) {
      device.synchronize();
    }
    auto end = Clock::now();
    double empty_sync_ns =
        elapsed_ns(begin, end) / static_cast<double>(round_trip_iterations);

    begin = Clock::now();
    for (uint32_t i = 0; i < burst_size; ++i) {
      lrrt::launch(kernel, config, args);
    }
    end = Clock::now();
    double enqueue_ns =
        elapsed_ns(begin, end) / static_cast<double>(burst_size);
    device.synchronize();

    lrrt::Event device_start(device);
    lrrt::Event device_end(device);
    device_start.record();
    for (uint32_t i = 0; i < burst_size; ++i) {
      lrrt::launch(kernel, config, args);
    }
    device_end.record();
    device_end.synchronize();
    device_start.synchronize();
    double device_ns =
        static_cast<double>(lrrt::elapsed_time_ns(device_start, device_end)) /
        static_cast<double>(burst_size);
    device.synchronize();

    begin = Clock::now();
    for (uint32_t i = 0; i < iterations; ++i) {
      lrrt::launch(kernel, config, args);
    }
    device.synchronize();
    end = Clock::now();
    double batch_ns = elapsed_ns(begin, end) / static_cast<double>(iterations);

    begin = Clock::now();
    for (uint32_t i = 0; i < round_trip_iterations; ++i) {
      lrrt::launch(kernel, config, args);
      device.synchronize();
    }
    end = Clock::now();
    double round_trip_ns =
        elapsed_ns(begin, end) / static_cast<double>(round_trip_iterations);

    const Colors colors = output_colors();
    printf("\n%sLRRT Launch Overhead Benchmark%s\n", colors.title,
           colors.reset);
    printf("%s==============================%s\n", colors.title, colors.reset);
    printf("Device index:             %u\n", device.index());
    printf("Sustained iterations:     %u\n", iterations);
    printf("Warm-up iterations:       %u\n", warmup_iterations);
    printf("Burst iterations:         %u\n", burst_size);
    printf("Round-trip iterations:    %u\n", round_trip_iterations);
    printf("\n");
    printf("%s%-28s %12s  %s%s\n", colors.label, "Metric", "Time",
           "Measurement", colors.reset);
    printf("%-28s %12s  %s\n", "----------------------------", "------------",
           "--------------------------------");
    printf("%-28s %s%9.3f us%s  %s\n", "Idle synchronize", colors.time,
           empty_sync_ns / 1.0e3, colors.reset, "No queued work");
    printf("%-28s %s%9.3f us%s  %s\n", "Host enqueue", colors.time,
           enqueue_ns / 1.0e3, colors.reset, "Burst, final sync excluded");
    printf("%-28s %s%9.3f us%s  %s\n", "Device batch interval", colors.time,
           device_ns / 1.0e3, colors.reset, "HSA event interval per launch");
    printf("%-28s %s%9.3f us%s  %s\n", "Submit and synchronize", colors.time,
           batch_ns / 1.0e3, colors.reset, "One sync after the full batch");
    printf("%-28s %s%9.3f us%s  %s\n", "Launch round trip", colors.time,
           round_trip_ns / 1.0e3, colors.reset, "One sync after every launch");
    printf("\n%sSustained throughput: %.0f launches/s%s\n\n", colors.throughput,
           1.0e9 / batch_ns, colors.reset);
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
