#include "lrrt/lrrt.hpp"

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

#ifndef LRRT_ASYNC_COPY_LAUNCH_BENCHMARK_HSACO
#define LRRT_ASYNC_COPY_LAUNCH_BENCHMARK_HSACO                                 \
  "async_copy_launch_benchmark_kernel.hsaco"
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct KernelArgs {
  const float *input;
  float *output;
  float alpha;
  int32_t index;
};

struct Colors {
  const char *title;
  const char *label;
  const char *time;
  const char *speedup;
  const char *reset;
};

struct Measurements {
  double submission_ns;
  double round_trip_ns;
};

struct Comparison {
  Measurements host_wait;
  Measurements device_dependency;
};

Colors output_colors() {
  const char *term = getenv("TERM");
  bool enabled = isatty(fileno(stdout)) && getenv("NO_COLOR") == nullptr &&
                 (!term || strcmp(term, "dumb") != 0);
  if (!enabled) {
    return {"", "", "", "", ""};
  }
  return {"\033[1;32m", "\033[1m", "\033[32m", "\033[1;32m", "\033[0m"};
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
        "usage: lrrt_async_copy_launch_benchmark [count]");
  }
  if (argc == 1) {
    return 100;
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

Measurements measure_once(lrrt::Device device, lrrt::DeviceBuffer &staging,
                          const lrrt::DeviceBuffer &source,
                          const lrrt::Kernel &kernel,
                          const lr_launch_config_t &config,
                          const KernelArgs &args, lrrt::Event &copy_complete,
                          bool host_wait) {
  auto round_trip_begin = Clock::now();
  lrrt::copy_device_to_device_async(staging, source, staging.size(),
                                    copy_complete);
  auto submission_begin = Clock::now();
  if (host_wait) {
    copy_complete.synchronize();
  }
  lrrt::launch(kernel, config, args);
  auto submission_end = Clock::now();
  device.synchronize();
  auto round_trip_end = Clock::now();
  return {elapsed_ns(submission_begin, submission_end),
          elapsed_ns(round_trip_begin, round_trip_end)};
}

Comparison compare_paths(lrrt::Device device, lrrt::DeviceBuffer &staging,
                         const lrrt::DeviceBuffer &source,
                         const lrrt::Kernel &kernel,
                         const lr_launch_config_t &config,
                         const KernelArgs &args, uint32_t iterations) {
  lrrt::Event host_wait_event(device);
  lrrt::Event device_dependency_event(device);
  Comparison totals{};

  auto add = [](Measurements *total, Measurements sample) {
    total->submission_ns += sample.submission_ns;
    total->round_trip_ns += sample.round_trip_ns;
  };
  auto run_host_wait = [&] {
    add(&totals.host_wait, measure_once(device, staging, source, kernel, config,
                                        args, host_wait_event, true));
  };
  auto run_device_dependency = [&] {
    add(&totals.device_dependency,
        measure_once(device, staging, source, kernel, config, args,
                     device_dependency_event, false));
  };

  for (uint32_t i = 0; i < iterations; ++i) {
    if (i % 2 == 0) {
      run_host_wait();
      run_device_dependency();
    } else {
      run_device_dependency();
      run_host_wait();
    }
  }

  const double count = static_cast<double>(iterations);
  totals.host_wait.submission_ns /= count;
  totals.host_wait.round_trip_ns /= count;
  totals.device_dependency.submission_ns /= count;
  totals.device_dependency.round_trip_ns /= count;
  return totals;
}

Measurements measure_launch_copy_once(
    lrrt::Device device, lrrt::DeviceBuffer &copy_output,
    const lrrt::DeviceBuffer &kernel_output, const lrrt::Kernel &kernel,
    const lr_launch_config_t &config, const KernelArgs &args,
    lrrt::Event &copy_complete, bool host_wait) {
  auto round_trip_begin = Clock::now();
  lrrt::launch(kernel, config, args);
  auto submission_begin = Clock::now();
  if (host_wait) {
    device.synchronize();
  }
  lrrt::copy_device_to_device_async(copy_output, kernel_output, sizeof(float),
                                    copy_complete);
  auto submission_end = Clock::now();
  copy_complete.synchronize();
  auto round_trip_end = Clock::now();
  device.synchronize();
  return {elapsed_ns(submission_begin, submission_end),
          elapsed_ns(round_trip_begin, round_trip_end)};
}

Comparison compare_launch_copy_paths(lrrt::Device device,
                                     lrrt::DeviceBuffer &copy_output,
                                     const lrrt::DeviceBuffer &kernel_output,
                                     const lrrt::Kernel &kernel,
                                     const lr_launch_config_t &config,
                                     const KernelArgs &args,
                                     uint32_t iterations) {
  lrrt::Event host_wait_event(device);
  lrrt::Event device_dependency_event(device);
  Comparison totals{};

  auto add = [](Measurements *total, Measurements sample) {
    total->submission_ns += sample.submission_ns;
    total->round_trip_ns += sample.round_trip_ns;
  };
  auto run_host_wait = [&] {
    add(&totals.host_wait,
        measure_launch_copy_once(device, copy_output, kernel_output, kernel,
                                 config, args, host_wait_event, true));
  };
  auto run_device_dependency = [&] {
    add(&totals.device_dependency,
        measure_launch_copy_once(device, copy_output, kernel_output, kernel,
                                 config, args, device_dependency_event, false));
  };

  for (uint32_t i = 0; i < iterations; ++i) {
    if (i % 2 == 0) {
      run_host_wait();
      run_device_dependency();
    } else {
      run_device_dependency();
      run_host_wait();
    }
  }

  const double count = static_cast<double>(iterations);
  totals.host_wait.submission_ns /= count;
  totals.host_wait.round_trip_ns /= count;
  totals.device_dependency.submission_ns /= count;
  totals.device_dependency.round_trip_ns /= count;
  return totals;
}

void print_comparison(const char *title, const Comparison &comparison,
                      const Colors &colors) {
  double submission_speedup = comparison.host_wait.submission_ns /
                              comparison.device_dependency.submission_ns;
  double round_trip_speedup = comparison.host_wait.round_trip_ns /
                              comparison.device_dependency.round_trip_ns;
  printf("%s%s%s\n", colors.label, title, colors.reset);
  printf("%-24s %14s %14s\n", "Path", "Submission", "End-to-end");
  printf("%-24s %14s %14s\n", "------------------------", "--------------",
         "--------------");
  printf("%-24s %s%11.3f us%s %s%11.3f us%s\n", "Explicit host wait",
         colors.time, comparison.host_wait.submission_ns / 1.0e3, colors.reset,
         colors.time, comparison.host_wait.round_trip_ns / 1.0e3, colors.reset);
  printf("%-24s %s%11.3f us%s %s%11.3f us%s\n", "Device dependency",
         colors.time, comparison.device_dependency.submission_ns / 1.0e3,
         colors.reset, colors.time,
         comparison.device_dependency.round_trip_ns / 1.0e3, colors.reset);
  printf("%sSubmission speedup: %.2fx%s\n", colors.speedup, submission_speedup,
         colors.reset);
  printf("%sEnd-to-end speedup: %.2fx%s\n\n", colors.speedup,
         round_trip_speedup, colors.reset);
}

void verify_result(lrrt::DeviceBuffer &result, float expected) {
  float actual = 0.0f;
  lrrt::copy_to_host(&actual, result, sizeof(actual));
  if (actual != expected) {
    throw std::runtime_error("benchmark result validation failed");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const uint32_t iterations = parse_iterations(argc, argv);
    const uint32_t warmup_iterations = 10;
    const size_t element_count = 1u << 20;
    const size_t copy_bytes = element_count * sizeof(float);
    const float alpha = 2.5f;

    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      fprintf(stderr, "async_copy_launch_benchmark: no GPU devices\n");
      return 1;
    }
    lrrt::Device device = runtime.open_device(0);
    std::vector<float> input(element_count, 1.0f);
    input.back() = 3.0f;
    lrrt::DeviceBuffer source(device, copy_bytes);
    lrrt::DeviceBuffer staging(device, copy_bytes);
    lrrt::DeviceBuffer result(device, sizeof(float));
    lrrt::DeviceBuffer copied_result(device, sizeof(float));
    lrrt::copy_to_device(source, input);

    std::vector<unsigned char> hsaco =
        read_file(LRRT_ASYNC_COPY_LAUNCH_BENCHMARK_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel("async_copy_launch_benchmark_kernel");
    const lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
    const KernelArgs args = {
        static_cast<const float *>(staging.data()),
        static_cast<float *>(result.data()),
        alpha,
        static_cast<int32_t>(element_count - 1),
    };

    compare_paths(device, staging, source, kernel, config, args,
                  warmup_iterations);
    Comparison comparison = compare_paths(device, staging, source, kernel,
                                          config, args, iterations);
    verify_result(result, input.back() * alpha);
    compare_launch_copy_paths(device, copied_result, result, kernel, config,
                              args, warmup_iterations);
    Comparison launch_copy = compare_launch_copy_paths(
        device, copied_result, result, kernel, config, args, iterations);
    verify_result(copied_result, input.back() * alpha);
    const Colors colors = output_colors();

    printf("\n%sLRRT Async Dependency Benchmark%s\n", colors.title,
           colors.reset);
    printf("%s===============================%s\n", colors.title, colors.reset);
    printf("Device index:       %u\n", device.index());
    printf("Device name:        %s\n", device.name().c_str());
    printf("Copy -> launch:     %.1f MiB D2D\n",
           static_cast<double>(copy_bytes) / (1024.0 * 1024.0));
    printf("Launch -> copy:     %zu B D2D\n", sizeof(float));
    printf("Iterations:         %u\n", iterations);
    printf("Warm-up iterations: %u per path\n\n", warmup_iterations);
    print_comparison("Copy -> launch", comparison, colors);
    print_comparison("Launch -> copy", launch_copy, colors);
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
