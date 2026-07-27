#include "lrrt/lrrt.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kKiB = 1024;
constexpr size_t kMiB = 1024 * kKiB;
constexpr size_t kGiB = 1024 * kMiB;

struct Options {
  uint32_t iterations = 0;
  uint32_t warmup_iterations = 2;
  size_t max_size = kGiB;
};

struct Measurement {
  double first_total_ns = 0.0;
  double submit_ns = 0.0;
  double wait_ns = 0.0;
  double total_ns = 0.0;
};

struct Transfer {
  void *dst;
  const void *src;
  lr_memcpy_kind_t kind;
};

using HostMemory = std::unique_ptr<void, decltype(&std::free)>;

double elapsed_ns(Clock::time_point begin, Clock::time_point end) {
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count());
}

uint32_t parse_uint32(const char *text, const char *option) {
  char *end = nullptr;
  unsigned long value = std::strtoul(text, &end, 10);
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

    if (argument == "--iterations") {
      options.iterations =
          parse_uint32(require_value("--iterations"), "--iterations");
    } else if (argument == "--warmup") {
      options.warmup_iterations =
          parse_uint32(require_value("--warmup"), "--warmup");
    } else if (argument == "--max-size-mib") {
      const uint32_t mib =
          parse_uint32(require_value("--max-size-mib"), "--max-size-mib");
      if (mib > kGiB / kMiB) {
        throw std::invalid_argument("--max-size-mib must not exceed 1024");
      }
      options.max_size = static_cast<size_t>(mib) * kMiB;
    } else if (argument == "--help") {
      std::printf(
          "usage: lrrt_pinned_host_transfer_benchmark "
          "[--iterations N] [--warmup N] [--max-size-mib N]\n\n"
          "Without --iterations, each row transfers about 256 MiB, clamped "
          "to 3..1000 iterations.\n");
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  return options;
}

std::vector<size_t> transfer_sizes(size_t max_size) {
  std::vector<size_t> sizes;
  for (size_t size = 4 * kKiB; size <= max_size;) {
    sizes.push_back(size);
    if (size > max_size / 4) {
      break;
    }
    size *= 4;
  }
  if (sizes.empty() || sizes.back() != max_size) {
    sizes.push_back(max_size);
  }
  return sizes;
}

uint32_t iteration_count(size_t size, uint32_t requested) {
  if (requested != 0) {
    return requested;
  }
  constexpr size_t target_bytes = 256 * kMiB;
  const size_t count = std::max<size_t>(1, target_bytes / size);
  return static_cast<uint32_t>(std::clamp<size_t>(count, 3, 1000));
}

Measurement measure_sync(lr_device_t device, const Transfer &transfer,
                         size_t size, uint32_t warmup_iterations,
                         uint32_t iterations) {
  auto run = [&] {
    const auto begin = Clock::now();
    lrrt::check(
        lr_memcpy(device, transfer.dst, transfer.src, size, transfer.kind),
        "lr_memcpy benchmark");
    const auto end = Clock::now();
    return elapsed_ns(begin, end);
  };

  Measurement result;
  result.first_total_ns = run();
  for (uint32_t i = 0; i < warmup_iterations; ++i) {
    run();
  }
  for (uint32_t i = 0; i < iterations; ++i) {
    result.total_ns += run();
  }
  result.total_ns /= static_cast<double>(iterations);
  result.submit_ns = result.total_ns;
  return result;
}

Measurement measure_async(lrrt::Device device, const Transfer &transfer,
                          size_t size, uint32_t warmup_iterations,
                          uint32_t iterations) {
  lrrt::Event complete(device);
  auto run = [&] {
    const auto begin = Clock::now();
    lrrt::check(lr_memcpy_async(device.get(), transfer.dst, transfer.src, size,
                                transfer.kind, complete.get()),
                "lr_memcpy_async benchmark");
    const auto submitted = Clock::now();
    complete.synchronize();
    const auto completed = Clock::now();
    return Measurement{
        elapsed_ns(begin, completed), elapsed_ns(begin, submitted),
        elapsed_ns(submitted, completed), elapsed_ns(begin, completed)};
  };

  Measurement result;
  result.first_total_ns = run().total_ns;
  for (uint32_t i = 0; i < warmup_iterations; ++i) {
    run();
  }
  for (uint32_t i = 0; i < iterations; ++i) {
    const Measurement sample = run();
    result.submit_ns += sample.submit_ns;
    result.wait_ns += sample.wait_ns;
    result.total_ns += sample.total_ns;
  }
  const double count = static_cast<double>(iterations);
  result.submit_ns /= count;
  result.wait_ns /= count;
  result.total_ns /= count;
  return result;
}

double gib_per_second(size_t size, double nanoseconds) {
  return (static_cast<double>(size) / static_cast<double>(kGiB)) /
         (nanoseconds / 1.0e9);
}

std::string format_size(size_t size) {
  char text[32] = {};
  if (size >= kGiB) {
    std::snprintf(text, sizeof(text), "%.2f GiB",
                  static_cast<double>(size) / static_cast<double>(kGiB));
  } else if (size >= kMiB) {
    std::snprintf(text, sizeof(text), "%.2f MiB",
                  static_cast<double>(size) / static_cast<double>(kMiB));
  } else {
    std::snprintf(text, sizeof(text), "%.0f KiB",
                  static_cast<double>(size) / static_cast<double>(kKiB));
  }
  return text;
}

void print_measurement(const char *direction, const char *memory,
                       const char *mode, size_t size, uint32_t iterations,
                       const Measurement &measurement) {
  std::printf("%-3s %-8s %-5s %10s %6u %11.2f %11.2f %11.2f %11.2f %9.2f\n",
              direction, memory, mode, format_size(size).c_str(), iterations,
              measurement.first_total_ns / 1.0e3, measurement.submit_ns / 1.0e3,
              measurement.wait_ns / 1.0e3, measurement.total_ns / 1.0e3,
              gib_per_second(size, measurement.total_ns));
}

void seed_device(lrrt::Device device, lrrt::DeviceBuffer &buffer,
                 const void *host, size_t size) {
  lrrt::check(lr_memcpy(device.get(), buffer.data(), host, size,
                        LR_MEMCPY_HOST_TO_DEVICE),
              "seed device buffer");
}

void verify_h2d(lrrt::Device device, const lrrt::DeviceBuffer &buffer,
                size_t size, unsigned char expected) {
  unsigned char first = 0;
  unsigned char last = 0;
  const auto *device_bytes = static_cast<const unsigned char *>(buffer.data());
  lrrt::check(lr_memcpy(device.get(), &first, device_bytes, 1,
                        LR_MEMCPY_DEVICE_TO_HOST),
              "verify H2D first byte");
  lrrt::check(lr_memcpy(device.get(), &last, device_bytes + size - 1, 1,
                        LR_MEMCPY_DEVICE_TO_HOST),
              "verify H2D last byte");
  if (first != expected || last != expected) {
    throw std::runtime_error("H2D benchmark validation failed");
  }
}

void verify_d2h(const void *host, size_t size, unsigned char expected) {
  const auto *bytes = static_cast<const unsigned char *>(host);
  if (bytes[0] != expected || bytes[size - 1] != expected) {
    throw std::runtime_error("D2H benchmark validation failed");
  }
}

void run_host_memory(lrrt::Device device, lrrt::DeviceBuffer &device_buffer,
                     void *host, size_t size, const char *memory,
                     const Options &options) {
  constexpr unsigned char pattern = 0xa5;
  const uint32_t iterations = iteration_count(size, options.iterations);
  std::memset(host, pattern, size);

  const Transfer h2d = {device_buffer.data(), host, LR_MEMCPY_HOST_TO_DEVICE};
  Measurement h2d_sync = measure_sync(device.get(), h2d, size,
                                      options.warmup_iterations, iterations);
  verify_h2d(device, device_buffer, size, pattern);
  Measurement h2d_async =
      measure_async(device, h2d, size, options.warmup_iterations, iterations);
  verify_h2d(device, device_buffer, size, pattern);

  seed_device(device, device_buffer, host, size);
  std::memset(host, 0, size);
  const Transfer d2h = {host, device_buffer.data(), LR_MEMCPY_DEVICE_TO_HOST};
  Measurement d2h_sync = measure_sync(device.get(), d2h, size,
                                      options.warmup_iterations, iterations);
  verify_d2h(host, size, pattern);
  std::memset(host, 0, size);
  Measurement d2h_async =
      measure_async(device, d2h, size, options.warmup_iterations, iterations);
  verify_d2h(host, size, pattern);

  print_measurement("H2D", memory, "sync", size, iterations, h2d_sync);
  print_measurement("H2D", memory, "async", size, iterations, h2d_async);
  print_measurement("D2H", memory, "sync", size, iterations, d2h_sync);
  print_measurement("D2H", memory, "async", size, iterations, d2h_async);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      std::fprintf(stderr, "pinned_host_transfer_benchmark: no GPU devices\n");
      return 1;
    }
    lrrt::Device device = runtime.open_device(0);

    std::printf("\nLRRT Pageable vs Pinned Host Transfer Benchmark\n");
    std::printf("==============================================\n");
    std::printf("Device index:       %u\n", device.index());
    std::printf("Device name:        %s\n", device.name().c_str());
    std::printf("Maximum size:       %s\n",
                format_size(options.max_size).c_str());
    std::printf("Warm-up iterations: %u per row\n", options.warmup_iterations);
    if (options.iterations == 0) {
      std::printf("Measured iterations: adaptive (256 MiB target, 3..1000)\n");
    } else {
      std::printf("Measured iterations: %u per row\n", options.iterations);
    }
    std::printf("\n%-3s %-8s %-5s %10s %6s %11s %11s %11s %11s %9s\n", "Dir",
                "Memory", "Mode", "Size", "Count", "First us", "Call us",
                "Wait us", "Total us", "GiB/s");
    std::printf("%-3s %-8s %-5s %10s %6s %11s %11s %11s %11s %9s\n", "---",
                "--------", "-----", "----------", "------", "-----------",
                "-----------", "-----------", "-----------", "---------");

    for (size_t size : transfer_sizes(options.max_size)) {
      lrrt::DeviceBuffer device_buffer(device, size);

      HostMemory pageable(std::malloc(size), &std::free);
      if (!pageable) {
        throw std::bad_alloc();
      }
      run_host_memory(device, device_buffer, pageable.get(), size, "pageable",
                      options);
      pageable.reset();

      lrrt::PinnedHostBuffer pinned(device, size);
      run_host_memory(device, device_buffer, pinned.data(), size, "pinned",
                      options);
      std::printf("\n");
    }

    std::printf(
        "First us is the first call for that path. Call us is host API time; "
        "Wait us is event wait time.\n"
        "For synchronous copies Call us equals Total us. Bandwidth uses "
        "steady-state Total us.\n");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
