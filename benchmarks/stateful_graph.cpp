#include "executor/stateful_graph/stateful_graph_abi.hpp"
#include "lrrt/lrrt.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef LRRT_STATEFUL_GRAPH_HSACO
#define LRRT_STATEFUL_GRAPH_HSACO "stateful_graph_kernel.hsaco"
#endif

namespace {

namespace sg = lrrt::stateful_graph;
using Clock = std::chrono::steady_clock;

constexpr uint32_t kBlockSize = sg::kWorkerThreads;
constexpr uint32_t kElementsPerWorker = 1024;
constexpr uint32_t kWarmupTrials = 5;
constexpr uint32_t kDefaultWorkers = 4;
constexpr uint32_t kMaximumWorkers = 16;

struct Measurement {
  double device_ns = 0.0;
  double wall_ns = 0.0;
};

std::vector<unsigned char> read_file(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("failed to open stateful-graph HSACO");
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("stateful-graph HSACO is empty");
  }
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error("failed to read stateful-graph HSACO");
  }
  return data;
}

uint32_t parse_u32(const char *text, const char *label) {
  char *end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (!end || *end != '\0' || value == 0 ||
      value > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(std::string(label) +
                                " must be a positive uint32");
  }
  return static_cast<uint32_t>(value);
}

template <typename Setup, typename Submit>
Measurement
measure(lrrt::Device device, uint32_t trials, lrrt::DeviceBuffer &values,
        const std::vector<uint32_t> &zero_values, Setup setup, Submit submit) {
  lrrt::Event start(device);
  lrrt::Event end(device);
  Measurement result;
  for (uint32_t trial = 0; trial < trials; ++trial) {
    lrrt::copy_to_device(values, zero_values);
    setup();
    start.record();
    const auto begin = Clock::now();
    submit();
    end.record();
    end.synchronize();
    start.synchronize();
    result.wall_ns += static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                             begin)
            .count());
    result.device_ns += static_cast<double>(lrrt::elapsed_time_ns(start, end));
  }
  result.device_ns /= static_cast<double>(trials);
  result.wall_ns /= static_cast<double>(trials);
  return result;
}

void verify_values(const std::vector<uint32_t> &values, uint32_t expected,
                   const char *path) {
  for (size_t index = 0; index < values.size(); ++index) {
    if (values[index] != expected) {
      std::fprintf(stderr, "%s value[%zu]: expected %u, got %u\n", path, index,
                   expected, values[index]);
      throw std::runtime_error("stateful-graph result mismatch");
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc > 4) {
      throw std::invalid_argument(
          "usage: lrrt_stateful_graph_benchmark [steps] [trials] [workers]");
    }
    const uint32_t steps = argc >= 2 ? parse_u32(argv[1], "steps") : 64;
    const uint32_t trials = argc >= 3 ? parse_u32(argv[2], "trials") : 20;
    const uint32_t workers =
        argc >= 4 ? parse_u32(argv[3], "workers") : kDefaultWorkers;
    if (steps > std::numeric_limits<uint32_t>::max() / 2u) {
      throw std::invalid_argument("steps is too large");
    }
    if (workers > kMaximumWorkers) {
      throw std::invalid_argument("workers must not exceed 16");
    }

    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      std::fprintf(stderr, "stateful_graph_benchmark: no GPU devices\n");
      return 1;
    }
    lrrt::Device device = runtime.open_device(0);
    lrrt::Module module(device, read_file(LRRT_STATEFUL_GRAPH_HSACO));
    lrrt::Kernel resident = module.kernel("stateful_graph_resident_kernel");
    lrrt::Kernel host_step = module.kernel("stateful_graph_host_step_kernel");

    const uint32_t element_count = workers * kElementsPerWorker;
    std::vector<sg::Task> tasks(workers);
    for (uint32_t worker = 0; worker < workers; ++worker) {
      tasks[worker] = {
          static_cast<uint32_t>(sg::Opcode::increment_u32),
          0,
          worker * kElementsPerWorker,
          (worker + 1u) * kElementsPerWorker,
          1,
          0,
          {0, 0},
      };
    }
    const std::vector<uint32_t> zero_values(element_count, 0);

    lrrt::DeviceBuffer task_buffer(device, tasks.size() * sizeof(sg::Task));
    lrrt::DeviceBuffer resident_values(device,
                                       element_count * sizeof(uint32_t));
    lrrt::DeviceBuffer host_values(device, element_count * sizeof(uint32_t));
    lrrt::DeviceBuffer state_buffer(device, sizeof(sg::State));
    lrrt::copy_to_device(task_buffer, tasks);

    const lr_launch_config_t config = {
        {workers * kBlockSize, 1, 1}, {kBlockSize, 1, 1}, 0};
    const sg::ResidentArgs resident_args = {
        static_cast<sg::State *>(state_buffer.data()),
        static_cast<const sg::Task *>(task_buffer.data()),
        static_cast<uint32_t *>(resident_values.data()),
    };
    const sg::HostStepArgs host_args = {
        static_cast<const sg::Task *>(task_buffer.data()),
        workers,
        0,
        static_cast<uint32_t *>(host_values.data()),
    };

    const auto initialize_state = [&] {
      const sg::State state = {
          sg::kAbiVersion,
          static_cast<uint32_t>(sg::Status::ready),
          0,
          0,
          0,
          steps * 2u,
          steps,
          workers,
          0,
          0,
          {0, 0, 0, 0, 0},
      };
      lrrt::copy_to_device(state_buffer, &state, sizeof(state));
    };

    for (uint32_t trial = 0; trial < kWarmupTrials; ++trial) {
      initialize_state();
      lrrt::copy_to_device(resident_values, zero_values);
      lrrt::launch(resident, config, resident_args);
      device.synchronize();
      lrrt::copy_to_device(host_values, zero_values);
      for (uint32_t step = 0; step < steps; ++step) {
        lrrt::launch(host_step, config, host_args);
      }
      device.synchronize();
    }

    const Measurement resident_measurement =
        measure(device, trials, resident_values, zero_values, initialize_state,
                [&] { lrrt::launch(resident, config, resident_args); });
    const Measurement host_measurement = measure(
        device, trials, host_values, zero_values, [] {},
        [&] {
          for (uint32_t step = 0; step < steps; ++step) {
            lrrt::launch(host_step, config, host_args);
          }
        });

    sg::State final_state{};
    lrrt::copy_to_host(&final_state, state_buffer, sizeof(final_state));
    if (final_state.status != static_cast<uint32_t>(sg::Status::complete) ||
        final_state.step != steps ||
        final_state.executed_tasks != static_cast<uint64_t>(steps) * workers) {
      throw std::runtime_error(
          "resident graph did not stop at the device predicate");
    }
    std::vector<uint32_t> resident_result(element_count);
    std::vector<uint32_t> host_result(element_count);
    lrrt::copy_to_host(resident_result, resident_values);
    lrrt::copy_to_host(host_result, host_values);
    verify_values(resident_result, steps, "resident");
    verify_values(host_result, steps, "host-driven");

    std::printf("\nLRRT Stateful Graph Benchmark\n");
    std::printf("=============================\n");
    std::printf("Workers:                  %u persistent workgroups\n",
                workers);
    std::printf("Conditional steps:        %u (max=%u)\n", steps, steps * 2u);
    std::printf("Tasks executed:           %llu\n",
                static_cast<unsigned long long>(final_state.executed_tasks));
    std::printf("Measured trials:          %u\n\n", trials);
    std::printf("%-24s %12s %12s %12s\n", "Path", "Device us", "Wall us",
                "Launches");
    std::printf("%-24s %12.3f %12.3f %12u\n", "Device-resident graph",
                resident_measurement.device_ns / 1.0e3,
                resident_measurement.wall_ns / 1.0e3, 1u);
    std::printf("%-24s %12.3f %12.3f %12u\n", "Host-driven steps",
                host_measurement.device_ns / 1.0e3,
                host_measurement.wall_ns / 1.0e3, steps);
    std::printf("\nDevice speedup: %.2fx\n",
                host_measurement.device_ns / resident_measurement.device_ns);
    std::printf("Wall speedup:   %.2fx\n\n",
                host_measurement.wall_ns / resident_measurement.wall_ns);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
