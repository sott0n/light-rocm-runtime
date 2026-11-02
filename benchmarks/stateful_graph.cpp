#include "executor/stateful_graph/stateful_graph_abi.hpp"
#include "lrrt/lrrt.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
constexpr uint32_t kElementCount = 1024;
constexpr uint32_t kWarmupTrials = 5;
constexpr uint32_t kDefaultWorkers = 4;
constexpr uint32_t kTaskCount = 4;
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
        const std::vector<float> &initial_values, Setup setup, Submit submit) {
  lrrt::Event start(device);
  lrrt::Event end(device);
  Measurement result;
  for (uint32_t trial = 0; trial < trials; ++trial) {
    lrrt::copy_to_device(values, initial_values);
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

uint32_t float_bits(float value) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void verify_values(const std::vector<float> &resident,
                   const std::vector<float> &host) {
  for (size_t index = 0; index < resident.size(); ++index) {
    const float tolerance = 1.0e-5f * std::max(1.0f, std::fabs(host[index]));
    if (!std::isfinite(resident[index]) ||
        std::fabs(resident[index] - host[index]) > tolerance) {
      std::fprintf(stderr, "stateful graph value[%zu]: resident=%f, host=%f\n",
                   index, resident[index], host[index]);
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

    const std::vector<sg::Task> tasks = {
        {static_cast<uint32_t>(sg::Opcode::rmsnorm_f32),
         0,
         0,
         kElementCount,
         float_bits(1.0e-5f),
         0,
         {0, 0}},
        {static_cast<uint32_t>(sg::Opcode::silu_f32),
         1u << 0,
         0,
         kElementCount,
         0,
         0,
         {0, 0}},
        {static_cast<uint32_t>(sg::Opcode::residual_add_f32),
         1u << 1,
         0,
         kElementCount,
         float_bits(0.05f),
         0,
         {0, 0}},
        {static_cast<uint32_t>(sg::Opcode::advance_state),
         1u << 2,
         0,
         0,
         1,
         0,
         {0, 0}},
    };
    std::vector<float> initial_values(kElementCount);
    for (uint32_t index = 0; index < kElementCount; ++index) {
      initial_values[index] = 0.25f + 0.01f * static_cast<float>(index % 17u);
    }

    lrrt::DeviceBuffer task_buffer(device, tasks.size() * sizeof(sg::Task));
    lrrt::DeviceBuffer resident_values(device, kElementCount * sizeof(float));
    lrrt::DeviceBuffer resident_workspace(device,
                                          kElementCount * sizeof(float));
    lrrt::DeviceBuffer resident_state(device, sizeof(sg::State));
    lrrt::DeviceBuffer host_values(device, kElementCount * sizeof(float));
    lrrt::DeviceBuffer host_workspace(device, kElementCount * sizeof(float));
    lrrt::DeviceBuffer host_state(device, sizeof(sg::State));
    lrrt::copy_to_device(task_buffer, tasks);

    const lr_launch_config_t resident_config = {
        {workers * kBlockSize, 1, 1}, {kBlockSize, 1, 1}, 0};
    const lr_launch_config_t host_config = {
        {kBlockSize, 1, 1}, {kBlockSize, 1, 1}, 0};
    const sg::ResidentArgs resident_args = {
        static_cast<sg::State *>(resident_state.data()),
        static_cast<const sg::Task *>(task_buffer.data()),
        static_cast<float *>(resident_values.data()),
        static_cast<float *>(resident_workspace.data()),
    };
    const auto *device_tasks =
        static_cast<const sg::Task *>(task_buffer.data());
    std::vector<sg::HostStepArgs> host_args(kTaskCount);
    for (uint32_t task_index = 0; task_index < kTaskCount; ++task_index) {
      host_args[task_index] = {
          static_cast<sg::State *>(host_state.data()),
          device_tasks + task_index,
          1,
          0,
          static_cast<float *>(host_values.data()),
          static_cast<float *>(host_workspace.data()),
      };
    }

    const auto initialize_state = [&](lrrt::DeviceBuffer &state_buffer) {
      const sg::State state = {
          sg::kAbiVersion,
          static_cast<uint32_t>(sg::Status::ready),
          0,
          0,
          0,
          0,
          steps * 2u,
          steps,
          kTaskCount,
          0,
          0,
          0,
          {0, 0, 0},
      };
      lrrt::copy_to_device(state_buffer, &state, sizeof(state));
    };

    // Exercise device-side program validation before timing. Making the root
    // depend on the final task closes the linear chain into a cycle.
    std::vector<sg::Task> cyclic_tasks = tasks;
    cyclic_tasks[0].dependency_mask = 1u << (kTaskCount - 1u);
    lrrt::copy_to_device(task_buffer, cyclic_tasks);
    initialize_state(resident_state);
    lrrt::launch(resident, resident_config, resident_args);
    device.synchronize();
    sg::State invalid_state{};
    lrrt::copy_to_host(&invalid_state, resident_state, sizeof(invalid_state));
    if (invalid_state.status !=
            static_cast<uint32_t>(sg::Status::invalid_program) ||
        invalid_state.error_code != 1) {
      throw std::runtime_error("resident graph did not reject a cyclic DAG");
    }
    lrrt::copy_to_device(task_buffer, tasks);

    for (uint32_t trial = 0; trial < kWarmupTrials; ++trial) {
      initialize_state(resident_state);
      lrrt::copy_to_device(resident_values, initial_values);
      lrrt::launch(resident, resident_config, resident_args);
      device.synchronize();
      initialize_state(host_state);
      lrrt::copy_to_device(host_values, initial_values);
      for (uint32_t step = 0; step < steps; ++step) {
        for (const sg::HostStepArgs &args : host_args) {
          lrrt::launch(host_step, host_config, args);
        }
      }
      device.synchronize();
    }

    const Measurement resident_measurement = measure(
        device, trials, resident_values, initial_values,
        [&] { initialize_state(resident_state); },
        [&] { lrrt::launch(resident, resident_config, resident_args); });
    const Measurement host_measurement = measure(
        device, trials, host_values, initial_values,
        [&] { initialize_state(host_state); },
        [&] {
          for (uint32_t step = 0; step < steps; ++step) {
            for (const sg::HostStepArgs &args : host_args) {
              lrrt::launch(host_step, host_config, args);
            }
          }
        });

    sg::State final_state{};
    lrrt::copy_to_host(&final_state, resident_state, sizeof(final_state));
    if (final_state.status != static_cast<uint32_t>(sg::Status::complete) ||
        final_state.step != steps || final_state.predicate_value != steps ||
        final_state.executed_tasks !=
            static_cast<uint64_t>(steps) * kTaskCount) {
      throw std::runtime_error(
          "resident graph did not stop at the device predicate");
    }
    sg::State final_host_state{};
    lrrt::copy_to_host(&final_host_state, host_state, sizeof(final_host_state));
    if (final_host_state.predicate_value != steps) {
      throw std::runtime_error("host-driven graph did not advance its state");
    }
    std::vector<float> resident_result(kElementCount);
    std::vector<float> host_result(kElementCount);
    lrrt::copy_to_host(resident_result, resident_values);
    lrrt::copy_to_host(host_result, host_values);
    verify_values(resident_result, host_result);

    std::printf("\nLRRT Stateful Graph Benchmark\n");
    std::printf("=============================\n");
    std::printf("Workers:                  %u persistent workgroups\n",
                workers);
    std::printf("Stages per step:          %u heterogeneous tasks\n",
                kTaskCount);
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
                host_measurement.wall_ns / 1.0e3, steps * kTaskCount);
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
