#include "executor/stateful_graph/qwen_ffn_abi.hpp"
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
namespace qf = lrrt::stateful_graph::qwen_ffn;
using Clock = std::chrono::steady_clock;

constexpr uint32_t kBlockSize = sg::kWorkerThreads;
constexpr uint32_t kWarmupTrials = 5;
constexpr uint32_t kDefaultWorkers = 20;
constexpr uint32_t kMaximumWorkers = 20;

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

sg::Task make_task(qf::Opcode opcode, uint32_t dependency_mask, uint32_t begin,
                   uint32_t end, uint32_t argument0 = 0) {
  return {
      static_cast<uint32_t>(opcode),
      dependency_mask,
      begin,
      end,
      argument0,
      0,
      {0, 0},
  };
}

std::vector<sg::Task> make_qwen_ffn_tasks() {
  std::vector<sg::Task> tasks;
  tasks.reserve(qf::kTaskCount);
  tasks.push_back(make_task(qf::Opcode::rmsnorm_f32, 0, 0, qf::kHiddenSize,
                            float_bits(1.0e-5f)));

  uint32_t activated_mask = 0;
  for (uint32_t partition = 0; partition < qf::kProjectionPartitions;
       ++partition) {
    const uint32_t task_index = static_cast<uint32_t>(tasks.size());
    tasks.push_back(make_task(
        qf::Opcode::gate_up_silu_f32, 1u,
        qf::kIntermediateSize * partition / qf::kProjectionPartitions,
        qf::kIntermediateSize * (partition + 1u) / qf::kProjectionPartitions));
    activated_mask |= 1u << task_index;
  }

  uint32_t down_mask = 0;
  for (uint32_t partition = 0; partition < qf::kDownProjectionPartitions;
       ++partition) {
    const uint32_t task_index = static_cast<uint32_t>(tasks.size());
    tasks.push_back(make_task(
        qf::Opcode::down_matvec_f32, activated_mask,
        qf::kHiddenSize * partition / qf::kDownProjectionPartitions,
        qf::kHiddenSize * (partition + 1u) / qf::kDownProjectionPartitions));
    down_mask |= 1u << task_index;
  }

  tasks.push_back(make_task(qf::Opcode::residual_advance, down_mask, 0,
                            qf::kHiddenSize, 1));
  if (tasks.size() != qf::kTaskCount) {
    throw std::runtime_error("Qwen FFN task layout mismatch");
  }
  return tasks;
}

struct HostStage {
  uint32_t task_begin;
  uint32_t task_count;
};

constexpr HostStage kHostStages[] = {
    {0, 1},
    {1, 20},
    {21, 10},
    {31, 1},
};

void fill_weight(std::vector<float> &weight, uint32_t salt) {
  for (size_t index = 0; index < weight.size(); ++index) {
    const int32_t centered =
        static_cast<int32_t>((index * (salt * 2u + 1u) + salt) % 31u) - 15;
    weight[index] = static_cast<float>(centered) * 0.0005f;
  }
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
      throw std::invalid_argument("workers must not exceed 20");
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

    const std::vector<sg::Task> tasks = make_qwen_ffn_tasks();
    std::vector<float> initial_values(qf::kHiddenSize);
    std::vector<float> norm_weight(qf::kHiddenSize);
    for (uint32_t index = 0; index < qf::kHiddenSize; ++index) {
      initial_values[index] =
          0.03125f * static_cast<float>(static_cast<int32_t>(index % 31u) - 15);
      norm_weight[index] = 1.0f + 0.001f * static_cast<float>(index % 29u);
    }
    const size_t gate_elements =
        static_cast<size_t>(qf::kIntermediateSize) * qf::kHiddenSize;
    const size_t down_elements =
        static_cast<size_t>(qf::kHiddenSize) * qf::kIntermediateSize;
    std::vector<float> gate_weight(gate_elements);
    std::vector<float> up_weight(gate_elements);
    std::vector<float> down_weight(down_elements);
    fill_weight(gate_weight, 1);
    fill_weight(up_weight, 2);
    fill_weight(down_weight, 3);

    lrrt::DeviceBuffer task_buffer(device, tasks.size() * sizeof(sg::Task));
    lrrt::DeviceBuffer norm_weight_buffer(device,
                                          norm_weight.size() * sizeof(float));
    lrrt::DeviceBuffer gate_weight_buffer(device,
                                          gate_weight.size() * sizeof(float));
    lrrt::DeviceBuffer up_weight_buffer(device,
                                        up_weight.size() * sizeof(float));
    lrrt::DeviceBuffer down_weight_buffer(device,
                                          down_weight.size() * sizeof(float));
    lrrt::DeviceBuffer resident_values(device, qf::kHiddenSize * sizeof(float));
    lrrt::DeviceBuffer resident_normalized(device,
                                           qf::kHiddenSize * sizeof(float));
    lrrt::DeviceBuffer resident_activated(device, qf::kIntermediateSize *
                                                      sizeof(float));
    lrrt::DeviceBuffer resident_projected(device,
                                          qf::kHiddenSize * sizeof(float));
    lrrt::DeviceBuffer resident_state(device, sizeof(sg::State));
    lrrt::DeviceBuffer host_values(device, qf::kHiddenSize * sizeof(float));
    lrrt::DeviceBuffer host_normalized(device, qf::kHiddenSize * sizeof(float));
    lrrt::DeviceBuffer host_activated(device,
                                      qf::kIntermediateSize * sizeof(float));
    lrrt::DeviceBuffer host_projected(device, qf::kHiddenSize * sizeof(float));
    lrrt::DeviceBuffer host_state(device, sizeof(sg::State));
    lrrt::copy_to_device(task_buffer, tasks);
    lrrt::copy_to_device(norm_weight_buffer, norm_weight);
    lrrt::copy_to_device(gate_weight_buffer, gate_weight);
    lrrt::copy_to_device(up_weight_buffer, up_weight);
    lrrt::copy_to_device(down_weight_buffer, down_weight);

    const qf::Context resident_context = {
        static_cast<float *>(resident_values.data()),
        static_cast<const float *>(norm_weight_buffer.data()),
        static_cast<const float *>(gate_weight_buffer.data()),
        static_cast<const float *>(up_weight_buffer.data()),
        static_cast<const float *>(down_weight_buffer.data()),
        static_cast<float *>(resident_normalized.data()),
        static_cast<float *>(resident_activated.data()),
        static_cast<float *>(resident_projected.data()),
        qf::kHiddenSize,
        qf::kIntermediateSize,
        {0, 0},
    };
    const qf::Context host_context = {
        static_cast<float *>(host_values.data()),
        static_cast<const float *>(norm_weight_buffer.data()),
        static_cast<const float *>(gate_weight_buffer.data()),
        static_cast<const float *>(up_weight_buffer.data()),
        static_cast<const float *>(down_weight_buffer.data()),
        static_cast<float *>(host_normalized.data()),
        static_cast<float *>(host_activated.data()),
        static_cast<float *>(host_projected.data()),
        qf::kHiddenSize,
        qf::kIntermediateSize,
        {0, 0},
    };
    lrrt::DeviceBuffer resident_context_buffer(device, sizeof(qf::Context));
    lrrt::DeviceBuffer host_context_buffer(device, sizeof(qf::Context));
    lrrt::copy_to_device(resident_context_buffer, &resident_context,
                         sizeof(resident_context));
    lrrt::copy_to_device(host_context_buffer, &host_context,
                         sizeof(host_context));

    const lr_launch_config_t resident_config = {
        {workers * kBlockSize, 1, 1}, {kBlockSize, 1, 1}, 0};
    const sg::ResidentArgs resident_args = {
        static_cast<sg::State *>(resident_state.data()),
        static_cast<const sg::Task *>(task_buffer.data()),
        resident_context_buffer.data(),
    };
    const auto *device_tasks =
        static_cast<const sg::Task *>(task_buffer.data());
    struct HostLaunch {
      sg::HostStepArgs args;
      lr_launch_config_t config;
    };
    std::vector<HostLaunch> host_launches;
    for (const HostStage stage : kHostStages) {
      host_launches.push_back({
          {
              static_cast<sg::State *>(host_state.data()),
              device_tasks + stage.task_begin,
              stage.task_count,
              0,
              host_context_buffer.data(),
          },
          {{stage.task_count * kBlockSize, 1, 1}, {kBlockSize, 1, 1}, 0},
      });
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
          qf::kTaskCount,
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
    cyclic_tasks[0].dependency_mask = 1u << (qf::kTaskCount - 1u);
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
        for (const HostLaunch &launch : host_launches) {
          lrrt::launch(host_step, launch.config, launch.args);
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
            for (const HostLaunch &launch : host_launches) {
              lrrt::launch(host_step, launch.config, launch.args);
            }
          }
        });

    sg::State final_state{};
    lrrt::copy_to_host(&final_state, resident_state, sizeof(final_state));
    if (final_state.status != static_cast<uint32_t>(sg::Status::complete) ||
        final_state.step != steps || final_state.predicate_value != steps ||
        final_state.executed_tasks !=
            static_cast<uint64_t>(steps) * qf::kTaskCount) {
      throw std::runtime_error(
          "resident graph did not stop at the device predicate");
    }
    sg::State final_host_state{};
    lrrt::copy_to_host(&final_host_state, host_state, sizeof(final_host_state));
    if (final_host_state.predicate_value != steps) {
      throw std::runtime_error("host-driven graph did not advance its state");
    }
    std::vector<float> resident_result(qf::kHiddenSize);
    std::vector<float> host_result(qf::kHiddenSize);
    lrrt::copy_to_host(resident_result, resident_values);
    lrrt::copy_to_host(host_result, host_values);
    verify_values(resident_result, host_result);

    std::printf("\nLRRT Stateful Graph Benchmark\n");
    std::printf("=============================\n");
    std::printf("Workers:                  %u persistent workgroups\n",
                workers);
    std::printf("Qwen FFN shape:           hidden=%u intermediate=%u\n",
                qf::kHiddenSize, qf::kIntermediateSize);
    std::printf("Tasks per step:           %u dependency-scheduled tasks\n",
                qf::kTaskCount);
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
                host_measurement.wall_ns / 1.0e3,
                steps * static_cast<uint32_t>(host_launches.size()));
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
