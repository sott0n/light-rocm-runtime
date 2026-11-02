#include "executor/qwen/weight_bundle.hpp"
#include "executor/stateful_graph/qwen_decode_abi.hpp"
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
#include <utility>
#include <vector>

#ifndef LRRT_STATEFUL_GRAPH_HSACO
#define LRRT_STATEFUL_GRAPH_HSACO "stateful_graph_kernel.hsaco"
#endif

namespace {

namespace sg = lrrt::stateful_graph;
namespace qd = lrrt::stateful_graph::qwen_decode;
using Clock = std::chrono::steady_clock;

constexpr uint32_t kBlockSize = sg::kWorkerThreads;
constexpr uint32_t kWarmupTrials = 5;
constexpr uint32_t kDefaultWorkers = 20;
constexpr uint32_t kMaximumWorkers = 20;

struct Measurement {
  double device_ns = 0.0;
  double wall_ns = 0.0;
};

struct Options {
  uint32_t steps = 64;
  uint32_t trials = 20;
  uint32_t workers = kDefaultWorkers;
  const char *weights_path = nullptr;
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

Options parse_options(int argc, char **argv) {
  Options options;
  uint32_t positional = 0;
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--weights") == 0) {
      if (++index >= argc) {
        throw std::invalid_argument("--weights requires a manifest path");
      }
      options.weights_path = argv[index];
      continue;
    }
    if (argv[index][0] == '-') {
      throw std::invalid_argument(std::string("unknown option: ") +
                                  argv[index]);
    }
    switch (positional++) {
    case 0:
      options.steps = parse_u32(argv[index], "steps");
      break;
    case 1:
      options.trials = parse_u32(argv[index], "trials");
      break;
    case 2:
      options.workers = parse_u32(argv[index], "workers");
      break;
    default:
      throw std::invalid_argument(
          "usage: lrrt_stateful_graph_benchmark [steps] [trials] [workers] "
          "[--weights weights.json]");
    }
  }
  return options;
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

sg::Task make_task(qd::Opcode opcode, uint64_t dependency_mask, uint32_t begin,
                   uint32_t end, uint32_t argument0 = 0) {
  return {
      static_cast<uint32_t>(opcode),
      static_cast<uint32_t>(dependency_mask),
      begin,
      end,
      argument0,
      0,
      static_cast<uint32_t>(dependency_mask >> 32u),
      0,
  };
}

std::vector<sg::Task> make_qwen_decoder_tasks() {
  std::vector<sg::Task> tasks;
  tasks.reserve(qd::kTaskCount);
  tasks.push_back(make_task(qd::Opcode::attention_rmsnorm_f32, 0, 0,
                            qd::kHiddenSize, float_bits(1.0e-6f)));

  uint64_t qkv_mask = 0;
  for (uint32_t partition = 0; partition < qd::kAttentionProjectionPartitions;
       ++partition) {
    const uint32_t task_index = static_cast<uint32_t>(tasks.size());
    tasks.push_back(make_task(qd::Opcode::qkv_projection_f32, 1u,
                              qd::kPackedQkvSize * partition /
                                  qd::kAttentionProjectionPartitions,
                              qd::kPackedQkvSize * (partition + 1u) /
                                  qd::kAttentionProjectionPartitions));
    qkv_mask |= static_cast<uint64_t>(1) << task_index;
  }

  const uint32_t rope_task_index = static_cast<uint32_t>(tasks.size());
  tasks.push_back(
      make_task(qd::Opcode::rope_kv_cache_f32, qkv_mask, 0, qd::kRotaryHeads));
  const uint64_t rope_mask = static_cast<uint64_t>(1) << rope_task_index;
  tasks.push_back(make_task(qd::Opcode::mlp_rmsnorm_f32, rope_mask, 0,
                            qd::kHiddenSize, float_bits(1.0e-5f)));

  uint64_t activated_mask = 0;
  for (uint32_t partition = 0; partition < qd::kFfnProjectionPartitions;
       ++partition) {
    const uint32_t task_index = static_cast<uint32_t>(tasks.size());
    tasks.push_back(make_task(
        qd::Opcode::gate_up_silu_f32,
        static_cast<uint64_t>(1) << (qd::kAttentionTaskCount),
        qd::kIntermediateSize * partition / qd::kFfnProjectionPartitions,
        qd::kIntermediateSize * (partition + 1u) /
            qd::kFfnProjectionPartitions));
    activated_mask |= static_cast<uint64_t>(1) << task_index;
  }

  uint64_t down_mask = 0;
  for (uint32_t partition = 0; partition < qd::kDownProjectionPartitions;
       ++partition) {
    const uint32_t task_index = static_cast<uint32_t>(tasks.size());
    tasks.push_back(make_task(
        qd::Opcode::down_matvec_f32, activated_mask,
        qd::kHiddenSize * partition / qd::kDownProjectionPartitions,
        qd::kHiddenSize * (partition + 1u) / qd::kDownProjectionPartitions));
    down_mask |= static_cast<uint64_t>(1) << task_index;
  }

  tasks.push_back(make_task(qd::Opcode::residual_advance, down_mask, 0,
                            qd::kHiddenSize, 1));
  if (tasks.size() != qd::kTaskCount) {
    throw std::runtime_error("Qwen decoder task layout mismatch");
  }
  return tasks;
}

std::vector<sg::Task> make_scheduler_capacity_tasks() {
  std::vector<sg::Task> tasks;
  tasks.reserve(sg::kMaximumTasks);
  for (uint32_t index = 0; index < sg::kMaximumTasks; ++index) {
    const uint64_t dependency =
        index == 0 ? 0 : static_cast<uint64_t>(1) << (index - 1u);
    tasks.push_back(
        make_task(qd::Opcode::residual_advance, dependency, 0, 1, 1));
  }
  return tasks;
}

struct HostStage {
  uint32_t task_begin;
  uint32_t task_count;
};

constexpr HostStage kHostStages[] = {
    {0, 1}, {1, 8}, {9, 1}, {10, 1}, {11, 20}, {31, 10}, {41, 1},
};

void fill_weight(std::vector<float> &weight, uint32_t salt) {
  for (size_t index = 0; index < weight.size(); ++index) {
    const int32_t centered =
        static_cast<int32_t>((index * (salt * 2u + 1u) + salt) % 31u) - 15;
    weight[index] = static_cast<float>(centered) * 0.0005f;
  }
}

void verify_values(const char *label, const std::vector<float> &resident,
                   const std::vector<float> &host) {
  for (size_t index = 0; index < resident.size(); ++index) {
    const float tolerance = 1.0e-5f * std::max(1.0f, std::fabs(host[index]));
    if (!std::isfinite(resident[index]) ||
        std::fabs(resident[index] - host[index]) > tolerance) {
      std::fprintf(stderr, "stateful graph %s[%zu]: resident=%f, host=%f\n",
                   label, index, resident[index], host[index]);
      throw std::runtime_error("stateful-graph result mismatch");
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    const uint32_t steps = options.steps;
    const uint32_t trials = options.trials;
    const uint32_t workers = options.workers;
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

    const std::vector<sg::Task> tasks = make_qwen_decoder_tasks();
    const uint32_t cache_capacity = steps * 2u;
    std::vector<float> initial_values(qd::kHiddenSize);
    std::vector<float> attention_norm_weight(qd::kHiddenSize);
    std::vector<float> mlp_norm_weight(qd::kHiddenSize);
    for (uint32_t index = 0; index < qd::kHiddenSize; ++index) {
      initial_values[index] =
          0.03125f * static_cast<float>(static_cast<int32_t>(index % 31u) - 15);
      attention_norm_weight[index] =
          1.0f + 0.001f * static_cast<float>(index % 23u);
      mlp_norm_weight[index] = 1.0f + 0.001f * static_cast<float>(index % 29u);
    }
    const size_t q_elements =
        static_cast<size_t>(qd::kQuerySize) * qd::kHiddenSize;
    const size_t kv_elements =
        static_cast<size_t>(qd::kKvSize) * qd::kHiddenSize;
    const size_t gate_elements =
        static_cast<size_t>(qd::kIntermediateSize) * qd::kHiddenSize;
    const size_t down_elements =
        static_cast<size_t>(qd::kHiddenSize) * qd::kIntermediateSize;
    std::vector<float> q_weight(q_elements);
    std::vector<float> k_weight(kv_elements);
    std::vector<float> v_weight(kv_elements);
    std::vector<float> q_bias(qd::kQuerySize);
    std::vector<float> k_bias(qd::kKvSize);
    std::vector<float> v_bias(qd::kKvSize);
    std::vector<float> gate_weight(gate_elements);
    std::vector<float> up_weight(gate_elements);
    std::vector<float> down_weight(down_elements);
    float rope_theta = 1000000.0f;
    if (options.weights_path) {
      lrrt::executor::qwen::DecoderLayerWeights weights =
          lrrt::executor::qwen::load_decoder_layer_weights(
              options.weights_path);
      if (weights.shape.hidden != qd::kHiddenSize ||
          weights.shape.heads != qd::kAttentionHeads ||
          weights.shape.kv_heads != qd::kKvHeads ||
          weights.shape.head_dim != qd::kHeadSize ||
          weights.shape.intermediate != qd::kIntermediateSize) {
        throw std::runtime_error(
            "checkpoint decoder shape does not match the megakernel");
      }
      rope_theta = weights.rope_theta;
      attention_norm_weight = std::move(weights.attention_norm_weight);
      mlp_norm_weight = std::move(weights.mlp_norm_weight);
      q_weight = std::move(weights.q_weight);
      k_weight = std::move(weights.k_weight);
      v_weight = std::move(weights.v_weight);
      q_bias = std::move(weights.q_bias);
      k_bias = std::move(weights.k_bias);
      v_bias = std::move(weights.v_bias);
      gate_weight = std::move(weights.gate_weight);
      up_weight = std::move(weights.up_weight);
      down_weight = std::move(weights.down_weight);
    } else {
      fill_weight(q_weight, 1);
      fill_weight(k_weight, 2);
      fill_weight(v_weight, 3);
      fill_weight(q_bias, 4);
      fill_weight(k_bias, 5);
      fill_weight(v_bias, 6);
      fill_weight(gate_weight, 7);
      fill_weight(up_weight, 8);
      fill_weight(down_weight, 9);
    }

    const size_t rope_elements =
        static_cast<size_t>(cache_capacity) * qd::kRotarySize;
    const size_t cache_elements =
        static_cast<size_t>(cache_capacity) * qd::kKvSize;
    std::vector<float> rope_cos(rope_elements);
    std::vector<float> rope_sin(rope_elements);
    for (uint32_t position = 0; position < cache_capacity; ++position) {
      for (uint32_t frequency = 0; frequency < qd::kRotarySize; ++frequency) {
        const float exponent =
            -2.0f * static_cast<float>(frequency) / qd::kHeadSize;
        const float angle =
            static_cast<float>(position) * std::pow(rope_theta, exponent);
        const size_t index =
            static_cast<size_t>(position) * qd::kRotarySize + frequency;
        rope_cos[index] = std::cos(angle);
        rope_sin[index] = std::sin(angle);
      }
    }

    lrrt::DeviceBuffer task_buffer(device,
                                   sg::kMaximumTasks * sizeof(sg::Task));
    lrrt::DeviceBuffer attention_norm_weight_buffer(
        device, attention_norm_weight.size() * sizeof(float));
    lrrt::DeviceBuffer mlp_norm_weight_buffer(device, mlp_norm_weight.size() *
                                                          sizeof(float));
    lrrt::DeviceBuffer q_weight_buffer(device, q_weight.size() * sizeof(float));
    lrrt::DeviceBuffer k_weight_buffer(device, k_weight.size() * sizeof(float));
    lrrt::DeviceBuffer v_weight_buffer(device, v_weight.size() * sizeof(float));
    lrrt::DeviceBuffer q_bias_buffer(device, q_bias.size() * sizeof(float));
    lrrt::DeviceBuffer k_bias_buffer(device, k_bias.size() * sizeof(float));
    lrrt::DeviceBuffer v_bias_buffer(device, v_bias.size() * sizeof(float));
    lrrt::DeviceBuffer rope_cos_buffer(device, rope_cos.size() * sizeof(float));
    lrrt::DeviceBuffer rope_sin_buffer(device, rope_sin.size() * sizeof(float));
    lrrt::DeviceBuffer gate_weight_buffer(device,
                                          gate_weight.size() * sizeof(float));
    lrrt::DeviceBuffer up_weight_buffer(device,
                                        up_weight.size() * sizeof(float));
    lrrt::DeviceBuffer down_weight_buffer(device,
                                          down_weight.size() * sizeof(float));
    lrrt::DeviceBuffer resident_values(device, qd::kHiddenSize * sizeof(float));
    lrrt::DeviceBuffer resident_attention_normalized(device, qd::kHiddenSize *
                                                                 sizeof(float));
    lrrt::DeviceBuffer resident_query(device, qd::kQuerySize * sizeof(float));
    lrrt::DeviceBuffer resident_key(device, qd::kKvSize * sizeof(float));
    lrrt::DeviceBuffer resident_value(device, qd::kKvSize * sizeof(float));
    lrrt::DeviceBuffer resident_key_cache(device,
                                          cache_elements * sizeof(float));
    lrrt::DeviceBuffer resident_value_cache(device,
                                            cache_elements * sizeof(float));
    lrrt::DeviceBuffer resident_mlp_normalized(device,
                                               qd::kHiddenSize * sizeof(float));
    lrrt::DeviceBuffer resident_activated(device, qd::kIntermediateSize *
                                                      sizeof(float));
    lrrt::DeviceBuffer resident_projected(device,
                                          qd::kHiddenSize * sizeof(float));
    lrrt::DeviceBuffer resident_state(device, sizeof(sg::State));
    lrrt::DeviceBuffer host_values(device, qd::kHiddenSize * sizeof(float));
    lrrt::DeviceBuffer host_attention_normalized(device, qd::kHiddenSize *
                                                             sizeof(float));
    lrrt::DeviceBuffer host_query(device, qd::kQuerySize * sizeof(float));
    lrrt::DeviceBuffer host_key(device, qd::kKvSize * sizeof(float));
    lrrt::DeviceBuffer host_value(device, qd::kKvSize * sizeof(float));
    lrrt::DeviceBuffer host_key_cache(device, cache_elements * sizeof(float));
    lrrt::DeviceBuffer host_value_cache(device, cache_elements * sizeof(float));
    lrrt::DeviceBuffer host_mlp_normalized(device,
                                           qd::kHiddenSize * sizeof(float));
    lrrt::DeviceBuffer host_activated(device,
                                      qd::kIntermediateSize * sizeof(float));
    lrrt::DeviceBuffer host_projected(device, qd::kHiddenSize * sizeof(float));
    lrrt::DeviceBuffer host_state(device, sizeof(sg::State));
    lrrt::copy_to_device(task_buffer, tasks);
    lrrt::copy_to_device(attention_norm_weight_buffer, attention_norm_weight);
    lrrt::copy_to_device(mlp_norm_weight_buffer, mlp_norm_weight);
    lrrt::copy_to_device(q_weight_buffer, q_weight);
    lrrt::copy_to_device(k_weight_buffer, k_weight);
    lrrt::copy_to_device(v_weight_buffer, v_weight);
    lrrt::copy_to_device(q_bias_buffer, q_bias);
    lrrt::copy_to_device(k_bias_buffer, k_bias);
    lrrt::copy_to_device(v_bias_buffer, v_bias);
    lrrt::copy_to_device(rope_cos_buffer, rope_cos);
    lrrt::copy_to_device(rope_sin_buffer, rope_sin);
    lrrt::copy_to_device(gate_weight_buffer, gate_weight);
    lrrt::copy_to_device(up_weight_buffer, up_weight);
    lrrt::copy_to_device(down_weight_buffer, down_weight);

    const qd::Context resident_context = {
        static_cast<float *>(resident_values.data()),
        static_cast<const float *>(attention_norm_weight_buffer.data()),
        static_cast<const float *>(q_weight_buffer.data()),
        static_cast<const float *>(k_weight_buffer.data()),
        static_cast<const float *>(v_weight_buffer.data()),
        static_cast<const float *>(q_bias_buffer.data()),
        static_cast<const float *>(k_bias_buffer.data()),
        static_cast<const float *>(v_bias_buffer.data()),
        static_cast<const float *>(mlp_norm_weight_buffer.data()),
        static_cast<const float *>(gate_weight_buffer.data()),
        static_cast<const float *>(up_weight_buffer.data()),
        static_cast<const float *>(down_weight_buffer.data()),
        static_cast<float *>(resident_attention_normalized.data()),
        static_cast<float *>(resident_query.data()),
        static_cast<float *>(resident_key.data()),
        static_cast<float *>(resident_value.data()),
        static_cast<const float *>(rope_cos_buffer.data()),
        static_cast<const float *>(rope_sin_buffer.data()),
        static_cast<float *>(resident_key_cache.data()),
        static_cast<float *>(resident_value_cache.data()),
        static_cast<float *>(resident_mlp_normalized.data()),
        static_cast<float *>(resident_activated.data()),
        static_cast<float *>(resident_projected.data()),
        qd::kHiddenSize,
        qd::kQuerySize,
        qd::kKvSize,
        qd::kIntermediateSize,
        cache_capacity,
        0,
    };
    const qd::Context host_context = {
        static_cast<float *>(host_values.data()),
        static_cast<const float *>(attention_norm_weight_buffer.data()),
        static_cast<const float *>(q_weight_buffer.data()),
        static_cast<const float *>(k_weight_buffer.data()),
        static_cast<const float *>(v_weight_buffer.data()),
        static_cast<const float *>(q_bias_buffer.data()),
        static_cast<const float *>(k_bias_buffer.data()),
        static_cast<const float *>(v_bias_buffer.data()),
        static_cast<const float *>(mlp_norm_weight_buffer.data()),
        static_cast<const float *>(gate_weight_buffer.data()),
        static_cast<const float *>(up_weight_buffer.data()),
        static_cast<const float *>(down_weight_buffer.data()),
        static_cast<float *>(host_attention_normalized.data()),
        static_cast<float *>(host_query.data()),
        static_cast<float *>(host_key.data()),
        static_cast<float *>(host_value.data()),
        static_cast<const float *>(rope_cos_buffer.data()),
        static_cast<const float *>(rope_sin_buffer.data()),
        static_cast<float *>(host_key_cache.data()),
        static_cast<float *>(host_value_cache.data()),
        static_cast<float *>(host_mlp_normalized.data()),
        static_cast<float *>(host_activated.data()),
        static_cast<float *>(host_projected.data()),
        qd::kHiddenSize,
        qd::kQuerySize,
        qd::kKvSize,
        qd::kIntermediateSize,
        cache_capacity,
        0,
    };
    lrrt::DeviceBuffer resident_context_buffer(device, sizeof(qd::Context));
    lrrt::DeviceBuffer host_context_buffer(device, sizeof(qd::Context));
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

    const auto initialize_state = [&](lrrt::DeviceBuffer &state_buffer,
                                      uint32_t task_count, uint32_t max_steps,
                                      uint32_t stop_value) {
      const sg::State state = {
          sg::kAbiVersion,
          static_cast<uint32_t>(sg::Status::ready),
          0,
          0,
          0,
          0,
          max_steps,
          stop_value,
          task_count,
          0,
          0,
          0,
          0,
          0,
          workers << 16u,
      };
      lrrt::copy_to_device(state_buffer, &state, sizeof(state));
    };

    // Exercise both 64-bit mask words, including task bit 63, before timing.
    const std::vector<sg::Task> capacity_tasks =
        make_scheduler_capacity_tasks();
    const std::vector<float> zero_projected(qd::kHiddenSize, 0.0f);
    lrrt::copy_to_device(task_buffer, capacity_tasks);
    lrrt::copy_to_device(resident_projected, zero_projected);
    initialize_state(resident_state, sg::kMaximumTasks, 1, sg::kMaximumTasks);
    lrrt::launch(resident, resident_config, resident_args);
    device.synchronize();
    sg::State capacity_state{};
    lrrt::copy_to_host(&capacity_state, resident_state, sizeof(capacity_state));
    if (capacity_state.status != static_cast<uint32_t>(sg::Status::complete) ||
        capacity_state.step != 1 ||
        capacity_state.predicate_value != sg::kMaximumTasks ||
        capacity_state.executed_tasks != sg::kMaximumTasks) {
      throw std::runtime_error("resident graph did not execute 64-task DAG");
    }

    // Exercise device-side program validation before timing. Making the root
    // depend on the final task closes the linear chain into a cycle.
    std::vector<sg::Task> cyclic_tasks = tasks;
    const uint64_t final_task_mask = static_cast<uint64_t>(1)
                                     << (qd::kTaskCount - 1u);
    cyclic_tasks[0].dependency_mask = static_cast<uint32_t>(final_task_mask);
    cyclic_tasks[0].dependency_mask_high =
        static_cast<uint32_t>(final_task_mask >> 32u);
    lrrt::copy_to_device(task_buffer, cyclic_tasks);
    initialize_state(resident_state, qd::kTaskCount, steps * 2u, steps);
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
      initialize_state(resident_state, qd::kTaskCount, steps * 2u, steps);
      lrrt::copy_to_device(resident_values, initial_values);
      lrrt::launch(resident, resident_config, resident_args);
      device.synchronize();
      initialize_state(host_state, qd::kTaskCount, steps * 2u, steps);
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
        [&] {
          initialize_state(resident_state, qd::kTaskCount, steps * 2u, steps);
        },
        [&] { lrrt::launch(resident, resident_config, resident_args); });
    const Measurement host_measurement = measure(
        device, trials, host_values, initial_values,
        [&] {
          initialize_state(host_state, qd::kTaskCount, steps * 2u, steps);
        },
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
            static_cast<uint64_t>(steps) * qd::kTaskCount) {
      throw std::runtime_error(
          "resident graph did not stop at the device predicate");
    }
    sg::State final_host_state{};
    lrrt::copy_to_host(&final_host_state, host_state, sizeof(final_host_state));
    if (final_host_state.predicate_value != steps ||
        final_host_state.step != steps) {
      throw std::runtime_error("host-driven graph did not advance its state");
    }
    std::vector<float> resident_result(qd::kHiddenSize);
    std::vector<float> host_result(qd::kHiddenSize);
    lrrt::copy_to_host(resident_result, resident_values);
    lrrt::copy_to_host(host_result, host_values);
    verify_values("hidden", resident_result, host_result);
    std::vector<float> resident_q_result(qd::kQuerySize);
    std::vector<float> host_q_result(qd::kQuerySize);
    std::vector<float> resident_k_result(qd::kKvSize);
    std::vector<float> host_k_result(qd::kKvSize);
    std::vector<float> resident_v_result(qd::kKvSize);
    std::vector<float> host_v_result(qd::kKvSize);
    lrrt::copy_to_host(resident_q_result, resident_query);
    lrrt::copy_to_host(host_q_result, host_query);
    lrrt::copy_to_host(resident_k_result, resident_key);
    lrrt::copy_to_host(host_k_result, host_key);
    lrrt::copy_to_host(resident_v_result, resident_value);
    lrrt::copy_to_host(host_v_result, host_value);
    verify_values("query", resident_q_result, host_q_result);
    verify_values("key", resident_k_result, host_k_result);
    verify_values("value", resident_v_result, host_v_result);
    std::vector<float> resident_key_cache_result(static_cast<size_t>(steps) *
                                                 qd::kKvSize);
    std::vector<float> host_key_cache_result(resident_key_cache_result.size());
    std::vector<float> resident_value_cache_result(
        resident_key_cache_result.size());
    std::vector<float> host_value_cache_result(
        resident_key_cache_result.size());
    lrrt::copy_to_host(resident_key_cache_result, resident_key_cache);
    lrrt::copy_to_host(host_key_cache_result, host_key_cache);
    lrrt::copy_to_host(resident_value_cache_result, resident_value_cache);
    lrrt::copy_to_host(host_value_cache_result, host_value_cache);
    verify_values("key cache", resident_key_cache_result,
                  host_key_cache_result);
    verify_values("value cache", resident_value_cache_result,
                  host_value_cache_result);

    std::printf("\nLRRT Stateful Graph Benchmark\n");
    std::printf("=============================\n");
    std::printf("Workers:                  %u persistent workgroups\n",
                workers);
    std::printf("Qwen decoder shape:       hidden=%u heads=%u kv_heads=%u "
                "head_dim=%u intermediate=%u\n",
                qd::kHiddenSize, qd::kAttentionHeads, qd::kKvHeads,
                qd::kHeadSize, qd::kIntermediateSize);
    std::printf("Weights:                  %s\n",
                options.weights_path ? options.weights_path
                                     : "deterministic synthetic f32");
    std::printf("Tasks per step:           %u dependency-scheduled tasks\n",
                qd::kTaskCount);
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
