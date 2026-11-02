#include "executor/qwen/weight_bundle.hpp"
#include "executor/stateful_graph/qwen_decode_abi.hpp"
#include "lrrt/lrrt.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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
  uint32_t layers = 1;
  const char *weights_path = nullptr;
  const char *weights_dir = nullptr;
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
    if (std::strcmp(argv[index], "--weights-dir") == 0) {
      if (++index >= argc) {
        throw std::invalid_argument("--weights-dir requires a directory");
      }
      options.weights_dir = argv[index];
      continue;
    }
    if (std::strcmp(argv[index], "--layers") == 0) {
      if (++index >= argc) {
        throw std::invalid_argument("--layers requires a count");
      }
      options.layers = parse_u32(argv[index], "layers");
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
          "[--weights weights.json | --weights-dir directory --layers N]");
    }
  }
  if (options.weights_path && options.weights_dir) {
    throw std::invalid_argument("--weights and --weights-dir are exclusive");
  }
  if (options.weights_path && options.layers != 1) {
    throw std::invalid_argument("--weights supports exactly one layer");
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
                   uint32_t end, uint32_t argument0 = 0,
                   uint32_t argument1 = 0) {
  return {
      static_cast<uint32_t>(opcode),
      static_cast<uint32_t>(dependency_mask),
      begin,
      end,
      argument0,
      argument1,
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

  uint64_t attention_mask = 0;
  for (uint32_t partition = 0; partition < qd::kAttentionTaskPartitions;
       ++partition) {
    const uint32_t task_index = static_cast<uint32_t>(tasks.size());
    tasks.push_back(make_task(
        qd::Opcode::grouped_query_attention_f32, rope_mask,
        qd::kAttentionHeads * partition / qd::kAttentionTaskPartitions,
        qd::kAttentionHeads * (partition + 1u) / qd::kAttentionTaskPartitions));
    attention_mask |= static_cast<uint64_t>(1) << task_index;
  }

  uint64_t attention_output_mask = 0;
  for (uint32_t partition = 0; partition < qd::kAttentionOutputPartitions;
       ++partition) {
    const uint32_t task_index = static_cast<uint32_t>(tasks.size());
    tasks.push_back(make_task(
        qd::Opcode::attention_output_residual_f32, attention_mask,
        qd::kHiddenSize * partition / qd::kAttentionOutputPartitions,
        qd::kHiddenSize * (partition + 1u) / qd::kAttentionOutputPartitions));
    attention_output_mask |= static_cast<uint64_t>(1) << task_index;
  }

  tasks.push_back(make_task(qd::Opcode::mlp_rmsnorm_f32, attention_output_mask,
                            0, qd::kHiddenSize, float_bits(1.0e-6f)));

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
                            qd::kHiddenSize, 1, 1));
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
    {0, 1},
    {1, qd::kAttentionProjectionPartitions},
    {1 + qd::kAttentionProjectionPartitions, 1},
    {2 + qd::kAttentionProjectionPartitions, qd::kAttentionTaskPartitions},
    {2 + qd::kAttentionProjectionPartitions + qd::kAttentionTaskPartitions,
     qd::kAttentionOutputPartitions},
    {qd::kAttentionTaskCount, 1},
    {qd::kAttentionTaskCount + 1, qd::kFfnProjectionPartitions},
    {qd::kAttentionTaskCount + 1 + qd::kFfnProjectionPartitions,
     qd::kDownProjectionPartitions},
    {qd::kTaskCount - 1, 1},
};

void fill_weight(std::vector<float> &weight, uint32_t salt) {
  for (size_t index = 0; index < weight.size(); ++index) {
    const int32_t centered =
        static_cast<int32_t>((index * (salt * 2u + 1u) + salt) % 31u) - 15;
    weight[index] = static_cast<float>(centered) * 0.0005f;
  }
}

void validate_layer_shape(
    const lrrt::executor::qwen::DecoderLayerWeights &weights) {
  if (weights.shape.hidden != qd::kHiddenSize ||
      weights.shape.heads != qd::kAttentionHeads ||
      weights.shape.kv_heads != qd::kKvHeads ||
      weights.shape.head_dim != qd::kHeadSize ||
      weights.shape.intermediate != qd::kIntermediateSize) {
    throw std::runtime_error(
        "checkpoint decoder shape does not match the megakernel");
  }
}

struct DeviceLayerWeights {
  lrrt::DeviceBuffer attention_norm;
  lrrt::DeviceBuffer q;
  lrrt::DeviceBuffer k;
  lrrt::DeviceBuffer v;
  lrrt::DeviceBuffer q_bias;
  lrrt::DeviceBuffer k_bias;
  lrrt::DeviceBuffer v_bias;
  lrrt::DeviceBuffer output;
  lrrt::DeviceBuffer mlp_norm;
  lrrt::DeviceBuffer gate;
  lrrt::DeviceBuffer up;
  lrrt::DeviceBuffer down;

  DeviceLayerWeights(lrrt::Device device,
                     const lrrt::executor::qwen::DecoderLayerWeights &weights)
      : attention_norm(device,
                       weights.attention_norm_weight.size() * sizeof(float)),
        q(device, weights.q_weight.size() * sizeof(float)),
        k(device, weights.k_weight.size() * sizeof(float)),
        v(device, weights.v_weight.size() * sizeof(float)),
        q_bias(device, weights.q_bias.size() * sizeof(float)),
        k_bias(device, weights.k_bias.size() * sizeof(float)),
        v_bias(device, weights.v_bias.size() * sizeof(float)),
        output(device, weights.out_weight.size() * sizeof(float)),
        mlp_norm(device, weights.mlp_norm_weight.size() * sizeof(float)),
        gate(device, weights.gate_weight.size() * sizeof(float)),
        up(device, weights.up_weight.size() * sizeof(float)),
        down(device, weights.down_weight.size() * sizeof(float)) {
    lrrt::copy_to_device(attention_norm, weights.attention_norm_weight);
    lrrt::copy_to_device(q, weights.q_weight);
    lrrt::copy_to_device(k, weights.k_weight);
    lrrt::copy_to_device(v, weights.v_weight);
    lrrt::copy_to_device(q_bias, weights.q_bias);
    lrrt::copy_to_device(k_bias, weights.k_bias);
    lrrt::copy_to_device(v_bias, weights.v_bias);
    lrrt::copy_to_device(output, weights.out_weight);
    lrrt::copy_to_device(mlp_norm, weights.mlp_norm_weight);
    lrrt::copy_to_device(gate, weights.gate_weight);
    lrrt::copy_to_device(up, weights.up_weight);
    lrrt::copy_to_device(down, weights.down_weight);
  }
};

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
    if (options.layers > std::numeric_limits<uint32_t>::max() / 2u / steps) {
      throw std::invalid_argument("steps times layers is too large");
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
    const uint32_t layers = options.layers;
    const uint32_t graph_steps = steps * layers;
    const uint32_t maximum_graph_steps = graph_steps * 2u;
    const uint32_t cache_capacity = steps * 2u;
    std::vector<float> initial_values(qd::kHiddenSize);
    for (uint32_t index = 0; index < qd::kHiddenSize; ++index) {
      initial_values[index] =
          0.03125f * static_cast<float>(static_cast<int32_t>(index % 31u) - 15);
    }
    const size_t q_elements =
        static_cast<size_t>(qd::kQuerySize) * qd::kHiddenSize;
    const size_t kv_elements =
        static_cast<size_t>(qd::kKvSize) * qd::kHiddenSize;
    const size_t gate_elements =
        static_cast<size_t>(qd::kIntermediateSize) * qd::kHiddenSize;
    const size_t down_elements =
        static_cast<size_t>(qd::kHiddenSize) * qd::kIntermediateSize;
    float rope_theta = 1000000.0f;
    std::vector<DeviceLayerWeights> device_layer_weights;
    device_layer_weights.reserve(options.weights_dir ? layers : 1u);
    if (options.weights_path || options.weights_dir) {
      for (uint32_t layer = 0; layer < layers; ++layer) {
        const std::filesystem::path manifest =
            options.weights_path
                ? std::filesystem::path(options.weights_path)
                : std::filesystem::path(options.weights_dir) /
                      ("layer_" + std::to_string(layer)) / "weights.json";
        lrrt::executor::qwen::DecoderLayerWeights weights =
            lrrt::executor::qwen::load_decoder_layer_weights(
                manifest.string().c_str());
        validate_layer_shape(weights);
        if (layer == 0) {
          rope_theta = weights.rope_theta;
        } else if (weights.rope_theta != rope_theta) {
          throw std::runtime_error("decoder layers use different rope_theta");
        }
        device_layer_weights.emplace_back(device, weights);
      }
    } else {
      lrrt::executor::qwen::DecoderLayerWeights weights{};
      weights.shape = {12,           qd::kHiddenSize, qd::kAttentionHeads,
                       qd::kKvHeads, qd::kHeadSize,   qd::kIntermediateSize};
      weights.rope_theta = rope_theta;
      weights.attention_norm_weight.resize(qd::kHiddenSize);
      weights.mlp_norm_weight.resize(qd::kHiddenSize);
      for (uint32_t index = 0; index < qd::kHiddenSize; ++index) {
        weights.attention_norm_weight[index] =
            1.0f + 0.001f * static_cast<float>(index % 23u);
        weights.mlp_norm_weight[index] =
            1.0f + 0.001f * static_cast<float>(index % 29u);
      }
      weights.q_weight.resize(q_elements);
      weights.k_weight.resize(kv_elements);
      weights.v_weight.resize(kv_elements);
      weights.q_bias.resize(qd::kQuerySize);
      weights.k_bias.resize(qd::kKvSize);
      weights.v_bias.resize(qd::kKvSize);
      weights.out_weight.resize(q_elements);
      weights.gate_weight.resize(gate_elements);
      weights.up_weight.resize(gate_elements);
      weights.down_weight.resize(down_elements);
      fill_weight(weights.q_weight, 1);
      fill_weight(weights.k_weight, 2);
      fill_weight(weights.v_weight, 3);
      fill_weight(weights.q_bias, 4);
      fill_weight(weights.k_bias, 5);
      fill_weight(weights.v_bias, 6);
      fill_weight(weights.out_weight, 7);
      fill_weight(weights.gate_weight, 8);
      fill_weight(weights.up_weight, 9);
      fill_weight(weights.down_weight, 10);
      device_layer_weights.emplace_back(device, weights);
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
    lrrt::DeviceBuffer rope_cos_buffer(device, rope_cos.size() * sizeof(float));
    lrrt::DeviceBuffer rope_sin_buffer(device, rope_sin.size() * sizeof(float));
    lrrt::DeviceBuffer resident_values(device, qd::kHiddenSize * sizeof(float));
    lrrt::DeviceBuffer resident_attention_normalized(device, qd::kHiddenSize *
                                                                 sizeof(float));
    lrrt::DeviceBuffer resident_query(device, qd::kQuerySize * sizeof(float));
    lrrt::DeviceBuffer resident_key(device, qd::kKvSize * sizeof(float));
    lrrt::DeviceBuffer resident_value(device, qd::kKvSize * sizeof(float));
    lrrt::DeviceBuffer resident_key_cache(device, layers * cache_elements *
                                                      sizeof(float));
    lrrt::DeviceBuffer resident_value_cache(device, layers * cache_elements *
                                                        sizeof(float));
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
    lrrt::DeviceBuffer host_key_cache(device,
                                      layers * cache_elements * sizeof(float));
    lrrt::DeviceBuffer host_value_cache(device, layers * cache_elements *
                                                    sizeof(float));
    lrrt::DeviceBuffer host_mlp_normalized(device,
                                           qd::kHiddenSize * sizeof(float));
    lrrt::DeviceBuffer host_activated(device,
                                      qd::kIntermediateSize * sizeof(float));
    lrrt::DeviceBuffer host_projected(device, qd::kHiddenSize * sizeof(float));
    lrrt::DeviceBuffer host_state(device, sizeof(sg::State));
    lrrt::copy_to_device(task_buffer, tasks);
    lrrt::copy_to_device(rope_cos_buffer, rope_cos);
    lrrt::copy_to_device(rope_sin_buffer, rope_sin);

    std::vector<qd::Context> resident_contexts;
    std::vector<qd::Context> host_contexts;
    resident_contexts.reserve(layers);
    host_contexts.reserve(layers);
    for (uint32_t layer = 0; layer < layers; ++layer) {
      DeviceLayerWeights &weights =
          device_layer_weights[options.weights_dir ? layer : 0u];
      const size_t cache_offset = static_cast<size_t>(layer) * cache_elements;
      resident_contexts.push_back({
          static_cast<float *>(resident_values.data()),
          static_cast<const float *>(weights.attention_norm.data()),
          static_cast<const float *>(weights.q.data()),
          static_cast<const float *>(weights.k.data()),
          static_cast<const float *>(weights.v.data()),
          static_cast<const float *>(weights.q_bias.data()),
          static_cast<const float *>(weights.k_bias.data()),
          static_cast<const float *>(weights.v_bias.data()),
          static_cast<const float *>(weights.output.data()),
          static_cast<const float *>(weights.mlp_norm.data()),
          static_cast<const float *>(weights.gate.data()),
          static_cast<const float *>(weights.up.data()),
          static_cast<const float *>(weights.down.data()),
          static_cast<float *>(resident_attention_normalized.data()),
          static_cast<float *>(resident_query.data()),
          static_cast<float *>(resident_key.data()),
          static_cast<float *>(resident_value.data()),
          static_cast<const float *>(rope_cos_buffer.data()),
          static_cast<const float *>(rope_sin_buffer.data()),
          static_cast<float *>(resident_key_cache.data()) + cache_offset,
          static_cast<float *>(resident_value_cache.data()) + cache_offset,
          static_cast<float *>(resident_mlp_normalized.data()),
          static_cast<float *>(resident_activated.data()),
          static_cast<float *>(resident_projected.data()),
          qd::kHiddenSize,
          qd::kQuerySize,
          qd::kKvSize,
          qd::kIntermediateSize,
          cache_capacity,
          layers,
      });
      qd::Context host_context = resident_contexts.back();
      host_context.hidden = static_cast<float *>(host_values.data());
      host_context.attention_normalized =
          static_cast<float *>(host_attention_normalized.data());
      host_context.query = static_cast<float *>(host_query.data());
      host_context.key = static_cast<float *>(host_key.data());
      host_context.value = static_cast<float *>(host_value.data());
      host_context.key_cache =
          static_cast<float *>(host_key_cache.data()) + cache_offset;
      host_context.value_cache =
          static_cast<float *>(host_value_cache.data()) + cache_offset;
      host_context.mlp_normalized =
          static_cast<float *>(host_mlp_normalized.data());
      host_context.activated = static_cast<float *>(host_activated.data());
      host_context.projected = static_cast<float *>(host_projected.data());
      host_contexts.push_back(host_context);
    }
    lrrt::DeviceBuffer resident_context_buffer(
        device, resident_contexts.size() * sizeof(qd::Context));
    lrrt::DeviceBuffer host_context_buffer(device, host_contexts.size() *
                                                       sizeof(qd::Context));
    lrrt::copy_to_device(resident_context_buffer, resident_contexts);
    lrrt::copy_to_device(host_context_buffer, host_contexts);
    const std::vector<float> zero_caches(layers * cache_elements, 0.0f);
    lrrt::copy_to_device(resident_key_cache, zero_caches);
    lrrt::copy_to_device(resident_value_cache, zero_caches);
    lrrt::copy_to_device(host_key_cache, zero_caches);
    lrrt::copy_to_device(host_value_cache, zero_caches);

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
    initialize_state(resident_state, qd::kTaskCount, maximum_graph_steps,
                     steps);
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
      initialize_state(resident_state, qd::kTaskCount, maximum_graph_steps,
                       steps);
      lrrt::copy_to_device(resident_values, initial_values);
      lrrt::launch(resident, resident_config, resident_args);
      device.synchronize();
      initialize_state(host_state, qd::kTaskCount, maximum_graph_steps, steps);
      lrrt::copy_to_device(host_values, initial_values);
      for (uint32_t step = 0; step < steps; ++step) {
        for (uint32_t layer = 0; layer < layers; ++layer) {
          for (const HostLaunch &launch : host_launches) {
            lrrt::launch(host_step, launch.config, launch.args);
          }
        }
      }
      device.synchronize();
    }

    const Measurement resident_measurement = measure(
        device, trials, resident_values, initial_values,
        [&] {
          initialize_state(resident_state, qd::kTaskCount, maximum_graph_steps,
                           steps);
        },
        [&] { lrrt::launch(resident, resident_config, resident_args); });
    const Measurement host_measurement = measure(
        device, trials, host_values, initial_values,
        [&] {
          initialize_state(host_state, qd::kTaskCount, maximum_graph_steps,
                           steps);
        },
        [&] {
          for (uint32_t step = 0; step < steps; ++step) {
            for (uint32_t layer = 0; layer < layers; ++layer) {
              for (const HostLaunch &launch : host_launches) {
                lrrt::launch(host_step, launch.config, launch.args);
              }
            }
          }
        });

    sg::State final_state{};
    lrrt::copy_to_host(&final_state, resident_state, sizeof(final_state));
    if (final_state.status != static_cast<uint32_t>(sg::Status::complete) ||
        final_state.step != graph_steps ||
        final_state.predicate_value != steps ||
        final_state.executed_tasks !=
            static_cast<uint64_t>(graph_steps) * qd::kTaskCount) {
      throw std::runtime_error(
          "resident graph did not stop at the device predicate");
    }
    sg::State final_host_state{};
    lrrt::copy_to_host(&final_host_state, host_state, sizeof(final_host_state));
    if (final_host_state.predicate_value != steps ||
        final_host_state.step != graph_steps) {
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
    std::vector<float> resident_key_cache_result(layers * cache_elements);
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
    std::printf("Decoder layers:           %u\n", layers);
    std::printf("Weights:                  %s\n",
                options.weights_dir
                    ? options.weights_dir
                    : (options.weights_path ? options.weights_path
                                            : "deterministic synthetic f32"));
    std::printf("Tasks per layer:          %u dependency-scheduled tasks\n",
                qd::kTaskCount);
    std::printf("Conditional tokens:       %u (max=%u)\n", steps, steps * 2u);
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
                graph_steps * static_cast<uint32_t>(host_launches.size()));
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
