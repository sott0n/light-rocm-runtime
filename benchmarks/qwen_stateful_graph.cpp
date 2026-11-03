#include "executor/qwen/stateful_graph_context.hpp"
#include "executor/qwen/weight_bundle.hpp"
#include "executor/stateful_graph/resident_executor.hpp"
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

#ifndef LRRT_QWEN_STATEFUL_GRAPH_HSACO
#define LRRT_QWEN_STATEFUL_GRAPH_HSACO "qwen_stateful_graph_kernel.hsaco"
#endif

namespace {

namespace sg = lrrt::stateful_graph;
namespace qd = lrrt::qwen::stateful_graph;
using Clock = std::chrono::steady_clock;

constexpr uint32_t kBlockSize = sg::kWorkerThreads;
constexpr uint32_t kWarmupTrials = 5;
constexpr uint32_t kDefaultWorkers = 20;
constexpr uint32_t kMaximumWorkers = 20;
constexpr uint32_t kSyntheticVocabSize = 4096;

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
  const char *tail_path = nullptr;
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
    if (std::strcmp(argv[index], "--tail") == 0) {
      if (++index >= argc) {
        throw std::invalid_argument("--tail requires a manifest path");
      }
      options.tail_path = argv[index];
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
          "usage: lrrt_qwen_stateful_graph_benchmark [steps] [trials] "
          "[workers] "
          "[--weights weights.json | --weights-dir directory --layers N] "
          "[--tail model_tail/weights.json]");
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

sg::Program make_program(uint32_t task_offset, uint32_t task_count,
                         uint32_t next_program, uint32_t flags = 0,
                         uint32_t branch_program = 0) {
  return {task_offset, task_count,     next_program,
          flags,       branch_program, {0, 0, 0}};
}

struct ProgramTable {
  std::vector<sg::Task> tasks;
  std::vector<sg::Program> programs;
};

ProgramTable make_qwen_decoder_programs(uint32_t vocab_size,
                                        uint32_t lm_head_partitions) {
  if (vocab_size == 0 || lm_head_partitions == 0 ||
      lm_head_partitions > qd::kMaximumLmHeadPartitions ||
      lm_head_partitions > vocab_size) {
    throw std::invalid_argument("invalid LM head partitioning");
  }
  ProgramTable table;
  std::vector<sg::Task> &tasks = table.tasks;
  tasks.reserve(qd::kDecoderTaskCount + lm_head_partitions + 3u);
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

  for (uint32_t partition = 0; partition < qd::kAttentionOutputPartitions;
       ++partition) {
    tasks.push_back(make_task(
        qd::Opcode::attention_output_residual_f32, attention_mask,
        qd::kHiddenSize * partition / qd::kAttentionOutputPartitions,
        qd::kHiddenSize * (partition + 1u) / qd::kAttentionOutputPartitions));
  }
  if (tasks.size() != qd::kAttentionTaskCount) {
    throw std::runtime_error("Qwen attention program layout mismatch");
  }
  table.programs.push_back(make_program(0, qd::kAttentionTaskCount, 1));

  const uint32_t ffn_offset = static_cast<uint32_t>(tasks.size());
  tasks.push_back(make_task(qd::Opcode::mlp_rmsnorm_f32, 0, 0, qd::kHiddenSize,
                            float_bits(1.0e-6f)));

  uint64_t activated_mask = 0;
  for (uint32_t partition = 0; partition < qd::kFfnProjectionPartitions;
       ++partition) {
    const uint32_t local_task_index =
        static_cast<uint32_t>(tasks.size()) - ffn_offset;
    tasks.push_back(make_task(qd::Opcode::gate_up_silu_f32, 1u,
                              qd::kIntermediateSize * partition /
                                  qd::kFfnProjectionPartitions,
                              qd::kIntermediateSize * (partition + 1u) /
                                  qd::kFfnProjectionPartitions));
    activated_mask |= static_cast<uint64_t>(1) << local_task_index;
  }

  uint64_t down_mask = 0;
  for (uint32_t partition = 0; partition < qd::kDownProjectionPartitions;
       ++partition) {
    const uint32_t local_task_index =
        static_cast<uint32_t>(tasks.size()) - ffn_offset;
    tasks.push_back(make_task(
        qd::Opcode::down_matvec_f32, activated_mask,
        qd::kHiddenSize * partition / qd::kDownProjectionPartitions,
        qd::kHiddenSize * (partition + 1u) / qd::kDownProjectionPartitions));
    down_mask |= static_cast<uint64_t>(1) << local_task_index;
  }

  tasks.push_back(make_task(qd::Opcode::residual_advance, down_mask, 0,
                            qd::kHiddenSize, 1, 1));
  if (tasks.size() != qd::kDecoderTaskCount) {
    throw std::runtime_error("Qwen decoder task layout mismatch");
  }
  table.programs.push_back(
      make_program(ffn_offset, qd::kFfnTaskCount, 0,
                   sg::kProgramAdvancesStep | sg::kProgramBranchesOnState, 2));

  const uint32_t tail_offset = static_cast<uint32_t>(tasks.size());
  tasks.push_back(make_task(qd::Opcode::final_rmsnorm_f32, 0, 0,
                            qd::kHiddenSize, float_bits(1.0e-6f)));
  uint64_t partition_mask = 0;
  for (uint32_t partition = 0; partition < lm_head_partitions; ++partition) {
    const uint32_t local_task_index =
        static_cast<uint32_t>(tasks.size()) - tail_offset;
    tasks.push_back(
        make_task(qd::Opcode::lm_head_top1_partition_f32, 1u,
                  static_cast<uint32_t>(static_cast<uint64_t>(vocab_size) *
                                        partition / lm_head_partitions),
                  static_cast<uint32_t>(static_cast<uint64_t>(vocab_size) *
                                        (partition + 1u) / lm_head_partitions),
                  partition));
    partition_mask |= static_cast<uint64_t>(1) << local_task_index;
  }
  const uint32_t reduce_task_index =
      static_cast<uint32_t>(tasks.size()) - tail_offset;
  tasks.push_back(make_task(qd::Opcode::top1_reduce, partition_mask, 0,
                            lm_head_partitions));
  const uint64_t reduce_mask = static_cast<uint64_t>(1) << reduce_task_index;
  tasks.push_back(make_task(qd::Opcode::embedding_feedback_f32, reduce_mask, 0,
                            qd::kHiddenSize));
  table.programs.push_back(make_program(tail_offset, lm_head_partitions + 3u, 0,
                                        sg::kProgramEvaluatesPredicate));
  return table;
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
    {qd::kDecoderTaskCount - 1, 1},
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

void validate_tail_shape(
    const lrrt::executor::qwen::ModelTailWeights &weights) {
  if (weights.hidden != qd::kHiddenSize || weights.vocab == 0 ||
      weights.token_ids.size() != weights.vocab ||
      weights.token_embeddings.size() !=
          static_cast<size_t>(weights.vocab) * qd::kHiddenSize ||
      weights.final_norm_weight.size() != qd::kHiddenSize ||
      weights.lm_head_weight.size() !=
          static_cast<size_t>(weights.vocab) * qd::kHiddenSize) {
    throw std::runtime_error(
        "checkpoint model tail requires full-vocabulary embeddings");
  }
  for (uint32_t token = 0; token < weights.vocab; ++token) {
    if (weights.token_ids[token] != token) {
      throw std::runtime_error(
          "checkpoint model tail embeddings must use vocabulary order");
    }
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

struct DeviceTailWeights {
  lrrt::DeviceBuffer embeddings;
  lrrt::DeviceBuffer final_norm;
  lrrt::DeviceBuffer lm_head;

  DeviceTailWeights(lrrt::Device device,
                    const lrrt::executor::qwen::ModelTailWeights &weights)
      : embeddings(device, weights.token_embeddings.size() * sizeof(float)),
        final_norm(device, weights.final_norm_weight.size() * sizeof(float)),
        lm_head(device, weights.lm_head_weight.size() * sizeof(float)) {
    lrrt::copy_to_device(embeddings, weights.token_embeddings);
    lrrt::copy_to_device(final_norm, weights.final_norm_weight);
    lrrt::copy_to_device(lm_head, weights.lm_head_weight);
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
      std::fprintf(stderr, "qwen_stateful_graph_benchmark: no GPU devices\n");
      return 1;
    }
    lrrt::Device device = runtime.open_device(0);
    lrrt::Module module(device, read_file(LRRT_QWEN_STATEFUL_GRAPH_HSACO));
    lrrt::Kernel resident =
        module.kernel("qwen_stateful_graph_resident_kernel");
    lrrt::Kernel compiled_resident =
        module.kernel("qwen_stateful_graph_compiled_kernel");
    lrrt::Kernel host_step =
        module.kernel("qwen_stateful_graph_host_step_kernel");

    lrrt::executor::qwen::ModelTailWeights tail_weights{};
    std::string tail_source;
    std::filesystem::path inferred_tail;
    if (options.tail_path) {
      inferred_tail = options.tail_path;
    } else if (options.weights_dir) {
      const std::filesystem::path candidate =
          std::filesystem::path(options.weights_dir) / "model_tail" /
          "weights.json";
      if (std::filesystem::exists(candidate)) {
        inferred_tail = candidate;
      }
    }
    if (!inferred_tail.empty()) {
      tail_weights = lrrt::executor::qwen::load_model_tail_weights(
          inferred_tail.string().c_str());
      tail_source = inferred_tail.string();
    } else {
      tail_weights.hidden = qd::kHiddenSize;
      tail_weights.vocab = kSyntheticVocabSize;
      tail_weights.token_ids.resize(tail_weights.vocab);
      tail_weights.token_embeddings.resize(
          static_cast<size_t>(tail_weights.vocab) * qd::kHiddenSize);
      tail_weights.final_norm_weight.resize(qd::kHiddenSize);
      tail_weights.lm_head_weight.resize(
          static_cast<size_t>(tail_weights.vocab) * qd::kHiddenSize);
      for (uint32_t token = 0; token < tail_weights.vocab; ++token) {
        tail_weights.token_ids[token] = token;
      }
      for (uint32_t index = 0; index < qd::kHiddenSize; ++index) {
        tail_weights.final_norm_weight[index] =
            1.0f + 0.001f * static_cast<float>(index % 19u);
      }
      fill_weight(tail_weights.token_embeddings, 12);
      fill_weight(tail_weights.lm_head_weight, 11);
      tail_source = "deterministic synthetic f32";
    }
    validate_tail_shape(tail_weights);
    const uint32_t lm_head_partitions =
        std::min(tail_weights.vocab, qd::kMaximumLmHeadPartitions);
    DeviceTailWeights device_tail_weights(device, tail_weights);

    const ProgramTable qwen_program_table =
        make_qwen_decoder_programs(tail_weights.vocab, lm_head_partitions);
    const std::vector<sg::Task> &tasks = qwen_program_table.tasks;
    const std::vector<sg::Program> &programs = qwen_program_table.programs;
    const uint32_t layers = options.layers;
    const uint32_t graph_steps = steps * layers;
    const uint32_t maximum_graph_steps = graph_steps * 2u;
    const uint64_t expected_program_phases_u64 =
        static_cast<uint64_t>(steps) *
        (2u * static_cast<uint64_t>(layers) + 1u);
    const uint64_t phase_state_count_u64 =
        static_cast<uint64_t>(steps) * 2u *
            (2u * static_cast<uint64_t>(layers) + 1u) +
        1u;
    if (phase_state_count_u64 >
            static_cast<uint64_t>(sg::phase_generation(UINT32_MAX)) + 1u ||
        expected_program_phases_u64 >
            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      throw std::invalid_argument("program phase count is too large");
    }
    const uint32_t expected_program_phases =
        static_cast<uint32_t>(expected_program_phases_u64);
    const uint32_t phase_state_count =
        static_cast<uint32_t>(phase_state_count_u64);
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

    const sg::ResidentConfig resident_config{phase_state_count, workers};
    const sg::CompiledPlan compiled_plan =
        sg::compile_resident_plan(tasks, programs, resident_config);
    sg::ResidentExecutor dynamic_executor(device, resident, tasks, programs,
                                          resident_config);
    sg::ResidentExecutor resident_executor(device, compiled_resident, tasks,
                                           programs, resident_config,
                                           compiled_plan);
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
    lrrt::DeviceBuffer resident_final_normalized(device, qd::kHiddenSize *
                                                             sizeof(float));
    lrrt::DeviceBuffer resident_partial_max_values(device, lm_head_partitions *
                                                               sizeof(float));
    lrrt::DeviceBuffer resident_partial_max_indices(
        device, lm_head_partitions * sizeof(uint32_t));
    lrrt::DeviceBuffer resident_decode_state(device, sizeof(qd::DecodeState));
    lrrt::DeviceBuffer resident_output_tokens(device, steps * sizeof(uint32_t));
    lrrt::DeviceBuffer host_final_normalized(device,
                                             qd::kHiddenSize * sizeof(float));
    lrrt::DeviceBuffer host_partial_max_values(device, lm_head_partitions *
                                                           sizeof(float));
    lrrt::DeviceBuffer host_partial_max_indices(device, lm_head_partitions *
                                                            sizeof(uint32_t));
    lrrt::DeviceBuffer host_decode_state(device, sizeof(qd::DecodeState));
    lrrt::DeviceBuffer host_output_tokens(device, steps * sizeof(uint32_t));
    lrrt::DeviceBuffer resident_tail_context_buffer(device,
                                                    sizeof(qd::TailContext));
    lrrt::DeviceBuffer host_tail_context_buffer(device,
                                                sizeof(qd::TailContext));
    const qd::TailContext resident_tail_context = {
        static_cast<const float *>(device_tail_weights.final_norm.data()),
        static_cast<const float *>(device_tail_weights.lm_head.data()),
        static_cast<const float *>(device_tail_weights.embeddings.data()),
        static_cast<float *>(resident_final_normalized.data()),
        static_cast<float *>(resident_partial_max_values.data()),
        static_cast<uint32_t *>(resident_partial_max_indices.data()),
        static_cast<qd::DecodeState *>(resident_decode_state.data()),
        static_cast<uint32_t *>(resident_output_tokens.data()),
        tail_weights.vocab,
        lm_head_partitions,
        static_cast<uint32_t>(tail_weights.token_ids.size()),
        0,
    };
    const qd::TailContext host_tail_context = {
        static_cast<const float *>(device_tail_weights.final_norm.data()),
        static_cast<const float *>(device_tail_weights.lm_head.data()),
        static_cast<const float *>(device_tail_weights.embeddings.data()),
        static_cast<float *>(host_final_normalized.data()),
        static_cast<float *>(host_partial_max_values.data()),
        static_cast<uint32_t *>(host_partial_max_indices.data()),
        static_cast<qd::DecodeState *>(host_decode_state.data()),
        static_cast<uint32_t *>(host_output_tokens.data()),
        tail_weights.vocab,
        lm_head_partitions,
        static_cast<uint32_t>(tail_weights.token_ids.size()),
        0,
    };
    lrrt::copy_to_device(resident_tail_context_buffer, &resident_tail_context,
                         sizeof(resident_tail_context));
    lrrt::copy_to_device(host_tail_context_buffer, &host_tail_context,
                         sizeof(host_tail_context));
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
          static_cast<qd::TailContext *>(resident_tail_context_buffer.data()),
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
      host_context.tail =
          static_cast<qd::TailContext *>(host_tail_context_buffer.data());
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

    dynamic_executor.bind_context(resident_context_buffer.data());
    resident_executor.bind_context(resident_context_buffer.data());
    const auto *device_tasks = resident_executor.device_tasks();
    struct HostLaunch {
      sg::HostStepArgs args;
      lr_launch_config_t config;
    };
    std::vector<HostLaunch> decoder_host_launches;
    std::vector<HostLaunch> tail_host_launches;
    const auto append_host_launch = [&](std::vector<HostLaunch> &launches,
                                        HostStage stage) {
      launches.push_back({
          {
              static_cast<sg::State *>(host_state.data()),
              device_tasks + stage.task_begin,
              stage.task_count,
              0,
              host_context_buffer.data(),
          },
          {{stage.task_count * kBlockSize, 1, 1}, {kBlockSize, 1, 1}, 0},
      });
    };
    for (const HostStage stage : kHostStages) {
      append_host_launch(decoder_host_launches, stage);
    }
    append_host_launch(tail_host_launches, {qd::kDecoderTaskCount, 1});
    append_host_launch(tail_host_launches,
                       {qd::kDecoderTaskCount + 1u, lm_head_partitions});
    append_host_launch(tail_host_launches,
                       {qd::kDecoderTaskCount + 1u + lm_head_partitions, 1});
    append_host_launch(tail_host_launches,
                       {qd::kDecoderTaskCount + 2u + lm_head_partitions, 1});
    const auto launch_host_decode = [&] {
      for (uint32_t step = 0; step < steps; ++step) {
        for (uint32_t layer = 0; layer < layers; ++layer) {
          for (const HostLaunch &launch : decoder_host_launches) {
            lrrt::launch(host_step, launch.config, launch.args);
          }
        }
        for (const HostLaunch &launch : tail_host_launches) {
          lrrt::launch(host_step, launch.config, launch.args);
        }
      }
    };

    const auto initialize_host_state =
        [&](uint32_t program_index, uint32_t max_steps, uint32_t stop_value) {
          sg::State state{};
          state.abi_version = sg::kAbiVersion;
          state.status = static_cast<uint32_t>(sg::Status::ready);
          state.phase = sg::make_phase(0, program_index);
          state.max_steps = max_steps;
          state.stop_value = stop_value;
          state.worker_count = workers;
          lrrt::copy_to_device(host_state, &state, sizeof(state));
        };
    const auto initialize_decode_state = [&](lrrt::DeviceBuffer &buffer,
                                             uint32_t maximum_tokens,
                                             uint32_t eos_token = UINT32_MAX) {
      const qd::DecodeState state = {
          0,
          0,
          0,
          eos_token,
          0,
          maximum_tokens,
          static_cast<uint32_t>(qd::StopReason::none),
          0.0f,
      };
      lrrt::copy_to_device(buffer, &state, sizeof(state));
    };

    // Exercise both 64-bit mask words, including task bit 63, before timing.
    const std::vector<sg::Task> capacity_tasks =
        make_scheduler_capacity_tasks();
    const std::vector<sg::Program> capacity_programs = {make_program(
        0, sg::kMaximumTasks, 0,
        sg::kProgramAdvancesStep | sg::kProgramEvaluatesPredicate)};
    const std::vector<float> zero_projected(qd::kHiddenSize, 0.0f);
    dynamic_executor.upload_graph(capacity_tasks, capacity_programs);
    lrrt::copy_to_device(resident_projected, zero_projected);
    initialize_decode_state(resident_decode_state, 1);
    dynamic_executor.initialize(0, 1, sg::kMaximumTasks);
    dynamic_executor.launch();
    device.synchronize();
    const sg::State capacity_state = dynamic_executor.read_state();
    if (capacity_state.status != static_cast<uint32_t>(sg::Status::complete) ||
        capacity_state.step != 1 ||
        capacity_state.predicate_value != sg::kMaximumTasks ||
        capacity_state.executed_tasks != sg::kMaximumTasks) {
      throw std::runtime_error("resident graph did not execute 64-task DAG");
    }

    // Exercise device-side program validation before timing. Making the root
    // depend on the final attention task closes that program into a cycle.
    std::vector<sg::Task> cyclic_tasks = tasks;
    const uint64_t final_task_mask = static_cast<uint64_t>(1)
                                     << (qd::kAttentionTaskCount - 1u);
    cyclic_tasks[0].dependency_mask = static_cast<uint32_t>(final_task_mask);
    cyclic_tasks[0].dependency_mask_high =
        static_cast<uint32_t>(final_task_mask >> 32u);
    dynamic_executor.upload_graph(cyclic_tasks, programs);
    initialize_decode_state(resident_decode_state, steps);
    dynamic_executor.initialize(0, maximum_graph_steps, 1);
    dynamic_executor.launch();
    device.synchronize();
    const sg::State invalid_state = dynamic_executor.read_state();
    if (invalid_state.status !=
            static_cast<uint32_t>(sg::Status::invalid_program) ||
        invalid_state.error_code != 1) {
      throw std::runtime_error("resident graph did not reject a cyclic DAG");
    }
    dynamic_executor.upload_graph(tasks, programs);

    // Determine the first greedy token, then prove that the same graph stops
    // after one token when that value is configured as EOS. This exercises the
    // device-side conditional exit independently of the maximum-token limit.
    initialize_decode_state(resident_decode_state, 1);
    resident_executor.initialize(0, maximum_graph_steps, 1);
    lrrt::copy_to_device(resident_values, initial_values);
    resident_executor.launch();
    device.synchronize();
    qd::DecodeState first_token_decode{};
    lrrt::copy_to_host(&first_token_decode, resident_decode_state,
                       sizeof(first_token_decode));
    initialize_decode_state(resident_decode_state, std::max(steps, 2u),
                            first_token_decode.sampled_token);
    resident_executor.initialize(0, maximum_graph_steps, 1);
    lrrt::copy_to_device(resident_values, initial_values);
    resident_executor.launch();
    device.synchronize();
    const sg::State eos_state = resident_executor.read_state();
    qd::DecodeState eos_decode{};
    lrrt::copy_to_host(&eos_decode, resident_decode_state, sizeof(eos_decode));
    if (eos_state.status != static_cast<uint32_t>(sg::Status::complete) ||
        eos_state.step != layers || eos_decode.generated_tokens != 1 ||
        eos_decode.stop_reason !=
            static_cast<uint32_t>(qd::StopReason::eos_token) ||
        eos_decode.sampled_token != first_token_decode.sampled_token) {
      throw std::runtime_error("resident graph did not stop at EOS");
    }

    for (uint32_t trial = 0; trial < kWarmupTrials; ++trial) {
      initialize_decode_state(resident_decode_state, steps);
      dynamic_executor.initialize(0, maximum_graph_steps, 1);
      lrrt::copy_to_device(resident_values, initial_values);
      dynamic_executor.launch();
      device.synchronize();
      initialize_decode_state(resident_decode_state, steps);
      resident_executor.initialize(0, maximum_graph_steps, 1);
      lrrt::copy_to_device(resident_values, initial_values);
      resident_executor.launch();
      device.synchronize();
      initialize_decode_state(host_decode_state, steps);
      initialize_host_state(0, maximum_graph_steps, 1);
      lrrt::copy_to_device(host_values, initial_values);
      launch_host_decode();
      device.synchronize();
    }

    const Measurement dynamic_measurement = measure(
        device, trials, resident_values, initial_values,
        [&] {
          initialize_decode_state(resident_decode_state, steps);
          dynamic_executor.initialize(0, maximum_graph_steps, 1);
        },
        [&] { dynamic_executor.launch(); });
    const Measurement resident_measurement = measure(
        device, trials, resident_values, initial_values,
        [&] {
          initialize_decode_state(resident_decode_state, steps);
          resident_executor.initialize(0, maximum_graph_steps, 1);
        },
        [&] { resident_executor.launch(); });
    const Measurement host_measurement = measure(
        device, trials, host_values, initial_values,
        [&] {
          initialize_decode_state(host_decode_state, steps);
          initialize_host_state(0, maximum_graph_steps, 1);
        },
        launch_host_decode);

    const sg::State final_state = resident_executor.read_state();
    if (final_state.status != static_cast<uint32_t>(sg::Status::complete) ||
        final_state.step != graph_steps || final_state.predicate_value != 1 ||
        sg::phase_generation(final_state.phase) !=
            expected_program_phases - 1u ||
        sg::phase_program(final_state.phase) !=
            static_cast<uint32_t>(programs.size()) - 1u ||
        final_state.executed_tasks !=
            static_cast<uint64_t>(graph_steps) * qd::kDecoderTaskCount +
                static_cast<uint64_t>(steps) * (lm_head_partitions + 3u)) {
      throw std::runtime_error(
          "resident graph did not stop at the device predicate");
    }
    sg::State final_host_state{};
    lrrt::copy_to_host(&final_host_state, host_state, sizeof(final_host_state));
    if (final_host_state.predicate_value != 1 ||
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
    std::vector<float> resident_final_normalized_result(qd::kHiddenSize);
    std::vector<float> host_final_normalized_result(qd::kHiddenSize);
    lrrt::copy_to_host(resident_final_normalized_result,
                       resident_final_normalized);
    lrrt::copy_to_host(host_final_normalized_result, host_final_normalized);
    verify_values("final normalized", resident_final_normalized_result,
                  host_final_normalized_result);
    qd::DecodeState resident_decode{};
    qd::DecodeState host_decode{};
    lrrt::copy_to_host(&resident_decode, resident_decode_state,
                       sizeof(resident_decode));
    lrrt::copy_to_host(&host_decode, host_decode_state, sizeof(host_decode));
    if (resident_decode.generated_tokens != steps ||
        resident_decode.token_position != steps - 1u ||
        resident_decode.kv_length != steps - 1u ||
        resident_decode.stop_reason !=
            static_cast<uint32_t>(qd::StopReason::maximum_tokens) ||
        std::memcmp(&resident_decode, &host_decode, sizeof(resident_decode)) !=
            0) {
      throw std::runtime_error("resident decode state mismatch");
    }
    std::vector<uint32_t> resident_tokens(steps);
    std::vector<uint32_t> host_tokens(steps);
    lrrt::copy_to_host(resident_tokens.data(), resident_output_tokens,
                       resident_tokens.size() * sizeof(uint32_t));
    lrrt::copy_to_host(host_tokens.data(), host_output_tokens,
                       host_tokens.size() * sizeof(uint32_t));
    if (resident_tokens != host_tokens) {
      throw std::runtime_error("resident generated token sequence mismatch");
    }
    const uint32_t resident_top_token = resident_decode.sampled_token;
    const uint32_t host_top_token = host_decode.sampled_token;
    const float resident_top_logit = resident_decode.sampled_logit;
    const float host_top_logit = host_decode.sampled_logit;
    const float logit_tolerance =
        1.0e-5f * std::max(1.0f, std::fabs(host_top_logit));
    if (resident_top_token != host_top_token ||
        !std::isfinite(resident_top_logit) ||
        std::fabs(resident_top_logit - host_top_logit) > logit_tolerance) {
      throw std::runtime_error("resident top-1 result mismatch");
    }

    std::printf("\nLRRT Qwen Stateful Graph Benchmark\n");
    std::printf("==================================\n");
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
    std::printf("Model tail:               %s\n", tail_source.c_str());
    std::printf("Tasks per layer:          %u dependency-scheduled tasks\n",
                qd::kDecoderTaskCount);
    std::printf("Tail tasks:               %u (%u LM head partitions)\n",
                lm_head_partitions + 3u, lm_head_partitions);
    std::printf("Programs:                 %zu (attention -> FFN -> "
                "conditional LM head)\n",
                programs.size());
    std::printf("Compiled stages:          %zu static dependency levels\n",
                compiled_plan.stages.size());
    std::printf("Phase-state slots:        %u x %zu bytes\n", phase_state_count,
                sizeof(sg::MaskState));
    std::printf("Autoregressive tokens:    %u (device-resident feedback)\n",
                resident_decode.generated_tokens);
    std::printf("EOS termination:          verified with token %u\n",
                first_token_decode.sampled_token);
    std::printf("Tasks executed:           %llu\n",
                static_cast<unsigned long long>(final_state.executed_tasks));
    std::printf("Top-1 token:              %u (logit=%f, vocab=%u)\n",
                resident_top_token, resident_top_logit, tail_weights.vocab);
    std::printf("Measured trials:          %u\n\n", trials);
    std::printf("%-24s %12s %12s %12s\n", "Path", "Device us", "Wall us",
                "Launches");
    std::printf("%-24s %12.3f %12.3f %12u\n", "Dynamic resident graph",
                dynamic_measurement.device_ns / 1.0e3,
                dynamic_measurement.wall_ns / 1.0e3, 1u);
    std::printf("%-24s %12.3f %12.3f %12u\n", "Compiled resident graph",
                resident_measurement.device_ns / 1.0e3,
                resident_measurement.wall_ns / 1.0e3, 1u);
    std::printf(
        "%-24s %12.3f %12.3f %12u\n", "Host-driven steps",
        host_measurement.device_ns / 1.0e3, host_measurement.wall_ns / 1.0e3,
        steps * (layers * static_cast<uint32_t>(decoder_host_launches.size()) +
                 static_cast<uint32_t>(tail_host_launches.size())));
    std::printf("\nDynamic device speedup:  %.2fx\n",
                host_measurement.device_ns / dynamic_measurement.device_ns);
    std::printf("Compiled device speedup: %.2fx\n",
                host_measurement.device_ns / resident_measurement.device_ns);
    std::printf("Compiled wall speedup:   %.2fx\n",
                host_measurement.wall_ns / resident_measurement.wall_ns);
    std::printf("Compiled over dynamic:   %.2fx\n\n",
                dynamic_measurement.device_ns / resident_measurement.device_ns);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
