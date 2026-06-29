#include "mini_decoder_weights.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;

struct Colors {
  const char *title;
  const char *label;
  const char *time;
  const char *throughput;
  const char *reset;
};

struct BenchmarkCase {
  uint32_t keys;
  uint32_t hidden;
  uint32_t heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t intermediate;
  uint32_t valid_keys;
  std::vector<float> hidden_states;
  std::vector<float> attention_norm_weight;
  std::vector<float> mlp_norm_weight;
  std::vector<float> q_weight;
  std::vector<float> k_weight;
  std::vector<float> v_weight;
  std::vector<float> out_weight;
  std::vector<float> gate_weight;
  std::vector<float> up_weight;
  std::vector<float> down_weight;
  std::vector<float> cos;
  std::vector<float> sin;
};

struct Measurements {
  double cpu_round_trip_ns;
  double cpu_burst_interval_ns;
  double gpu_burst_interval_ns;
};

struct Options {
  uint32_t iterations;
  const char *weights_path;
  const char *weights_dir;
  uint32_t layers;
  uint32_t valid_keys;
  bool has_valid_keys;
};

uint32_t qkv_dim(const BenchmarkCase &benchmark_case) {
  return benchmark_case.heads * benchmark_case.head_dim;
}

uint32_t kv_dim(const BenchmarkCase &benchmark_case) {
  return benchmark_case.kv_heads * benchmark_case.head_dim;
}

uint32_t estimated_dispatches(const BenchmarkCase &benchmark_case) {
  return 10 + 2 * benchmark_case.valid_keys +
         2 * benchmark_case.kv_heads * benchmark_case.valid_keys +
         4 * benchmark_case.heads;
}

Colors output_colors() {
  const char *term = getenv("TERM");
  bool enabled = isatty(fileno(stdout)) && getenv("NO_COLOR") == nullptr &&
                 (!term || strcmp(term, "dumb") != 0);
  if (!enabled) {
    return {"", "", "", "", ""};
  }
  return {"\033[1;32m", "\033[1m", "\033[32m", "\033[1;32m", "\033[0m"};
}

uint32_t parse_u32(const char *text, const char *label) {
  if (!text || text[0] == '\0') {
    throw std::invalid_argument(std::string(label) + " is empty");
  }
  char *end = nullptr;
  unsigned long value = strtoul(text, &end, 10);
  if (!end || *end != '\0' || value == 0 ||
      value > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(std::string(label) +
                                " must be a positive uint32");
  }
  return static_cast<uint32_t>(value);
}

Options parse_options(int argc, char **argv) {
  Options options{20, nullptr, nullptr, 0, 0, false};
  bool saw_iterations = false;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--weights") == 0) {
      if (++i >= argc) {
        throw std::invalid_argument("--weights requires a manifest path");
      }
      options.weights_path = argv[i];
    } else if (strcmp(argv[i], "--weights-dir") == 0) {
      if (++i >= argc) {
        throw std::invalid_argument("--weights-dir requires a directory path");
      }
      options.weights_dir = argv[i];
    } else if (strcmp(argv[i], "--layers") == 0) {
      if (++i >= argc) {
        throw std::invalid_argument("--layers requires a count");
      }
      options.layers = parse_u32(argv[i], "layer count");
    } else if (strcmp(argv[i], "--valid-keys") == 0) {
      if (++i >= argc) {
        throw std::invalid_argument("--valid-keys requires a count");
      }
      options.valid_keys = parse_u32(argv[i], "valid key count");
      options.has_valid_keys = true;
    } else if (!saw_iterations) {
      options.iterations = parse_u32(argv[i], "benchmark count");
      saw_iterations = true;
    } else {
      throw std::invalid_argument(
          "usage: lrrt_triton_mini_decoder_layer_benchmark [count] "
          "[--weights weights.json | --weights-dir dir --layers count] "
          "[--valid-keys count]");
    }
  }
  if (options.weights_path && options.weights_dir) {
    throw std::invalid_argument(
        "--weights and --weights-dir are mutually exclusive");
  }
  if (options.weights_dir && options.layers == 0) {
    throw std::invalid_argument("--weights-dir requires --layers");
  }
  if (!options.weights_dir && options.layers != 0) {
    throw std::invalid_argument("--layers requires --weights-dir");
  }
  return options;
}

double elapsed_ns(Clock::time_point begin, Clock::time_point end) {
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count());
}

BenchmarkCase make_case(uint32_t keys, uint32_t hidden, uint32_t heads,
                        uint32_t kv_heads, uint32_t head_dim,
                        uint32_t intermediate, uint32_t valid_keys) {
  const uint32_t qkv_dim = heads * head_dim;
  const uint32_t kv_dim = kv_heads * head_dim;
  BenchmarkCase benchmark_case = {
      keys,
      hidden,
      heads,
      kv_heads,
      head_dim,
      intermediate,
      valid_keys,
      std::vector<float>(keys * hidden),
      std::vector<float>(hidden),
      std::vector<float>(hidden),
      std::vector<float>(qkv_dim * hidden),
      std::vector<float>(kv_dim * hidden),
      std::vector<float>(kv_dim * hidden),
      std::vector<float>(hidden * qkv_dim),
      std::vector<float>(intermediate * hidden),
      std::vector<float>(intermediate * hidden),
      std::vector<float>(hidden * intermediate),
      std::vector<float>(keys * (head_dim / 2)),
      std::vector<float>(keys * (head_dim / 2)),
  };

  for (uint32_t i = 0; i < benchmark_case.hidden_states.size(); ++i) {
    benchmark_case.hidden_states[i] =
        0.03125f * (float)((int32_t)((i * 5 + i / hidden) % 31) - 15);
  }
  for (uint32_t i = 0; i < hidden; ++i) {
    benchmark_case.attention_norm_weight[i] = 1.0f + 0.001f * (float)(i % 29);
    benchmark_case.mlp_norm_weight[i] = 1.0f - 0.001f * (float)(i % 17);
  }

  lrrt::executor::triton::mini::fill_projection_weight(benchmark_case.q_weight,
                                                       hidden, 1);
  lrrt::executor::triton::mini::fill_projection_weight(benchmark_case.k_weight,
                                                       hidden, 2);
  lrrt::executor::triton::mini::fill_projection_weight(benchmark_case.v_weight,
                                                       hidden, 3);
  lrrt::executor::triton::mini::fill_projection_weight(
      benchmark_case.out_weight, qkv_dim, 4);
  lrrt::executor::triton::mini::fill_projection_weight(
      benchmark_case.gate_weight, hidden, 5);
  lrrt::executor::triton::mini::fill_projection_weight(benchmark_case.up_weight,
                                                       hidden, 6);
  lrrt::executor::triton::mini::fill_projection_weight(
      benchmark_case.down_weight, intermediate, 7);

  const uint32_t half = head_dim / 2;
  for (uint32_t token = 0; token < keys; ++token) {
    for (uint32_t frequency = 0; frequency < half; ++frequency) {
      float exponent = -2.0f * (float)frequency / (float)head_dim;
      float inverse_frequency = powf(10000.0f, exponent);
      float angle = (float)(token + 7) * inverse_frequency;
      uint32_t index = token * half + frequency;
      benchmark_case.cos[index] = cosf(angle);
      benchmark_case.sin[index] = sinf(angle);
    }
  }
  return benchmark_case;
}

BenchmarkCase
make_case(const lrrt::executor::triton::mini::DecoderLayerWeights &weights,
          uint32_t valid_keys) {
  BenchmarkCase benchmark_case =
      make_case(weights.shape.keys, weights.shape.hidden, weights.shape.heads,
                weights.shape.kv_heads, weights.shape.head_dim,
                weights.shape.intermediate, valid_keys);
  benchmark_case.attention_norm_weight = weights.attention_norm_weight;
  benchmark_case.mlp_norm_weight = weights.mlp_norm_weight;
  benchmark_case.q_weight = weights.q_weight;
  benchmark_case.k_weight = weights.k_weight;
  benchmark_case.v_weight = weights.v_weight;
  benchmark_case.out_weight = weights.out_weight;
  benchmark_case.gate_weight = weights.gate_weight;
  benchmark_case.up_weight = weights.up_weight;
  benchmark_case.down_weight = weights.down_weight;
  return benchmark_case;
}

BenchmarkCase make_case(uint32_t keys, uint32_t hidden, uint32_t head_dim,
                        uint32_t intermediate, uint32_t valid_keys) {
  return make_case(keys, hidden, 1, 1, head_dim, intermediate, valid_keys);
}

void copy_inputs(lrrt::executor::triton::mini::DecoderLayer &executor,
                 const BenchmarkCase &benchmark_case) {
  executor.copy_inputs(benchmark_case.hidden_states,
                       benchmark_case.attention_norm_weight,
                       benchmark_case.mlp_norm_weight, benchmark_case.q_weight,
                       benchmark_case.k_weight, benchmark_case.v_weight,
                       benchmark_case.out_weight, benchmark_case.gate_weight,
                       benchmark_case.up_weight, benchmark_case.down_weight,
                       benchmark_case.cos, benchmark_case.sin);
}

std::vector<lrrt::executor::triton::mini::DecoderLayerWeights>
load_layer_weights(const char *weights_dir, uint32_t layers) {
  std::vector<lrrt::executor::triton::mini::DecoderLayerWeights> weights;
  weights.reserve(layers);
  const std::filesystem::path base(weights_dir);
  for (uint32_t layer = 0; layer < layers; ++layer) {
    const std::filesystem::path manifest =
        base / ("layer_" + std::to_string(layer)) / "weights.json";
    weights.push_back(lrrt::executor::triton::mini::load_decoder_layer_weights(
        manifest.string().c_str()));
  }
  return weights;
}

Measurements measure_case(lrrt::Device &device,
                          const BenchmarkCase &benchmark_case,
                          uint32_t iterations, uint32_t warmup_iterations) {
  lrrt::executor::triton::mini::DecoderLayer executor(
      device, benchmark_case.keys, benchmark_case.hidden, benchmark_case.heads,
      benchmark_case.kv_heads, benchmark_case.head_dim,
      benchmark_case.intermediate);
  copy_inputs(executor, benchmark_case);

  for (uint32_t i = 0; i < warmup_iterations; ++i) {
    executor.run(benchmark_case.valid_keys);
    executor.synchronize();
  }

  double cpu_round_trip_ns = 0.0;
  for (uint32_t i = 0; i < iterations; ++i) {
    auto begin = Clock::now();
    executor.run(benchmark_case.valid_keys);
    executor.synchronize();
    auto end = Clock::now();
    cpu_round_trip_ns += elapsed_ns(begin, end);
  }
  cpu_round_trip_ns /= static_cast<double>(iterations);

  auto begin = Clock::now();
  for (uint32_t i = 0; i < iterations; ++i) {
    executor.run(benchmark_case.valid_keys);
  }
  executor.synchronize();
  auto end = Clock::now();
  double cpu_burst_interval_ns =
      elapsed_ns(begin, end) / static_cast<double>(iterations);

  lrrt::Event gpu_start(device);
  lrrt::Event gpu_end(device);
  gpu_start.record(executor.queue());
  for (uint32_t i = 0; i < iterations; ++i) {
    executor.run(benchmark_case.valid_keys);
  }
  gpu_end.record(executor.queue());
  gpu_end.synchronize();
  gpu_start.synchronize();
  double gpu_burst_interval_ns =
      static_cast<double>(lrrt::elapsed_time_ns(gpu_start, gpu_end)) /
      static_cast<double>(iterations);

  return {cpu_round_trip_ns, cpu_burst_interval_ns, gpu_burst_interval_ns};
}

Measurements measure_stack_case(lrrt::Device &device,
                                const std::vector<BenchmarkCase> &layers,
                                uint32_t iterations,
                                uint32_t warmup_iterations) {
  if (layers.empty()) {
    throw std::runtime_error("decoder stack benchmark has no layers");
  }

  const BenchmarkCase &shape = layers.front();
  for (const BenchmarkCase &layer : layers) {
    if (layer.keys != shape.keys || layer.hidden != shape.hidden ||
        layer.heads != shape.heads || layer.kv_heads != shape.kv_heads ||
        layer.head_dim != shape.head_dim ||
        layer.intermediate != shape.intermediate ||
        layer.valid_keys != shape.valid_keys) {
      throw std::runtime_error(
          "decoder stack benchmark requires matching layer shapes");
    }
  }

  std::vector<std::unique_ptr<lrrt::executor::triton::mini::DecoderLayer>>
      executors;
  executors.reserve(layers.size());
  for (const BenchmarkCase &layer : layers) {
    executors.push_back(
        std::make_unique<lrrt::executor::triton::mini::DecoderLayer>(
            device, layer.keys, layer.hidden, layer.heads, layer.kv_heads,
            layer.head_dim, layer.intermediate));
    copy_inputs(*executors.back(), layer);
  }

  const uint32_t valid_keys = shape.valid_keys;
  const uint32_t hidden = shape.hidden;
  std::vector<float> current_hidden = shape.hidden_states;
  std::vector<float> next_hidden = current_hidden;
  std::vector<float> row(hidden);

  auto run_stack = [&]() {
    current_hidden = shape.hidden_states;
    for (size_t layer = 0; layer < layers.size(); ++layer) {
      next_hidden = current_hidden;
      executors[layer]->copy_hidden_states(current_hidden);
      for (uint32_t key = 1; key <= valid_keys; ++key) {
        executors[layer]->run(key);
        executors[layer]->synchronize();
        executors[layer]->copy_output(row);
        std::copy(row.begin(), row.end(),
                  next_hidden.begin() + static_cast<size_t>(key - 1) * hidden);
      }
      current_hidden.swap(next_hidden);
    }
  };

  for (uint32_t i = 0; i < warmup_iterations; ++i) {
    run_stack();
  }

  double cpu_round_trip_ns = 0.0;
  for (uint32_t i = 0; i < iterations; ++i) {
    auto begin = Clock::now();
    run_stack();
    auto end = Clock::now();
    cpu_round_trip_ns += elapsed_ns(begin, end);
  }
  cpu_round_trip_ns /= static_cast<double>(iterations);

  return {cpu_round_trip_ns, cpu_round_trip_ns, 0.0};
}

uint32_t estimated_stack_dispatches(const std::vector<BenchmarkCase> &layers) {
  if (layers.empty()) {
    return 0;
  }
  BenchmarkCase layer = layers.front();
  uint32_t total = 0;
  for (uint32_t key = 1; key <= layers.front().valid_keys; ++key) {
    layer.valid_keys = key;
    total += estimated_dispatches(layer) * static_cast<uint32_t>(layers.size());
  }
  return total;
}

void print_case(const BenchmarkCase &benchmark_case,
                const Measurements &measurements, const Colors &colors) {
  printf("%-26s %5u %7u %5u %7u %8u %7u %5u %12u %10u %10u %s%11.3f%s "
         "%s%11.3f%s %s%11.3f%s\n",
         "decoder layer", benchmark_case.keys, benchmark_case.hidden,
         benchmark_case.heads, benchmark_case.kv_heads, benchmark_case.head_dim,
         qkv_dim(benchmark_case), kv_dim(benchmark_case),
         benchmark_case.intermediate, benchmark_case.valid_keys,
         estimated_dispatches(benchmark_case), colors.time,
         measurements.cpu_round_trip_ns / 1.0e3, colors.reset, colors.time,
         measurements.cpu_burst_interval_ns / 1.0e3, colors.reset, colors.time,
         measurements.gpu_burst_interval_ns / 1.0e3, colors.reset);
}

void print_stack_case(const std::vector<BenchmarkCase> &layers,
                      const Measurements &measurements, const Colors &colors) {
  const BenchmarkCase &shape = layers.front();
  const uint32_t dispatches_per_stack = estimated_stack_dispatches(layers);
  printf("%-26s %5u %7u %5u %7u %8u %7u %5u %12u %10u %10u %s%11.3f%s "
         "%s%11s%s %s%11s%s\n",
         "decoder stack", shape.keys, shape.hidden, shape.heads, shape.kv_heads,
         shape.head_dim, qkv_dim(shape), kv_dim(shape), shape.intermediate,
         shape.valid_keys, dispatches_per_stack, colors.time,
         measurements.cpu_round_trip_ns / 1.0e3, colors.reset, colors.time,
         "host-chain", colors.reset, colors.time, "n/a", colors.reset);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    const uint32_t iterations = options.iterations;
    const uint32_t warmup_iterations = std::min(iterations, 5u);

    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      fprintf(stderr, "triton_mini_decoder_layer_benchmark: no GPU devices\n");
      return 1;
    }
    lrrt::Device device = runtime.open_device(0);
    std::vector<BenchmarkCase> cases;
    if (options.weights_path) {
      lrrt::executor::triton::mini::DecoderLayerWeights weights =
          lrrt::executor::triton::mini::load_decoder_layer_weights(
              options.weights_path);
      uint32_t valid_keys =
          options.has_valid_keys ? options.valid_keys : weights.shape.keys;
      if (valid_keys > weights.shape.keys) {
        throw std::invalid_argument(
            "valid key count exceeds weight shape keys");
      }
      cases.push_back(make_case(weights, valid_keys));
    } else if (options.weights_dir) {
      auto weights = load_layer_weights(options.weights_dir, options.layers);
      for (const auto &layer_weights : weights) {
        uint32_t valid_keys = options.has_valid_keys ? options.valid_keys
                                                     : layer_weights.shape.keys;
        if (valid_keys > layer_weights.shape.keys) {
          throw std::invalid_argument(
              "valid key count exceeds weight shape keys");
        }
        cases.push_back(make_case(layer_weights, valid_keys));
      }
    } else {
      cases.push_back(make_case(16, 768, 64, 2048, 7));
      cases.push_back(make_case(32, 768, 2, 2, 64, 2048, 19));
      cases.push_back(make_case(64, 1024, 2, 2, 128, 3072, 33));
      cases.push_back(make_case(16, 896, 14, 2, 64, 4864, 7));
    }

    const Colors colors = output_colors();
    printf("\n%sLRRT Triton Mini Decoder Layer Benchmark%s\n", colors.title,
           colors.reset);
    printf("%s========================================%s\n", colors.title,
           colors.reset);
    printf("Device index:       %u\n", device.index());
    printf("Device name:        %s\n", device.name().c_str());
    printf("Data type:          FP32 inputs / FP32 accumulation\n");
    printf("Cache layout:       [kv_heads, keys, head_dim]\n");
    printf("Queueing:           %s\n",
           options.weights_dir ? "ordered launches on one lrrt queue per layer"
                               : "ordered launches on one lrrt queue");
    printf("Timing source:      CPU steady_clock and HSA GPU event markers\n");
    printf("Weight source:      %s\n",
           options.weights_path
               ? options.weights_path
               : (options.weights_dir ? options.weights_dir : "synthetic"));
    if (options.weights_dir) {
      printf("Layer count:        %u\n", options.layers);
      printf("Stack handoff:      host copy between layers\n");
    }
    printf("Iterations:         %u\n", iterations);
    printf("Warm-up iterations: %u per shape\n\n", warmup_iterations);
    printf(
        "%s%-26s %5s %7s %5s %7s %8s %7s %5s %12s %10s %10s %11s %11s %11s%s\n",
        colors.label, "Workload", "Keys", "Hidden", "Heads", "KVHeads",
        "HeadDim", "QDim", "KVDim", "Intermediate", "ValidKeys", "Dispatches",
        "CPU round us", "CPU burst us", "GPU burst us", colors.reset);
    printf("%-26s %5s %7s %5s %7s %8s %7s %5s %12s %10s %10s %11s %11s %11s\n",
           "--------------------------", "-----", "-------", "-----", "-------",
           "--------", "-------", "-----", "------------", "----------",
           "----------", "-----------", "-----------", "-----------");

    if (options.weights_dir) {
      Measurements measurements =
          measure_stack_case(device, cases, iterations, warmup_iterations);
      print_stack_case(cases, measurements, colors);
    } else {
      for (const BenchmarkCase &benchmark_case : cases) {
        Measurements measurements =
            measure_case(device, benchmark_case, iterations, warmup_iterations);
        print_case(benchmark_case, measurements, colors);
      }
    }

    printf("\n%sCPU round%s measures executor.run() plus synchronize() for "
           "one layer per iteration with steady_clock.\n",
           colors.label, colors.reset);
    printf("%sCPU burst%s measures repeated executor.run() submissions "
           "followed by one final synchronize() with steady_clock.\n",
           colors.label, colors.reset);
    printf("%sGPU burst%s measures HSA event elapsed time around repeated "
           "executor.run() submissions, divided by iteration count.\n",
           colors.label, colors.reset);
    if (options.weights_dir) {
      printf("%sdecoder stack%s currently uses host-visible handoff between "
             "layers, so only CPU round-trip timing is meaningful for the "
             "stack row.\n",
             colors.label, colors.reset);
    }
    printf("%sDispatches%s is an estimate of kernel submissions per layer; "
           "multi-head attention currently dispatches per head.\n\n",
           colors.label, colors.reset);
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
