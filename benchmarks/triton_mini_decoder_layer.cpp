#include "mini_decoder_layer.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
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
  double round_trip_ns;
  double burst_interval_ns;
};

uint32_t qkv_dim(const BenchmarkCase &benchmark_case) {
  return benchmark_case.heads * benchmark_case.head_dim;
}

uint32_t estimated_dispatches(const BenchmarkCase &benchmark_case) {
  return 10 + 2 * benchmark_case.valid_keys +
         benchmark_case.heads * (4 + 2 * benchmark_case.valid_keys);
}

Colors output_colors() {
  const char *term = getenv("TERM");
  bool enabled = isatty(fileno(stdout)) && getenv("NO_COLOR") == nullptr &&
                 (!term || strcmp(term, "dumb") != 0);
  if (!enabled) {
    return {"", "", "", "", ""};
  }
  return {"\033[1;36m", "\033[1m", "\033[36m", "\033[1;32m", "\033[0m"};
}

uint32_t parse_iterations(int argc, char **argv) {
  if (argc > 2) {
    throw std::invalid_argument(
        "usage: lrrt_triton_mini_decoder_layer_benchmark [count]");
  }
  if (argc == 1) {
    return 20;
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

BenchmarkCase make_case(uint32_t keys, uint32_t hidden, uint32_t heads,
                        uint32_t head_dim, uint32_t intermediate,
                        uint32_t valid_keys) {
  const uint32_t qkv_dim = heads * head_dim;
  BenchmarkCase benchmark_case = {
      keys,
      hidden,
      heads,
      head_dim,
      intermediate,
      valid_keys,
      std::vector<float>(keys * hidden),
      std::vector<float>(hidden),
      std::vector<float>(hidden),
      std::vector<float>(qkv_dim * hidden),
      std::vector<float>(qkv_dim * hidden),
      std::vector<float>(qkv_dim * hidden),
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

BenchmarkCase make_case(uint32_t keys, uint32_t hidden, uint32_t head_dim,
                        uint32_t intermediate, uint32_t valid_keys) {
  return make_case(keys, hidden, 1, head_dim, intermediate, valid_keys);
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

Measurements measure_case(lrrt::Device &device,
                          const BenchmarkCase &benchmark_case,
                          uint32_t iterations, uint32_t warmup_iterations) {
  lrrt::executor::triton::mini::DecoderLayer executor(
      device, benchmark_case.keys, benchmark_case.hidden, benchmark_case.heads,
      benchmark_case.head_dim, benchmark_case.intermediate);
  copy_inputs(executor, benchmark_case);

  for (uint32_t i = 0; i < warmup_iterations; ++i) {
    executor.run(benchmark_case.valid_keys);
    executor.synchronize();
  }

  double round_trip_ns = 0.0;
  for (uint32_t i = 0; i < iterations; ++i) {
    auto begin = Clock::now();
    executor.run(benchmark_case.valid_keys);
    executor.synchronize();
    auto end = Clock::now();
    round_trip_ns += elapsed_ns(begin, end);
  }
  round_trip_ns /= static_cast<double>(iterations);

  auto begin = Clock::now();
  for (uint32_t i = 0; i < iterations; ++i) {
    executor.run(benchmark_case.valid_keys);
  }
  executor.synchronize();
  auto end = Clock::now();
  double burst_interval_ns =
      elapsed_ns(begin, end) / static_cast<double>(iterations);

  return {round_trip_ns, burst_interval_ns};
}

void print_case(const BenchmarkCase &benchmark_case,
                const Measurements &measurements, const Colors &colors) {
  printf("%-26s %5u %7u %5u %8u %7u %12u %10u %10u %s%11.3f us%s "
         "%s%11.3f us%s\n",
         "decoder layer", benchmark_case.keys, benchmark_case.hidden,
         benchmark_case.heads, benchmark_case.head_dim, qkv_dim(benchmark_case),
         benchmark_case.intermediate, benchmark_case.valid_keys,
         estimated_dispatches(benchmark_case), colors.time,
         measurements.round_trip_ns / 1.0e3, colors.reset, colors.time,
         measurements.burst_interval_ns / 1.0e3, colors.reset);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const uint32_t iterations = parse_iterations(argc, argv);
    const uint32_t warmup_iterations = std::min(iterations, 5u);

    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      fprintf(stderr, "triton_mini_decoder_layer_benchmark: no GPU devices\n");
      return 1;
    }
    lrrt::Device device = runtime.open_device(0);
    std::vector<BenchmarkCase> cases;
    cases.push_back(make_case(16, 768, 64, 2048, 7));
    cases.push_back(make_case(32, 768, 2, 64, 2048, 19));
    cases.push_back(make_case(64, 1024, 2, 128, 3072, 33));

    const Colors colors = output_colors();
    printf("\n%sLRRT Triton Mini Decoder Layer Benchmark%s\n", colors.title,
           colors.reset);
    printf("%s========================================%s\n", colors.title,
           colors.reset);
    printf("Device index:       %u\n", device.index());
    printf("Data type:          FP32 inputs / FP32 accumulation\n");
    printf("Cache layout:       [heads, keys, head_dim]\n");
    printf("Queueing:           ordered launches on one lrrt queue\n");
    printf("Timing source:      CPU steady_clock around executor calls\n");
    printf("Iterations:         %u\n", iterations);
    printf("Warm-up iterations: %u per shape\n\n", warmup_iterations);
    printf("%s%-26s %5s %7s %5s %8s %7s %12s %10s %10s %14s %14s%s\n",
           colors.label, "Workload", "Keys", "Hidden", "Heads", "HeadDim",
           "QKVDim", "Intermediate", "ValidKeys", "Dispatches", "Round trip",
           "Burst interval", colors.reset);
    printf("%-26s %5s %7s %5s %8s %7s %12s %10s %10s %14s %14s\n",
           "--------------------------", "-----", "-------", "-----",
           "--------", "-------", "------------", "----------", "----------",
           "--------------", "--------------");

    for (const BenchmarkCase &benchmark_case : cases) {
      Measurements measurements =
          measure_case(device, benchmark_case, iterations, warmup_iterations);
      print_case(benchmark_case, measurements, colors);
    }

    printf("\n%sRound trip%s measures executor.run() plus synchronize() for "
           "one layer per iteration.\n",
           colors.label, colors.reset);
    printf("%sBurst interval%s measures repeated executor.run() submissions "
           "followed by one final synchronize().\n",
           colors.label, colors.reset);
    printf("%sDispatches%s is an estimate of kernel submissions per layer; "
           "multi-head attention currently dispatches per head.\n\n",
           colors.label, colors.reset);
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
