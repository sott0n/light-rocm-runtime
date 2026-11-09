#include "lrrt/persistent_graph.hpp"
#include "benchmarks/persistent_graph_context.hpp"
#include "lrrt/lrrt.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef LRRT_PERSISTENT_GRAPH_BENCHMARK_HSACO
#error                                                                         \
    "LRRT_PERSISTENT_GRAPH_BENCHMARK_HSACO must name the benchmark code object"
#endif

namespace {

namespace sg = lrrt::persistent_graph;
namespace fixture = lrrt::benchmarks::persistent_graph;
using Clock = std::chrono::steady_clock;

constexpr uint32_t kWarmupTrials = 3;
constexpr uint64_t kConditioningCycles = 100000000;

struct Options {
  uint32_t steps = 256;
  uint32_t trials = 11;
  uint32_t maximum_workers = sg::kMaximumResidentWorkers;
};

struct GraphShape {
  uint32_t workers;
  uint32_t tasks;
  uint32_t stages;
  uint32_t programs;
};

struct GraphDefinition {
  std::vector<sg::Task> tasks;
  std::vector<sg::Program> programs;
};

struct Timing {
  double kernel_ns;
  double initialize_ns;
};

struct PathTimings {
  Timing one_step;
  Timing long_run;
};

struct InterleavedTimings {
  PathTimings dynamic;
  PathTimings compiled;
};

struct ClockConditioningArgs {
  uint64_t cycles;
  uint32_t *output;
};

struct ClockConditioner {
  const lrrt::Kernel &kernel;
  lrrt::DeviceBuffer &output;
  uint32_t workers;

  void run(lrrt::Queue &queue) const {
    const lr_launch_config_t config = {
        {workers * sg::kWorkerThreads, 1, 1},
        {sg::kWorkerThreads, 1, 1},
        0,
    };
    const ClockConditioningArgs args{
        kConditioningCycles,
        static_cast<uint32_t *>(output.data()),
    };
    lrrt::launch(queue, kernel, config, args);
    queue.synchronize();
  }
};

struct PathResult {
  std::string path;
  GraphShape shape;
  double setup_us;
  double one_step_us;
  double long_run_us;
  double startup_estimate_us;
  double steady_phase_ns;
  double steady_stage_ns;
  double task_rate_millions;
  double initialize_us;
};

struct ShapeResult {
  PathResult dynamic;
  PathResult compiled;
  double plan_compile_us;
};

uint32_t parse_u32(const char *text, const char *name) {
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (text == end || end == nullptr || *end != '\0' || value == 0 ||
      value > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(std::string(name) +
                                " must be a positive uint32");
  }
  return static_cast<uint32_t>(value);
}

Options parse_options(int argc, char **argv) {
  if (argc > 4) {
    throw std::invalid_argument(
        "usage: lrrt_persistent_graph_benchmark [steps] [trials] "
        "[max-workers]");
  }
  Options options;
  if (argc >= 2) {
    options.steps = parse_u32(argv[1], "steps");
  }
  if (argc >= 3) {
    options.trials = parse_u32(argv[2], "trials");
  }
  if (argc >= 4) {
    options.maximum_workers = parse_u32(argv[3], "max-workers");
  }
  if (options.steps < 2) {
    throw std::invalid_argument("steps must be at least 2");
  }
  return options;
}

std::vector<unsigned char> read_file(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error(std::string("failed to open ") + path);
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error(std::string("empty code object: ") + path);
  }
  std::vector<unsigned char> data(static_cast<size_t>(size));
  file.seekg(0, std::ios::beg);
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error(std::string("failed to read ") + path);
  }
  return data;
}

double elapsed_ns(Clock::time_point begin, Clock::time_point end) {
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count());
}

double median(std::vector<double> samples) {
  if (samples.empty()) {
    throw std::logic_error("cannot take the median of an empty sample set");
  }
  const size_t middle = samples.size() / 2;
  std::nth_element(samples.begin(), samples.begin() + middle, samples.end());
  if ((samples.size() & 1u) != 0) {
    return samples[middle];
  }
  const double upper = samples[middle];
  const double lower =
      *std::max_element(samples.begin(), samples.begin() + middle);
  return (lower + upper) * 0.5;
}

uint64_t phase_count(const GraphShape &shape, uint32_t steps) {
  return static_cast<uint64_t>(shape.programs) * steps;
}

GraphDefinition make_graph(const GraphShape &shape) {
  if (shape.tasks == 0 || shape.tasks > sg::kMaximumTasks ||
      shape.stages == 0 || shape.stages > shape.tasks || shape.programs == 0 ||
      shape.programs > sg::kMaximumPrograms) {
    throw std::invalid_argument("invalid persistent graph benchmark shape");
  }

  GraphDefinition graph;
  graph.tasks.reserve(static_cast<size_t>(shape.tasks) * shape.programs);
  graph.programs.reserve(shape.programs);
  for (uint32_t program_index = 0; program_index < shape.programs;
       ++program_index) {
    const uint32_t task_offset = static_cast<uint32_t>(graph.tasks.size());
    uint64_t completed_mask = 0;
    for (uint32_t stage_index = 0; stage_index < shape.stages; ++stage_index) {
      const uint32_t stage_begin = static_cast<uint32_t>(
          (static_cast<uint64_t>(stage_index) * shape.tasks) / shape.stages);
      const uint32_t stage_end = static_cast<uint32_t>(
          (static_cast<uint64_t>(stage_index + 1u) * shape.tasks) /
          shape.stages);
      for (uint32_t task_index = stage_begin; task_index < stage_end;
           ++task_index) {
        graph.tasks.push_back({
            fixture::kIncrementOpcode,
            static_cast<uint32_t>(completed_mask),
            0,
            1,
            task_offset + task_index,
            1,
            static_cast<uint32_t>(completed_mask >> 32u),
            0,
        });
      }
      for (uint32_t task_index = stage_begin; task_index < stage_end;
           ++task_index) {
        completed_mask |= static_cast<uint64_t>(1) << task_index;
      }
    }

    const bool final_program = program_index + 1u == shape.programs;
    graph.programs.push_back({
        task_offset,
        shape.tasks,
        final_program ? 0u : program_index + 1u,
        final_program ? sg::kProgramAdvancesStep : 0u,
        0,
        {0, 0, 0},
    });
  }
  return graph;
}

void validate_state(const char *path, const sg::State &state,
                    const GraphShape &shape, uint32_t steps) {
  const uint64_t phases = phase_count(shape, steps);
  const uint64_t expected_tasks = phases * shape.tasks;
  const uint32_t expected_phase =
      sg::make_phase(static_cast<uint32_t>(phases - 1u), shape.programs - 1u);
  if (state.status != static_cast<uint32_t>(sg::Status::max_steps) ||
      state.step != steps || state.phase != expected_phase ||
      state.executed_tasks != expected_tasks || state.error_code != 0) {
    throw std::runtime_error(std::string(path) + " result mismatch: status=" +
                             std::to_string(state.status) +
                             " step=" + std::to_string(state.step) +
                             " phase=" + std::to_string(state.phase) +
                             " tasks=" + std::to_string(state.executed_tasks) +
                             " error=" + std::to_string(state.error_code));
  }
}

void warm_up_path(lrrt::Queue &queue, sg::ResidentExecutor &executor,
                  const char *path, const GraphShape &shape, uint32_t steps) {
  executor.initialize(0, steps, std::numeric_limits<uint32_t>::max());
  executor.launch(queue);
  queue.synchronize();
  validate_state(path, executor.read_state(), shape, steps);
}

Timing sample_path(lrrt::Queue &queue, lrrt::Event &start, lrrt::Event &end,
                   sg::ResidentExecutor &executor, const char *path,
                   const GraphShape &shape, uint32_t steps) {
  const auto initialize_begin = Clock::now();
  executor.initialize(0, steps, std::numeric_limits<uint32_t>::max());
  const auto initialize_end = Clock::now();
  start.record(queue);
  executor.launch(queue);
  end.record(queue);
  end.synchronize();
  start.synchronize();
  const double kernel_ns =
      static_cast<double>(lrrt::elapsed_time_ns(start, end));
  validate_state(path, executor.read_state(), shape, steps);
  return {kernel_ns, elapsed_ns(initialize_begin, initialize_end)};
}

InterleavedTimings measure_interleaved(lrrt::Device device, lrrt::Queue &queue,
                                       sg::ResidentExecutor &dynamic_executor,
                                       sg::ResidentExecutor &compiled_executor,
                                       const GraphShape &shape,
                                       const Options &options,
                                       const ClockConditioner &conditioner) {
  for (uint32_t trial = 0; trial < kWarmupTrials; ++trial) {
    conditioner.run(queue);
    if ((trial & 1u) == 0) {
      warm_up_path(queue, dynamic_executor, "dynamic", shape, options.steps);
      warm_up_path(queue, compiled_executor, "compiled", shape, options.steps);
    } else {
      warm_up_path(queue, compiled_executor, "compiled", shape, options.steps);
      warm_up_path(queue, dynamic_executor, "dynamic", shape, options.steps);
    }
  }

  std::vector<double> dynamic_short_kernel;
  std::vector<double> dynamic_short_initialize;
  std::vector<double> dynamic_long_kernel;
  std::vector<double> dynamic_long_initialize;
  std::vector<double> compiled_short_kernel;
  std::vector<double> compiled_short_initialize;
  std::vector<double> compiled_long_kernel;
  std::vector<double> compiled_long_initialize;
  for (std::vector<double> *samples :
       {&dynamic_short_kernel, &dynamic_short_initialize, &dynamic_long_kernel,
        &dynamic_long_initialize, &compiled_short_kernel,
        &compiled_short_initialize, &compiled_long_kernel,
        &compiled_long_initialize}) {
    samples->reserve(options.trials);
  }

  lrrt::Event start(device);
  lrrt::Event end(device);
  auto collect = [&](sg::ResidentExecutor &executor, const char *path,
                     uint32_t steps, std::vector<double> &kernel_samples,
                     std::vector<double> &initialize_samples) {
    const Timing timing =
        sample_path(queue, start, end, executor, path, shape, steps);
    kernel_samples.push_back(timing.kernel_ns);
    initialize_samples.push_back(timing.initialize_ns);
  };
  for (uint32_t trial = 0; trial < options.trials; ++trial) {
    conditioner.run(queue);
    if ((trial & 1u) == 0) {
      collect(dynamic_executor, "dynamic", 1, dynamic_short_kernel,
              dynamic_short_initialize);
      collect(compiled_executor, "compiled", 1, compiled_short_kernel,
              compiled_short_initialize);
      collect(compiled_executor, "compiled", options.steps,
              compiled_long_kernel, compiled_long_initialize);
      collect(dynamic_executor, "dynamic", options.steps, dynamic_long_kernel,
              dynamic_long_initialize);
    } else {
      collect(dynamic_executor, "dynamic", options.steps, dynamic_long_kernel,
              dynamic_long_initialize);
      collect(compiled_executor, "compiled", options.steps,
              compiled_long_kernel, compiled_long_initialize);
      collect(compiled_executor, "compiled", 1, compiled_short_kernel,
              compiled_short_initialize);
      collect(dynamic_executor, "dynamic", 1, dynamic_short_kernel,
              dynamic_short_initialize);
    }
  }
  return {
      {{median(std::move(dynamic_short_kernel)),
        median(std::move(dynamic_short_initialize))},
       {median(std::move(dynamic_long_kernel)),
        median(std::move(dynamic_long_initialize))}},
      {{median(std::move(compiled_short_kernel)),
        median(std::move(compiled_short_initialize))},
       {median(std::move(compiled_long_kernel)),
        median(std::move(compiled_long_initialize))}},
  };
}

PathResult make_path_result(const char *path, const GraphShape &shape,
                            const Options &options, double setup_us,
                            const PathTimings &timings) {
  const Timing &one_step = timings.one_step;
  const Timing &long_run = timings.long_run;
  const double additional_phases = static_cast<double>(
      phase_count(shape, options.steps) - phase_count(shape, 1));
  const double steady_phase_ns =
      (long_run.kernel_ns - one_step.kernel_ns) / additional_phases;
  if (steady_phase_ns <= 0) {
    throw std::runtime_error(std::string(path) +
                             " steady-state timing is not positive");
  }
  const double startup_estimate_ns =
      one_step.kernel_ns - steady_phase_ns * shape.programs;

  return {
      path,
      shape,
      setup_us,
      one_step.kernel_ns / 1.0e3,
      long_run.kernel_ns / 1.0e3,
      startup_estimate_ns / 1.0e3,
      steady_phase_ns,
      steady_phase_ns / shape.stages,
      static_cast<double>(shape.tasks) * 1.0e3 / steady_phase_ns,
      long_run.initialize_ns / 1.0e3,
  };
}

ShapeResult run_shape(lrrt::Device device, lrrt::Queue &queue,
                      const lrrt::Kernel &dynamic_kernel,
                      const lrrt::Kernel &compiled_kernel,
                      const GraphShape &shape, const Options &options,
                      const ClockConditioner &conditioner) {
  const uint64_t long_phases = phase_count(shape, options.steps);
  if (long_phases - 1u >
      sg::phase_generation(std::numeric_limits<uint32_t>::max())) {
    throw std::invalid_argument("steps and programs exceed phase encoding");
  }
  GraphDefinition graph = make_graph(shape);
  const sg::ResidentConfig config{static_cast<uint32_t>(long_phases),
                                  shape.workers};
  const auto plan_begin = Clock::now();
  const sg::CompiledPlan plan =
      sg::compile_resident_plan(graph.tasks, graph.programs, config);
  const auto plan_end = Clock::now();

  lrrt::DeviceBuffer counters(device, graph.tasks.size() * sizeof(uint32_t));
  const std::vector<uint32_t> empty_counters(graph.tasks.size());
  lrrt::copy_to_device(counters, empty_counters);
  const fixture::Context context{
      fixture::kContextMagic,
      static_cast<uint32_t>(graph.tasks.size()),
      static_cast<uint32_t *>(counters.data()),
  };
  lrrt::DeviceBuffer context_buffer(device, sizeof(context));
  lrrt::copy_to_device(context_buffer, &context, sizeof(context));

  const auto dynamic_setup_begin = Clock::now();
  sg::ResidentExecutor dynamic_executor(device, dynamic_kernel, graph.tasks,
                                        graph.programs, config);
  dynamic_executor.bind_context(context_buffer.data());
  const auto dynamic_setup_end = Clock::now();
  const auto compiled_setup_begin = Clock::now();
  sg::ResidentExecutor compiled_executor(device, compiled_kernel, graph.tasks,
                                         graph.programs, config, plan);
  compiled_executor.bind_context(context_buffer.data());
  const auto compiled_setup_end = Clock::now();

  const InterleavedTimings timings =
      measure_interleaved(device, queue, dynamic_executor, compiled_executor,
                          shape, options, conditioner);
  PathResult dynamic = make_path_result(
      "dynamic", shape, options,
      elapsed_ns(dynamic_setup_begin, dynamic_setup_end) / 1.0e3,
      timings.dynamic);
  PathResult compiled = make_path_result(
      "compiled", shape, options,
      elapsed_ns(compiled_setup_begin, compiled_setup_end) / 1.0e3,
      timings.compiled);
  return {std::move(dynamic), std::move(compiled),
          elapsed_ns(plan_begin, plan_end) / 1.0e3};
}

void print_header() {
  std::printf("%-8s %3s %3s %3s %3s %10s %10s %10s %10s %10s %10s\n", "Path",
              "W", "T", "S", "P", "1-step us", "long us", "start us",
              "ns/phase", "ns/stage", "Mtask/s");
  std::printf("%-8s %3s %3s %3s %3s %10s %10s %10s %10s %10s %10s\n",
              "--------", "---", "---", "---", "---", "----------",
              "----------", "----------", "----------", "----------",
              "----------");
}

void print_path(const PathResult &result) {
  std::printf(
      "%-8s %3u %3u %3u %3u %10.3f %10.3f %10.3f %10.2f %10.2f %10.2f\n",
      result.path.c_str(), result.shape.workers, result.shape.tasks,
      result.shape.stages, result.shape.programs, result.one_step_us,
      result.long_run_us, result.startup_estimate_us, result.steady_phase_ns,
      result.steady_stage_ns, result.task_rate_millions);
}

double stage_slope(const std::vector<PathResult> &results) {
  double sum_x = 0;
  double sum_y = 0;
  double sum_xx = 0;
  double sum_xy = 0;
  for (const PathResult &result : results) {
    const double x = result.shape.stages;
    const double y = result.steady_phase_ns;
    sum_x += x;
    sum_y += y;
    sum_xx += x * x;
    sum_xy += x * y;
  }
  const double count = static_cast<double>(results.size());
  const double denominator = count * sum_xx - sum_x * sum_x;
  return denominator == 0 ? 0 : (count * sum_xy - sum_x * sum_y) / denominator;
}

std::vector<uint32_t> worker_sweep(uint32_t maximum_workers) {
  std::vector<uint32_t> workers;
  for (uint32_t value = 1; value <= maximum_workers && value != 0; value *= 2) {
    workers.push_back(value);
  }
  if (workers.empty() || workers.back() != maximum_workers) {
    workers.push_back(maximum_workers);
  }
  return workers;
}

template <typename ShapeFactory>
std::vector<ShapeResult>
run_sweep(const char *name, const std::vector<uint32_t> &values,
          ShapeFactory make_shape, lrrt::Device device, lrrt::Queue &queue,
          const lrrt::Kernel &dynamic_kernel,
          const lrrt::Kernel &compiled_kernel, const Options &options,
          const ClockConditioner &conditioner) {
  std::printf("\n%s sweep\n", name);
  print_header();
  std::vector<ShapeResult> results;
  results.reserve(values.size());
  for (uint32_t value : values) {
    ShapeResult result =
        run_shape(device, queue, dynamic_kernel, compiled_kernel,
                  make_shape(value), options, conditioner);
    print_path(result.dynamic);
    print_path(result.compiled);
    results.push_back(std::move(result));
  }
  return results;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      std::fprintf(stderr, "persistent_graph_benchmark: no GPU devices\n");
      return 1;
    }
    const lrrt::Device device = runtime.open_device(0);
    const std::vector<unsigned char> image =
        read_file(LRRT_PERSISTENT_GRAPH_BENCHMARK_HSACO);
    const lrrt::Module module(device, image);
    const lrrt::Kernel dynamic_kernel =
        module.kernel("persistent_graph_dynamic_benchmark_kernel");
    const lrrt::Kernel compiled_kernel =
        module.kernel("persistent_graph_compiled_benchmark_kernel");
    const lrrt::Kernel conditioning_kernel =
        module.kernel("persistent_graph_clock_conditioning_kernel");
    const uint32_t safe_workers =
        std::min(sg::resident_worker_limit(device.properties(),
                                           dynamic_kernel.properties()),
                 sg::resident_worker_limit(device.properties(),
                                           compiled_kernel.properties()));
    const uint32_t maximum_workers =
        std::min(options.maximum_workers, safe_workers);
    const uint32_t baseline_workers = std::min(8u, maximum_workers);
    const uint32_t conditioning_workers =
        device.properties().compute_unit_count;
    lrrt::Queue queue(device);
    lrrt::DeviceBuffer conditioning_output(
        device, conditioning_workers * sg::kWorkerThreads * sizeof(uint32_t));
    const ClockConditioner conditioner{
        conditioning_kernel,
        conditioning_output,
        conditioning_workers,
    };

    std::printf("Persistent Graph Runtime Benchmark\n");
    std::printf("==================================\n");
    std::printf("Device:              %s\n", device.name().c_str());
    std::printf("Steady-state steps:  %u\n", options.steps);
    std::printf("Trials (median):     %u\n", options.trials);
    std::printf("Warm-up trials:      %u\n", kWarmupTrials);
    std::printf("Resident worker cap: %u\n", safe_workers);
    std::printf("Task payload:        one 32-bit counter update\n");

    std::printf("Clock conditioning:  %u CU-sized workers before every trial\n",
                conditioning_workers);

    const std::vector<ShapeResult> workers = run_sweep(
        "Workers", worker_sweep(maximum_workers),
        [](uint32_t value) { return GraphShape{value, 32, 4, 4}; }, device,
        queue, dynamic_kernel, compiled_kernel, options, conditioner);
    (void)workers;

    const std::vector<uint32_t> task_values = {1, 8, 32, 64};
    const std::vector<ShapeResult> tasks = run_sweep(
        "Tasks per program", task_values,
        [baseline_workers](uint32_t value) {
          return GraphShape{baseline_workers, value, std::min(4u, value), 4};
        },
        device, queue, dynamic_kernel, compiled_kernel, options, conditioner);
    (void)tasks;

    const std::vector<uint32_t> stage_values = {1, 2, 4, 8, 16, 32};
    const std::vector<ShapeResult> stages = run_sweep(
        "Stages per program", stage_values,
        [baseline_workers](uint32_t value) {
          return GraphShape{baseline_workers, 32, value, 4};
        },
        device, queue, dynamic_kernel, compiled_kernel, options, conditioner);

    const uint64_t program_phase_budget =
        static_cast<uint64_t>(options.steps) * 4u;
    const std::vector<uint32_t> program_values = {1, 2, 4, 8};
    std::printf("\nPrograms per step sweep (%llu total phases)\n",
                static_cast<unsigned long long>(program_phase_budget));
    print_header();
    for (uint32_t programs : program_values) {
      if (static_cast<uint64_t>(programs) * 2u > program_phase_budget) {
        continue;
      }
      Options program_options = options;
      program_options.steps =
          static_cast<uint32_t>(program_phase_budget / programs);
      ShapeResult result = run_shape(
          device, queue, dynamic_kernel, compiled_kernel,
          {baseline_workers, 32, 4, programs}, program_options, conditioner);
      print_path(result.dynamic);
      print_path(result.compiled);
    }

    std::vector<PathResult> dynamic_stages;
    std::vector<PathResult> compiled_stages;
    for (const ShapeResult &result : stages) {
      dynamic_stages.push_back(result.dynamic);
      compiled_stages.push_back(result.compiled);
    }
    const ShapeResult &baseline = stages[2];
    std::printf("\nDerived control costs\n");
    std::printf("---------------------\n");
    std::printf("Dynamic additional dependency stage: %.2f ns/phase-stage\n",
                stage_slope(dynamic_stages));
    std::printf("Compiled additional stage barrier:    %.2f ns/barrier\n",
                stage_slope(compiled_stages));
    std::printf("Baseline host plan compilation:       %.3f us\n",
                baseline.plan_compile_us);
    std::printf("Baseline dynamic executor setup:      %.3f us\n",
                baseline.dynamic.setup_us);
    std::printf("Baseline compiled executor setup:     %.3f us\n",
                baseline.compiled.setup_us);
    std::printf("Baseline per-run state initialize:    %.3f us dynamic, "
                "%.3f us compiled\n",
                baseline.dynamic.initialize_us,
                baseline.compiled.initialize_us);
    std::printf("\nstart us is the extrapolated device validation/launch "
                "intercept; "
                "ns/phase and Mtask/s are marginal steady-state values.\n");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "persistent_graph_benchmark: %s\n", error.what());
    return 1;
  }
}
