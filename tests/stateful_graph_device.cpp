#include "lrrt/lrrt.hpp"
#include "lrrt/persistent_graph.hpp"
#include "tests/stateful_graph_device_context.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef LRRT_STATEFUL_GRAPH_TEST_HSACO
#error "LRRT_STATEFUL_GRAPH_TEST_HSACO must name the test code object"
#endif

namespace {

namespace sg = lrrt::persistent_graph;
namespace fixture = lrrt::persistent_graph::test;

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

template <size_t Size>
bool expect_array(const char *path, const char *name,
                  const std::array<uint32_t, Size> &actual,
                  const std::array<uint32_t, Size> &expected) {
  for (size_t index = 0; index < Size; ++index) {
    if (actual[index] != expected[index]) {
      std::fprintf(stderr, "%s %s[%zu]: got %u, expected %u\n", path, name,
                   index, actual[index], expected[index]);
      return false;
    }
  }
  return true;
}

struct FixtureBuffers {
  explicit FixtureBuffers(lrrt::Device device)
      : values(device, fixture::kValueCount * sizeof(uint32_t)),
        step_checksums(device, fixture::kMaximumSteps * sizeof(uint32_t)),
        branch_steps(device, fixture::kMaximumSteps * sizeof(uint32_t)),
        task_executions(device, fixture::kTaskCount * sizeof(uint32_t)),
        validation_calls(device, sizeof(uint32_t)),
        context(device, sizeof(fixture::Context)) {}

  void reset(bool reset_validation = true) {
    const std::array<uint32_t, fixture::kValueCount> empty_values{};
    const std::array<uint32_t, fixture::kMaximumSteps> empty_checksums{};
    const std::array<uint32_t, fixture::kMaximumSteps> empty_branch_steps{};
    const std::array<uint32_t, fixture::kTaskCount> empty_task_executions{};
    lrrt::copy_to_device(values, empty_values.data(), sizeof(empty_values));
    lrrt::copy_to_device(step_checksums, empty_checksums.data(),
                         sizeof(empty_checksums));
    lrrt::copy_to_device(branch_steps, empty_branch_steps.data(),
                         sizeof(empty_branch_steps));
    lrrt::copy_to_device(task_executions, empty_task_executions.data(),
                         sizeof(empty_task_executions));
    if (reset_validation) {
      const uint32_t empty_validation_calls = 0;
      lrrt::copy_to_device(validation_calls, &empty_validation_calls,
                           sizeof(empty_validation_calls));
    }

    const fixture::Context host_context{
        fixture::kContextMagic,
        fixture::kValueCount,
        static_cast<uint32_t *>(values.data()),
        static_cast<uint32_t *>(step_checksums.data()),
        static_cast<uint32_t *>(branch_steps.data()),
        static_cast<uint32_t *>(task_executions.data()),
        static_cast<uint32_t *>(validation_calls.data()),
    };
    lrrt::copy_to_device(context, &host_context, sizeof(host_context));
  }

  lrrt::DeviceBuffer values;
  lrrt::DeviceBuffer step_checksums;
  lrrt::DeviceBuffer branch_steps;
  lrrt::DeviceBuffer task_executions;
  lrrt::DeviceBuffer validation_calls;
  lrrt::DeviceBuffer context;
};

bool expect_execution(const char *label, const sg::ResidentExecutor &executor,
                      FixtureBuffers *buffers) {
  const sg::State state = executor.read_state();
  if (state.status != static_cast<uint32_t>(sg::Status::complete) ||
      state.step != fixture::kStepCount || state.predicate_value != 2 ||
      state.branch_value != 1 || state.executed_tasks != 18 ||
      state.error_code != 0 || state.phase != sg::make_phase(9, 2)) {
    std::fprintf(stderr,
                 "%s state mismatch: status=%u step=%u predicate=%u "
                 "branch=%u tasks=%llu error=%u phase=%u\n",
                 label, state.status, state.step, state.predicate_value,
                 state.branch_value,
                 static_cast<unsigned long long>(state.executed_tasks),
                 state.error_code, state.phase);
    return false;
  }

  std::array<uint32_t, fixture::kValueCount> values{};
  std::array<uint32_t, fixture::kMaximumSteps> checksums{};
  std::array<uint32_t, fixture::kMaximumSteps> branch_steps{};
  std::array<uint32_t, fixture::kTaskCount> task_executions{};
  lrrt::copy_to_host(values.data(), buffers->values, sizeof(values));
  lrrt::copy_to_host(checksums.data(), buffers->step_checksums,
                     sizeof(checksums));
  lrrt::copy_to_host(branch_steps.data(), buffers->branch_steps,
                     sizeof(branch_steps));
  lrrt::copy_to_host(task_executions.data(), buffers->task_executions,
                     sizeof(task_executions));

  return expect_array(label, "values", values,
                      {10, 10, 10, 10, 14, 14, 14, 14}) &&
         expect_array(label, "checksums", checksums,
                      {12, 32, 60, 96, 0, 0, 0, 0}) &&
         expect_array(label, "branch_steps", branch_steps,
                      {2, 4, 0, 0, 0, 0, 0, 0}) &&
         expect_array(label, "task_executions", task_executions,
                      {4, 4, 4, 4, 2});
}

bool run_path(const char *label, lrrt::Device device,
              const lrrt::Kernel &kernel, const std::vector<sg::Task> &tasks,
              const std::vector<sg::Program> &programs,
              const sg::ResidentConfig &config, const sg::CompiledPlan *plan,
              FixtureBuffers *buffers) {
  buffers->reset();
  sg::ResidentExecutor executor =
      plan == nullptr
          ? sg::ResidentExecutor(device, kernel, tasks, programs, config)
          : sg::ResidentExecutor(device, kernel, tasks, programs, config,
                                 *plan);
  executor.bind_context(buffers->context.data());
  for (uint32_t run = 0; run < 3; ++run) {
    if (run != 0) {
      buffers->reset(false);
    }
    if (run == 2) {
      // Rebinding even the same address explicitly invalidates cached
      // workload validation for callers that changed context contents.
      executor.bind_context(buffers->context.data());
    }
    executor.initialize(0, fixture::kMaximumSteps, 2);
    executor.launch();
    device.synchronize();
    if (!expect_execution(label, executor, buffers)) {
      return false;
    }

    uint32_t validation_calls = 0;
    lrrt::copy_to_host(&validation_calls, buffers->validation_calls,
                       sizeof(validation_calls));
    const uint32_t expected_calls = run == 2 ? 2u : 1u;
    if (validation_calls != expected_calls) {
      std::fprintf(stderr, "%s validation calls: got %u, expected %u\n", label,
                   validation_calls, expected_calls);
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      std::printf("stateful graph device: skipped, no GPU devices\n");
      return 0;
    }
    const lrrt::Device device = runtime.open_device(0);
    const std::vector<unsigned char> image =
        read_file(LRRT_STATEFUL_GRAPH_TEST_HSACO);
    const lrrt::Module module(device, image);
    const lrrt::Kernel dynamic_kernel =
        module.kernel("stateful_graph_dynamic_test_kernel");
    const lrrt::Kernel compiled_kernel =
        module.kernel("stateful_graph_compiled_test_kernel");

    const std::vector<sg::Task> tasks = {
        {static_cast<uint32_t>(fixture::Opcode::add_range), 0, 0, 4, 1, 0, 0,
         0},
        {static_cast<uint32_t>(fixture::Opcode::add_range), 0, 4, 8, 2, 1, 0,
         0},
        {static_cast<uint32_t>(fixture::Opcode::checksum), 3, 0, 8, 0, 2, 0, 0},
        {static_cast<uint32_t>(fixture::Opcode::select_branch), 0, 0, 1, 2, 3,
         0, 0},
        {static_cast<uint32_t>(fixture::Opcode::advance_predicate), 0, 0, 1, 1,
         4, 0, 0},
    };
    const std::vector<sg::Program> programs = {
        {0, 3, 1, 0, 0, {0, 0, 0}},
        {3,
         1,
         0,
         sg::kProgramAdvancesStep | sg::kProgramBranchesOnState,
         2,
         {0, 0, 0}},
        {4, 1, 0, sg::kProgramEvaluatesPredicate, 0, {0, 0, 0}},
    };
    const sg::ResidentAdmission admission =
        sg::admit_resident_workers(device, compiled_kernel, 4);
    const sg::ResidentConfig config{16, admission.worker_count};
    const sg::CompiledPlan plan =
        sg::compile_resident_plan(tasks, programs, config);
    FixtureBuffers buffers(device);

    if (!run_path("dynamic", device, dynamic_kernel, tasks, programs, config,
                  nullptr, &buffers) ||
        !run_path("compiled", device, compiled_kernel, tasks, programs, config,
                  &plan, &buffers)) {
      return 1;
    }
    std::printf("stateful graph device: ok (%u workers)\n",
                admission.worker_count);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "stateful graph device: %s\n", error.what());
    return 1;
  }
}
