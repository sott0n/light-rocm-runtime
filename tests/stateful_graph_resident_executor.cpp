#include "lrrt/persistent_graph.hpp"

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

namespace sg = lrrt::persistent_graph;

bool expect_invalid(const char *label, const std::vector<sg::Task> &tasks,
                    const std::vector<sg::Program> &programs,
                    sg::ResidentConfig config) {
  try {
    sg::validate_resident_graph(tasks, programs, config);
  } catch (const std::invalid_argument &) {
    return true;
  }
  std::fprintf(stderr, "%s: expected invalid_argument\n", label);
  return false;
}

bool expect_compile_invalid(const char *label,
                            const std::vector<sg::Task> &tasks,
                            const std::vector<sg::Program> &programs,
                            sg::ResidentConfig config) {
  try {
    (void)sg::compile_resident_plan(tasks, programs, config);
  } catch (const std::invalid_argument &) {
    return true;
  }
  std::fprintf(stderr, "%s: expected compile failure\n", label);
  return false;
}

} // namespace

int main() {
  const lrrt::DeviceProperties device_properties = {60, 32, 32, 1024,
                                                    64u * 1024u};
  const lrrt::KernelProperties kernel_properties = {128, 1060, 0};
  const sg::ResidentAdmission admission =
      sg::admit_resident_workers(device_properties, kernel_properties, 128);
  if (admission.requested_worker_count != 128 ||
      admission.safe_worker_limit != sg::kMaximumResidentWorkers ||
      admission.worker_count != sg::kMaximumResidentWorkers) {
    std::fprintf(stderr, "unexpected resident admission result\n");
    return 1;
  }
  try {
    (void)sg::resident_worker_limit(device_properties,
                                    {128, 64u * 1024u + 1u, 0});
    std::fprintf(stderr, "oversized resident LDS was admitted\n");
    return 1;
  } catch (const std::invalid_argument &) {
  }

  const std::vector<sg::Task> tasks = {
      {0, 0, 0, 1, 0, 0, 0, 0},
      {1, 1, 0, 1, 0, 0, 0, 0},
  };
  const std::vector<sg::Program> programs = {
      {0, 2, 0, sg::kProgramAdvancesStep, 0, {0, 0, 0}},
  };
  const sg::ResidentConfig config{8, 4};
  sg::validate_resident_graph(tasks, programs, config);
  const sg::CompiledPlan plan =
      sg::compile_resident_plan(tasks, programs, config);
  if (plan.programs.size() != 1 || plan.stages.size() != 2 ||
      plan.programs[0].stage_offset != 0 || plan.programs[0].stage_count != 2 ||
      plan.stages[0].task_offset != 0 || plan.stages[0].task_count != 1 ||
      plan.stages[1].task_offset != 1 || plan.stages[1].task_count != 1) {
    std::fprintf(stderr, "compiled plan has incorrect stages\n");
    return 1;
  }

  std::vector<sg::Program> invalid_programs = programs;
  invalid_programs[0].task_count = sg::kMaximumTasks + 1;
  std::vector<sg::Task> invalid_tasks = tasks;
  invalid_tasks[1].dependency_mask = 4;
  std::vector<sg::Task> cyclic_tasks = tasks;
  cyclic_tasks[0].dependency_mask = 2;
  if (!expect_invalid("empty tasks", {}, programs, config) ||
      !expect_invalid("zero workers", tasks, programs, {8, 0}) ||
      !expect_invalid("too many workers", tasks, programs,
                      {8, sg::kMaximumResidentWorkers + 1}) ||
      !expect_invalid("zero phase states", tasks, programs, {0, 4}) ||
      !expect_invalid("task range", tasks, invalid_programs, config) ||
      !expect_invalid("dependency mask", invalid_tasks, programs, config) ||
      !expect_compile_invalid("cyclic graph", cyclic_tasks, programs, config)) {
    return 1;
  }
  return 0;
}
