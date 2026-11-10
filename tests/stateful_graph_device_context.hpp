#ifndef LRRT_TESTS_STATEFUL_GRAPH_DEVICE_CONTEXT_HPP_
#define LRRT_TESTS_STATEFUL_GRAPH_DEVICE_CONTEXT_HPP_

#include <cstdint>
#include <type_traits>

namespace lrrt::persistent_graph::test {

constexpr uint32_t kContextMagic = 0x53504754u;
constexpr uint32_t kValueCount = 8;
constexpr uint32_t kStepCount = 4;
constexpr uint32_t kMaximumSteps = 8;
constexpr uint32_t kTaskCount = 5;

enum class Opcode : uint32_t {
  add_range = 0,
  checksum = 1,
  select_branch = 2,
  advance_predicate = 3,
};

struct Context {
  uint32_t magic;
  uint32_t value_count;
  uint32_t *values;
  uint32_t *step_checksums;
  uint32_t *branch_steps;
  uint32_t *task_executions;
  uint32_t *validation_calls;
};

static_assert(std::is_standard_layout_v<Context>);

} // namespace lrrt::persistent_graph::test

#endif // LRRT_TESTS_STATEFUL_GRAPH_DEVICE_CONTEXT_HPP_
