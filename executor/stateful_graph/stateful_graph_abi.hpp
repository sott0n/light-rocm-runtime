#ifndef LRRT_STATEFUL_GRAPH_ABI_HPP_
#define LRRT_STATEFUL_GRAPH_ABI_HPP_

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace lrrt::stateful_graph {

constexpr uint32_t kAbiVersion = 3;
constexpr uint32_t kWorkerThreads = 64;
constexpr uint32_t kMaximumTasks = 32;
constexpr uint32_t kNoTask = UINT32_MAX;

enum class Status : uint32_t {
  ready = 0,
  running = 1,
  complete = 2,
  max_steps = 3,
  invalid_program = 4,
};

enum class Opcode : uint32_t {
  rmsnorm_f32 = 1,
  silu_f32 = 2,
  residual_add_f32 = 3,
  advance_state = 4,
};

// Mutable device-resident control state. Workers atomically claim tasks whose
// dependency masks are satisfied. The worker completing the final task resets
// the masks, advances the epoch, and evaluates the loop predicate.
struct alignas(64) State {
  uint32_t abi_version;
  uint32_t status;
  uint32_t epoch;
  uint32_t claimed_mask;
  uint32_t completed_mask;
  uint32_t step;
  uint32_t max_steps;
  uint32_t stop_value;
  uint32_t task_count;
  uint32_t error_code;
  uint64_t executed_tasks;
  uint32_t predicate_value;
  uint32_t reserved[3];
};

// A deliberately small task descriptor. Opcodes interpret the range and two
// arguments as stage-specific parameters; dependency_mask remains common to
// every stage so scheduling is independent of model semantics.
struct alignas(32) Task {
  uint32_t opcode;
  uint32_t dependency_mask;
  uint32_t begin;
  uint32_t end;
  uint32_t argument0;
  uint32_t argument1;
  uint32_t reserved[2];
};

struct ResidentArgs {
  State *state;
  const Task *tasks;
  float *values;
  float *workspace;
};

struct HostStepArgs {
  State *state;
  const Task *tasks;
  uint32_t task_count;
  uint32_t reserved;
  float *values;
  float *workspace;
};

static_assert(std::is_standard_layout_v<State>);
static_assert(std::is_standard_layout_v<Task>);
static_assert(std::is_standard_layout_v<ResidentArgs>);
static_assert(std::is_standard_layout_v<HostStepArgs>);
static_assert(sizeof(State) == 64);
static_assert(sizeof(Task) == 32);
static_assert(sizeof(ResidentArgs) == 32);
static_assert(sizeof(HostStepArgs) == 40);

} // namespace lrrt::stateful_graph

#endif // LRRT_STATEFUL_GRAPH_ABI_HPP_
