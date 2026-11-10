#ifndef LRRT_PERSISTENT_GRAPH_ABI_HPP_
#define LRRT_PERSISTENT_GRAPH_ABI_HPP_

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace lrrt::persistent_graph {

constexpr uint32_t kAbiVersion = 9;
constexpr uint32_t kWorkerThreads = 256;
constexpr uint32_t kMaximumResidentWorkers = 32;
constexpr uint32_t kMaximumTasks = 64;
constexpr uint32_t kMaximumPrograms = 256;
constexpr uint32_t kNoTask = UINT32_MAX;
constexpr uint32_t kProgramIndexBits = 8;
constexpr uint32_t kProgramIndexMask = (1u << kProgramIndexBits) - 1u;

constexpr uint32_t make_phase(uint32_t generation, uint32_t program_index) {
  return (generation << kProgramIndexBits) | program_index;
}

constexpr uint32_t phase_generation(uint32_t phase) {
  return phase >> kProgramIndexBits;
}

constexpr uint32_t phase_program(uint32_t phase) {
  return phase & kProgramIndexMask;
}

enum class Status : uint32_t {
  ready = 0,
  running = 1,
  complete = 2,
  max_steps = 3,
  invalid_program = 4,
};

// Mutable device-resident control state. Workers atomically claim tasks whose
// dependency masks are satisfied. The worker winning a completed phase
// advances the Program and evaluates the loop predicate.
struct alignas(64) State {
  uint32_t abi_version;
  uint32_t status;
  // Generation and active Program are published together with make_phase().
  uint32_t phase;
  uint32_t step;
  uint32_t max_steps;
  uint32_t stop_value;
  uint32_t error_code;
  uint32_t worker_count;
  uint64_t executed_tasks;
  uint32_t predicate_value;
  // Tasks may set this value before a Program with kProgramBranchesOnState
  // completes. A nonzero value selects branch_program.
  uint32_t branch_value;
  uint32_t reserved[4];
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
  uint32_t dependency_mask_high;
  uint32_t reserved;
};

enum ProgramFlags : uint32_t {
  kProgramAdvancesStep = 1u << 0u,
  kProgramEvaluatesPredicate = 1u << 1u,
  kProgramBranchesOnState = 1u << 2u,
};

// A program selects one task DAG from a contiguous task table. The completing
// worker publishes next_program or branch_program in the packed phase word, so
// no host dispatch is needed between execution phases.
struct alignas(16) Program {
  uint32_t task_offset;
  uint32_t task_count;
  uint32_t next_program;
  uint32_t flags;
  uint32_t branch_program;
  uint32_t reserved[3];
};

struct MaskState {
  uint32_t claimed_mask;
  uint32_t completed_mask;
  uint32_t claimed_mask_high;
  uint32_t completed_mask_high;
  uint32_t completed_count;
};

enum class ValidationStatus : uint32_t {
  unvalidated = 0,
  valid = 1,
  invalid = 2,
};

// Persists device validation across repeated launches of an unchanged graph.
// ResidentExecutor invalidates it whenever graph structure, workload context,
// or validation-relevant initial state changes.
struct alignas(8) ValidationState {
  uint32_t status;
  uint32_t reserved;
};

// A compiled stage is a contiguous group of dependency-independent tasks.
// Persistent workers execute the group directly instead of atomically scanning
// and claiming individual tasks.
struct alignas(8) Stage {
  uint32_t task_offset;
  uint32_t task_count;
};

struct alignas(8) CompiledProgram {
  uint32_t stage_offset;
  uint32_t stage_count;
};

struct ResidentArgs {
  State *state;
  const Task *tasks;
  const Program *programs;
  MaskState *phase_states;
  uint32_t program_count;
  uint32_t task_count;
  uint32_t phase_state_count;
  void *context;
  const Stage *stages;
  const CompiledProgram *compiled_programs;
  uint32_t stage_count;
  uint32_t reserved;
  ValidationState *validation_state;
};

struct HostStepArgs {
  State *state;
  const Task *tasks;
  uint32_t task_count;
  uint32_t reserved;
  void *context;
};

static_assert(std::is_standard_layout_v<State>);
static_assert(std::is_standard_layout_v<Task>);
static_assert(std::is_standard_layout_v<Program>);
static_assert(std::is_standard_layout_v<MaskState>);
static_assert(std::is_standard_layout_v<ValidationState>);
static_assert(std::is_standard_layout_v<Stage>);
static_assert(std::is_standard_layout_v<CompiledProgram>);
static_assert(std::is_standard_layout_v<ResidentArgs>);
static_assert(std::is_standard_layout_v<HostStepArgs>);
static_assert(sizeof(State) == 64);
static_assert(sizeof(Task) == 32);
static_assert(sizeof(Program) == 32);
static_assert(sizeof(MaskState) == 20);
static_assert(sizeof(ValidationState) == 8);
static_assert(sizeof(Stage) == 8);
static_assert(sizeof(CompiledProgram) == 8);
static_assert(sizeof(ResidentArgs) == 88);
static_assert(sizeof(HostStepArgs) == 32);

} // namespace lrrt::persistent_graph

#endif // LRRT_PERSISTENT_GRAPH_ABI_HPP_
