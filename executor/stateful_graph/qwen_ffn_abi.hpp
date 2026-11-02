#ifndef LRRT_STATEFUL_GRAPH_QWEN_FFN_ABI_HPP_
#define LRRT_STATEFUL_GRAPH_QWEN_FFN_ABI_HPP_

#include "executor/stateful_graph/stateful_graph_abi.hpp"

#include <cstdint>
#include <type_traits>

namespace lrrt::stateful_graph::qwen_ffn {

constexpr uint32_t kHiddenSize = 896;
constexpr uint32_t kIntermediateSize = 4864;
constexpr uint32_t kProjectionPartitions = 20;
constexpr uint32_t kDownProjectionPartitions = 10;
constexpr uint32_t kTaskCount = 32;

enum class Opcode : uint32_t {
  rmsnorm_f32 = 1,
  gate_up_silu_f32 = 2,
  down_matvec_f32 = 3,
  residual_advance = 4,
};

// Device pointers and dimensions consumed by the Qwen FFN stage megakernel.
// The scheduler sees this only as the opaque context pointer in ResidentArgs.
struct alignas(16) Context {
  float *hidden;
  const float *norm_weight;
  const float *gate_weight;
  const float *up_weight;
  const float *down_weight;
  float *normalized;
  float *activated;
  float *projected;
  uint32_t hidden_size;
  uint32_t intermediate_size;
  uint32_t reserved[2];
};

static_assert(std::is_standard_layout_v<Context>);
static_assert(sizeof(Context) == 80);
static_assert(kTaskCount <= kMaximumTasks);

} // namespace lrrt::stateful_graph::qwen_ffn

#endif // LRRT_STATEFUL_GRAPH_QWEN_FFN_ABI_HPP_
