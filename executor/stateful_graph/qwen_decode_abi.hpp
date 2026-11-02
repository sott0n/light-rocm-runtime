#ifndef LRRT_STATEFUL_GRAPH_QWEN_DECODE_ABI_HPP_
#define LRRT_STATEFUL_GRAPH_QWEN_DECODE_ABI_HPP_

#include "executor/stateful_graph/stateful_graph_abi.hpp"

#include <cstdint>
#include <type_traits>

namespace lrrt::stateful_graph::qwen_decode {

constexpr uint32_t kHiddenSize = 896;
constexpr uint32_t kAttentionHeads = 14;
constexpr uint32_t kKvHeads = 2;
constexpr uint32_t kHeadSize = 64;
constexpr uint32_t kQuerySize = kAttentionHeads * kHeadSize;
constexpr uint32_t kKvSize = kKvHeads * kHeadSize;
constexpr uint32_t kPackedQkvSize = kQuerySize + 2u * kKvSize;
constexpr uint32_t kIntermediateSize = 4864;
constexpr uint32_t kAttentionProjectionPartitions = 8;
constexpr uint32_t kFfnProjectionPartitions = 20;
constexpr uint32_t kDownProjectionPartitions = 10;
constexpr uint32_t kAttentionTaskCount = 9;
constexpr uint32_t kFfnTaskCount = 32;
constexpr uint32_t kTaskCount = kAttentionTaskCount + kFfnTaskCount;

enum class Opcode : uint32_t {
  attention_rmsnorm_f32 = 1,
  qkv_projection_f32 = 2,
  mlp_rmsnorm_f32 = 3,
  gate_up_silu_f32 = 4,
  down_matvec_f32 = 5,
  residual_advance = 6,
};

// Device pointers and dimensions consumed by the Qwen decoder-stage
// megakernel. The scheduler sees this only as the opaque context pointer in
// ResidentArgs.
struct alignas(16) Context {
  float *hidden;
  const float *attention_norm_weight;
  const float *q_weight;
  const float *k_weight;
  const float *v_weight;
  const float *q_bias;
  const float *k_bias;
  const float *v_bias;
  const float *mlp_norm_weight;
  const float *gate_weight;
  const float *up_weight;
  const float *down_weight;
  float *attention_normalized;
  float *query;
  float *key;
  float *value;
  float *mlp_normalized;
  float *activated;
  float *projected;
  uint32_t hidden_size;
  uint32_t query_size;
  uint32_t kv_size;
  uint32_t intermediate_size;
  uint32_t reserved[2];
};

static_assert(std::is_standard_layout_v<Context>);
static_assert(sizeof(Context) == 176);
static_assert(kTaskCount <= kMaximumTasks);

} // namespace lrrt::stateful_graph::qwen_decode

#endif // LRRT_STATEFUL_GRAPH_QWEN_DECODE_ABI_HPP_
