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
constexpr uint32_t kRotarySize = kHeadSize / 2u;
constexpr uint32_t kRotaryHeads = kAttentionHeads + kKvHeads;
constexpr uint32_t kQuerySize = kAttentionHeads * kHeadSize;
constexpr uint32_t kKvSize = kKvHeads * kHeadSize;
constexpr uint32_t kPackedQkvSize = kQuerySize + 2u * kKvSize;
constexpr uint32_t kIntermediateSize = 4864;
constexpr uint32_t kAttentionProjectionPartitions = 8;
constexpr uint32_t kAttentionTaskPartitions = 7;
constexpr uint32_t kAttentionOutputPartitions = 8;
constexpr uint32_t kFfnProjectionPartitions = 20;
constexpr uint32_t kDownProjectionPartitions = 10;
constexpr uint32_t kMaximumLmHeadPartitions = 32;
constexpr uint32_t kAttentionTaskCount = 2u + kAttentionProjectionPartitions +
                                         kAttentionTaskPartitions +
                                         kAttentionOutputPartitions;
constexpr uint32_t kFfnTaskCount = 32;
constexpr uint32_t kDecoderTaskCount = kAttentionTaskCount + kFfnTaskCount;
constexpr uint32_t kMaximumTailTaskCount = 2u + kMaximumLmHeadPartitions;

enum class Opcode : uint32_t {
  attention_rmsnorm_f32 = 1,
  qkv_projection_f32 = 2,
  rope_kv_cache_f32 = 3,
  grouped_query_attention_f32 = 4,
  attention_output_residual_f32 = 5,
  mlp_rmsnorm_f32 = 6,
  gate_up_silu_f32 = 7,
  down_matvec_f32 = 8,
  residual_advance = 9,
  final_rmsnorm_f32 = 10,
  lm_head_top1_partition_f32 = 11,
  top1_reduce = 12,
};

// Shared model-tail state. Every decoder-layer Context points to the same
// instance so a conditional tail Program can run after the final layer without
// changing the scheduler's opaque-context ABI.
struct alignas(16) TailContext {
  const float *final_norm_weight;
  const float *lm_head_weight;
  float *final_normalized;
  float *partial_max_values;
  uint32_t *partial_max_indices;
  uint32_t *sampled_token;
  float *sampled_logit;
  uint32_t vocab_size;
  uint32_t partition_count;
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
  const float *out_weight;
  const float *mlp_norm_weight;
  const float *gate_weight;
  const float *up_weight;
  const float *down_weight;
  float *attention_normalized;
  float *query;
  float *key;
  float *value;
  const float *rope_cos;
  const float *rope_sin;
  float *key_cache;
  float *value_cache;
  float *mlp_normalized;
  float *activated;
  float *projected;
  TailContext *tail;
  uint32_t hidden_size;
  uint32_t query_size;
  uint32_t kv_size;
  uint32_t intermediate_size;
  uint32_t cache_capacity;
  // The opaque resident context is an array of this many layer contexts.
  // Scratch buffers may be shared, but weights and KV-cache slices are
  // layer-local.
  uint32_t layer_count;
};

static_assert(std::is_standard_layout_v<Context>);
static_assert(std::is_standard_layout_v<TailContext>);
static_assert(sizeof(Context) == 224);
static_assert(sizeof(TailContext) == 64);
static_assert(kAttentionTaskCount <= kMaximumTasks);
static_assert(kFfnTaskCount <= kMaximumTasks);
static_assert(kMaximumTailTaskCount <= kMaximumTasks);

} // namespace lrrt::stateful_graph::qwen_decode

#endif // LRRT_STATEFUL_GRAPH_QWEN_DECODE_ABI_HPP_
