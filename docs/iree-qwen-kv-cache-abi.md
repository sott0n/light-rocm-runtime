# IREE Qwen KV Cache ABI

This document defines the current lrrt/IREE Qwen KV cache boundary for the
real-weight Qwen decode runner.

## Runtime Boundary

`light-rocm-runtime` should continue to act as a low-overhead dispatcher and
predictable resource manager. It should not own Qwen attention semantics,
tokenizer behavior, graph scheduling, or cache layout policy.

For IREE integration, the boundary is:

- IREE-compiled VMFB exports implement the model math and choose tensor shapes.
- The lrrt IREE HAL adapter executes the VMFB dispatches and owns GPU resource
  allocation, queue submission, synchronization, and buffer-view handoff.
- The Qwen runner owns model-level sequencing: layer order, token steps,
  per-layer cache slots, checkpoint weight views, and final logits inspection.

## Fixed Decode ABI

The real-weight Qwen E2E runner still supports the original one-, two-, and
three-token decode milestones.

`qwen_decode1_layer` consumes:

| Input | Shape | Meaning |
| --- | --- | --- |
| `hidden` | `1x896xf32` | Current token hidden state entering one layer |
| layer weights | fixed Qwen 0.5B fp32 views | RMSNorm, Q/K/V/O, MLP gate/up/down weights |

It returns:

| Output | Shape | Meaning |
| --- | --- | --- |
| `key_cache` | `1x128xf32` | Token-0 K cache for this layer |
| `value_cache` | `1x128xf32` | Token-0 V cache for this layer |
| `hidden` | `1x896xf32` | Layer output |

`qwen_decode2_layer_kv_cache` consumes the current token hidden state, the
previous layer-local cache from step 1, and the same layer weights:

| Input | Shape | Meaning |
| --- | --- | --- |
| `hidden` | `1x896xf32` | Current token hidden state |
| `old_key_cache` | `1x128xf32` | Previous visible K cache |
| `old_value_cache` | `1x128xf32` | Previous visible V cache |
| layer weights | fixed Qwen 0.5B fp32 views | RMSNorm, Q/K/V/O, MLP gate/up/down weights |

It returns:

| Output | Shape | Meaning |
| --- | --- | --- |
| `key_cache` | `2x128xf32` | Updated visible K cache for this layer |
| `value_cache` | `2x128xf32` | Updated visible V cache for this layer |
| `hidden` | `1x896xf32` | Layer output |

`qwen_decode3_layer_kv_cache` follows the same contract with a `2x128xf32`
input cache and a `3x128xf32` output cache.

The `128` dimension is Qwen 0.5B's full KV width:

```text
kv_heads = 2
head_dim = 64
kv_dim = kv_heads * head_dim = 128
```

The runner keeps one `key_cache`/`value_cache` pair per decoder layer as IREE
`buffer_view`s. The cache remains device-resident between token steps; the
runner passes the previous step's buffer views directly into the next VMFB
invocation for the same layer.

## Max-Cache Decode ABI

The preferred path is now `qwen_decode_layer_kv_cache_max8`, which removes the
need for a new VMFB per token count. It uses a fixed cache capacity and an
explicit current position:

| Input | Shape | Meaning |
| --- | --- | --- |
| `hidden` | `1x896xf32` | Current token hidden state |
| `old_key_cache` | `8x128xf32` | Layer-local K cache storage |
| `old_value_cache` | `8x128xf32` | Layer-local V cache storage |
| `position` | `1xi32` | Current token index to append |
| layer weights | fixed Qwen 0.5B fp32 views | RMSNorm, Q/K/V/O, MLP gate/up/down weights |

It returns:

| Output | Shape | Meaning |
| --- | --- | --- |
| `key_cache` | `8x128xf32` | Updated K cache storage |
| `value_cache` | `8x128xf32` | Updated V cache storage |
| `hidden` | `1x896xf32` | Layer output |

The VMFB inserts the current token's K/V rows at `position` and masks attention
scores so softmax only sees columns `0..position`. Future cache rows can
therefore be zero-initialized without changing the attention result. This is not
fully unbounded dynamic shape support yet; it is an arbitrary-step runner ABI
within the compiled `max_tokens = 8` capacity.

The runner exposes this path as:

```text
lrrt_iree_qwen_decode1_e2e --steps N --max-cache-tokens 8 \
  <qwen_decode_layer_kv_cache_max8.vmfb> <tail.vmfb> <weights-dir> [layers]
```

`N` must be less than or equal to `8`.

## Decode Loop Contract

The runner-level decode loop is:

```text
for token_step in 0..decode_steps:
  hidden = embedding(current_token)
  for layer in 0..num_layers:
    key_cache[layer], value_cache[layer], hidden =
      qwen_decode_layer_kv_cache_max8(hidden,
                                      key_cache[layer],
                                      value_cache[layer],
                                      position=token_step,
                                      weights[layer])
logits = qwen_decode1_tail(hidden, tail_weights)
current_token = argmax(logits)
```

The runner uses the first token id stored in the tail bundle as the initial
token, then feeds the logits-selected `top_token` back through the tail bundle's
token embedding table for the next step. Multi-step generation therefore
requires a tail bundle that contains the selected token ids. The converter's
`--full-token-embeddings` option stores every embedding row and is the expected
format for real E2E generation experiments.

## Longer-Context ABI Direction

To support full autoregressive generation beyond the current fixed
`max_tokens = 8` VMFB, the VMFB-level ABI should make the cache capacity a
manifest-level specialization or move to dynamic cache shapes where practical.
The target shape is:

| Field | Direction | Meaning |
| --- | --- | --- |
| `hidden` | input | Current token hidden state |
| `key_cache` | input/output | Layer-local K cache storage |
| `value_cache` | input/output | Layer-local V cache storage |
| `position` | input scalar | Current token index |
| `valid_tokens` | input scalar | Visible prefix length for attention |
| layer weights | input | Layer-local checkpoint weights |
| `hidden` | output | Updated hidden state |

The preferred cache layout for Qwen 0.5B is:

```text
key_cache:   [max_tokens, kv_heads, head_dim] or [max_tokens, kv_dim]
value_cache: [max_tokens, kv_heads, head_dim] or [max_tokens, kv_dim]
```

The layout should be chosen by the compiler/export layer and documented in the
VMFB manifest. lrrt should treat the cache as an opaque device allocation whose
shape is validated at the runner or adapter boundary.

## Non-Goals

- Do not move Qwen attention, RoPE, sampling, or tokenizer semantics into the
  runtime core.
- Do not make the HAL adapter parse model graphs or infer cache layouts.
- Do not copy KV cache through the host between token steps.
- Do not optimize synchronization before the arbitrary-step E2E path is
  functionally correct.
