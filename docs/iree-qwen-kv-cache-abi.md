# IREE Qwen KV Cache ABI

This document defines the current lrrt/IREE Qwen KV cache boundary and the
direction for extending it from the fixed three-token milestone to an
autoregressive decode loop.

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

## Current Fixed ABI

The real-weight Qwen E2E runner currently supports one-, two-, and three-token
decode milestones.

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

## Decode Loop Contract

The runner-level decode loop is:

```text
for token_step in 0..decode_steps:
  hidden = embedding(current_token)
  for layer in 0..num_layers:
    if token_step == 0:
      key_cache[layer], value_cache[layer], hidden =
        qwen_decode1_layer(hidden, weights[layer])
    else:
      key_cache[layer], value_cache[layer], hidden =
        qwen_decodeN_layer(hidden, key_cache[layer], value_cache[layer],
                           weights[layer], token_step)
logits = qwen_decode1_tail(hidden, tail_weights)
current_token = argmax(logits)
```

The runner uses the first token id stored in the tail bundle as the initial
token, then feeds the logits-selected `top_token` back through the tail bundle's
token embedding table for the next step. Multi-step generation therefore
requires a tail bundle that contains the selected token ids. The converter's
`--full-token-embeddings` option stores every embedding row and is the expected
format for real E2E generation experiments.

The current implementation only has VMFB exports for `token_step == 0`,
`token_step == 1`, and `token_step == 2`. Requests beyond three decode steps
must fail explicitly until a variable-length cache VMFB is introduced.

## Variable-Length ABI Direction

To support full autoregressive generation, the next VMFB-level ABI should stop
encoding the number of visible tokens in the function name and static output
shape. The target shape is:

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
