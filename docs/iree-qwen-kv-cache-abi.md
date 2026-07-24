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

The preferred path is now the `qwen_decode_layer_kv_cache_max<N>` family, which
removes the need for a new VMFB per token step. Each VMFB uses a fixed cache
capacity specialization and an explicit current position. The current supported
specializations are `max8`, `max16`, and `max32`.

| Input | Shape | Meaning |
| --- | --- | --- |
| `hidden` | `1x896xf32` | Current token hidden state |
| `old_key_cache` | `Nx128xf32` | Layer-local K cache storage |
| `old_value_cache` | `Nx128xf32` | Layer-local V cache storage |
| `position` | `1xi32` | Current token index to append |
| layer weights | fixed Qwen 0.5B fp32 views | RMSNorm, Q/K/V/O, MLP gate/up/down weights |

It returns:

| Output | Shape | Meaning |
| --- | --- | --- |
| `key_cache` | `Nx128xf32` | Updated K cache storage |
| `value_cache` | `Nx128xf32` | Updated V cache storage |
| `hidden` | `1x896xf32` | Layer output |

The VMFB inserts the current token's K/V rows at `position` and masks attention
scores so softmax only sees columns `0..position`. Future cache rows can
therefore be zero-initialized without changing the attention result. This is not
fully unbounded dynamic shape support yet; it is an arbitrary-step runner ABI
within the compiled `max_tokens = N` capacity.

The runner exposes this path as:

```text
lrrt_iree_qwen_decode1_e2e --max-new-tokens N --max-cache-tokens <8|16|32> \
  <qwen_decode_layer_kv_cache_max*.vmfb> <tail.vmfb> <weights-dir> [layers]
```

`N` must be less than or equal to the selected cache capacity. This direct
runner form is a low-level VMFB ABI/debug interface; user-facing inference
wrappers should generally expose `max_seq_len` instead and derive the cache
capacity from it.

The preferred path is to put that execution ABI in an IREE Qwen decode bundle
manifest:

```json
{
  "manifest_version": 1,
  "target": "gfx1101",
  "layer_vmfb": "qwen_decode_layer_kv_cache_max8_gfx1101.vmfb",
  "tail_vmfb": "qwen_decode1_tail_gfx1101.vmfb",
  "layer_export": "qwen_decode_layer_kv_cache_max8",
  "tail_export": "qwen_decode1_tail",
  "sequence_capacity": 8,
  "max_cache_tokens": 8,
  "kv_cache_shape": [8, 128]
}
```

The runner accepts this bundle form:

```text
lrrt_iree_qwen_decode1_e2e --max-new-tokens N --max-seq-len S \
  --bundle <bundle-dir> <weights-dir> [layers]
```

The higher-level `tools/run_qwen_e2e.py` wrapper exposes the inference-side
limit as `--max-seq-len`. The wrapper discovers available VMFB cache capacities
from the IREE probe output directory and chooses the smallest VMFB that can hold
that sequence length. For example, if a `max16` VMFB exists,
`--max-seq-len 10` selects the `qwen_decode_layer_kv_cache_max16` VMFB and writes
`sequence_capacity: 10` and `max_cache_tokens: 16` to the decode bundle
manifest.
The wrapper exposes output length separately as `--max-new-tokens`; that value
must be positive and must not exceed `--max-seq-len`. The runner validates
`max_new_tokens <= max_seq_len <= sequence_capacity`; `max_cache_tokens` is the
current physical static tensor extent of the VMFB.
When an existing decode bundle is reused, the wrapper reads its manifest and
rejects `--max-seq-len` values larger than the recorded `sequence_capacity`
before launching the runner.
The full 24-layer Qwen E2E path has been validated through
`--max-new-tokens 32 --max-seq-len 32` using the `max32` cache specialization.

The bundle can be created from compiled VMFB artifacts with:

```text
tools/write_iree_qwen_decode_bundle.py \
  --target gfx1101 \
  --layer-vmfb <qwen_decode_layer_kv_cache_max*.vmfb> \
  --tail-vmfb <qwen_decode1_tail.vmfb> \
  --out-dir <bundle-dir> \
  --sequence-capacity <max_seq_len> \
  --max-cache-tokens <vmfb_cache_capacity>
```

The VMFB paths are relative to `<bundle-dir>`. Absolute paths and `..` path
components are rejected so that a bundle manifest cannot silently point outside
the bundle directory. `max_cache_tokens` is a bundle ABI field rather than a
user-facing inference setting. It must match `kv_cache_shape[0]`; the runner
still treats the cache as opaque f32 device tensors with shape
`[max_cache_tokens, 128]`. `sequence_capacity` must be positive and cannot
exceed `max_cache_tokens`; this keeps the current static-shape implementation
compatible with a future single-capacity bundle where runtime `max_seq_len`
selects the usable prefix.

## Decode Loop Contract

The runner-level decode loop is:

```text
for token_step in 0..decode_steps:
  hidden = embedding(current_token)
  for layer in 0..num_layers:
    key_cache[layer], value_cache[layer], hidden =
      qwen_decode_layer_kv_cache_maxN(hidden,
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
format for real E2E generation experiments. The high-level IREE E2E wrapper
passes that converter option automatically and also requests a directory bundle,
so even a one-layer run produces both `layer_0` and `model_tail`.

## Longer-Context ABI Direction

To support full autoregressive generation beyond the current fixed
specializations, the VMFB-level ABI should eventually move to larger
manifest-level specializations or dynamic cache shapes where practical. The
target shape is:

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
