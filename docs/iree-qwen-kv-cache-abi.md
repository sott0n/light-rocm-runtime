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
| layer weights | fixed Qwen 0.5B fp32 views | RMSNorm, Q/K/V/O and MLP gate/up/down weights |

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
specializations are `max8`, `max16`, `max32`, and `max64`, each compiled for
`f32`, `f16`, and `bf16`. CMake generates them from
`tools/iree_qwen_decode_layer_kv_cache.mlir.in`, so extending the capacity or
precision list does not duplicate the Qwen layer implementation.

| Input | Shape | Meaning |
| --- | --- | --- |
| `hidden` | `1x896xT` | Current token hidden state |
| `old_key_cache` | `Nx128xT` | Layer-local K cache storage |
| `old_value_cache` | `Nx128xT` | Layer-local V cache storage |
| `position` | `1xi32` | Current token index to append |
| layer weights | fixed Qwen 0.5B `T` views | RMSNorm, Q/K/V/O, MLP gate/up/down weights, Q/K/V biases, and `rope_theta` |

`T` is the precision recorded by the decode bundle: `f32`, `f16`, or `bf16`.

It returns:

| Output | Shape | Meaning |
| --- | --- | --- |
| `key_cache` | `Nx128xT` | Updated K cache storage |
| `value_cache` | `Nx128xT` | Updated V cache storage |
| `hidden` | `1x896xT` | Layer output |

The VMFB adds Q/K/V projection biases, applies Qwen's split-half
`rotate_half` RoPE convention to Q and K using `position` and `rope_theta`,
inserts the rotated K and unrotated V rows at `position`, and masks attention
scores so softmax only sees columns `0..position`. Future cache rows can
therefore be zero-initialized without changing the attention result. This is
not fully unbounded dynamic shape support yet; it is an arbitrary-step runner
ABI within the compiled `max_tokens = N` capacity.

The runner exposes this path as:

```text
lrrt_iree_qwen_decode1_e2e --max-new-tokens N --max-cache-tokens <8|16|32|64> \
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
  "manifest_version": 2,
  "target": "gfx1101",
  "precision": "bf16",
  "layer_vmfb": "qwen_decode_layer_kv_cache_max8_bf16_gfx1101.vmfb",
  "tail_vmfb": "qwen_decode1_tail_bf16_gfx1101.vmfb",
  "layer_export": "qwen_decode_layer_kv_cache_max8_bf16",
  "tail_export": "qwen_decode1_tail_bf16",
  "sequence_capacity": 8,
  "max_cache_tokens": 8,
  "kv_cache_shape": [8, 128]
}
```

Manifest version 1 remains accepted and implies `precision: "f32"`.

The runner accepts this bundle form:

```text
lrrt_iree_qwen_decode1_e2e --max-new-tokens N --max-seq-len S \
  [--eos-token-id <id>] --bundle <bundle-dir> <weights-dir> [layers]
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
before launching the runner. It also rejects a reused bundle whose `precision`
does not match `--iree-precision`.
The full 24-layer Qwen E2E path has been validated with an 11-token prompt and
40 generated tokens using `--max-new-tokens 40 --max-seq-len 64`. All 40
greedy decode steps matched the Hugging Face FP32 reference with a maximum
top-logit absolute difference of `5.9552002e-05`.

When `--eos-token-id` is present, the runner includes the selected EOS token in
`generated_token_ids=[...]` and stops before the next decode step. It reports
`stop_reason=eos_token` for that path and
`stop_reason=max_new_tokens` when the requested output limit is reached.

The native runner flushes concise progress records as work begins and
completes. These identify module initialization, weight loading, every prompt
token entering prefill, every generated decode step, and elapsed time for
completed prefill, decode, and startup phases. The runner waits for every VM
invocation fence. Model-tail weights are transposed and uploaded once during
weight loading, then reused across decode steps. This makes an interrupted or
slow run distinguishable from a silent startup stall without producing one
line per decoder layer. Pass
`--verbose-layers` to the native runner, or `--iree-verbose-layers` through
`tools/run_qwen_e2e.py`, to additionally print
per-layer weight loading and `step <N> layer <M>/<layers> complete` after every
decoder layer.

The bundle can be created from compiled VMFB artifacts with:

```text
tools/write_iree_qwen_decode_bundle.py \
  --target gfx1101 \
  --precision <f32|f16|bf16> \
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
treats the cache as opaque device tensors of the manifest precision with shape
`[max_cache_tokens, 128]`. `sequence_capacity` must be positive and cannot
exceed `max_cache_tokens`; this keeps the current static-shape implementation
compatible with a future single-capacity bundle where runtime `max_seq_len`
selects the usable prefix.

The shared checkpoint bundle remains FP32 on disk. At startup the runner
transposes layer matrices, converts them to the selected device precision, and
uploads them. Token embeddings are converted as they enter the decoder. The
precision-specific tail accepts the final `f16` or `bf16` hidden state, extends
it to `f32`, and runs final RMSNorm and the language-model head with FP32
weights and FP32 logits.

For batch-one decode, generated FP16 modules select explicit `linalg.vecmat`
operations for the seven dominant decoder projections. Generated BF16 modules
use 16-row WMMA contractions with BF16 inputs and weights and FP32
accumulation for the dominant projections. Conversion back to BF16 is combined
with the following bias, residual, or SiLU operation. K/V use fourteen
64-element BF16 WMMA partial reductions followed by an FP32 sum; accumulating
the complete 896-element reduction in BF16 was faster but changed the tested
greedy sequence. FP32 decoder modules retain `linalg.matmul` because vecmat
regressed FP32 latency. The FP16/BF16 tail modules use vecmat for their FP32
language-model head.

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
