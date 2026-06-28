# Mini Inference Pipeline Sketch

## Purpose

This document tracks how the current Triton bundle examples should become a
small Qwen-style inference and benchmark path on top of
`light-rocm-runtime`.

The goal is not to implement a model runtime in the core library. The goal is
to identify the smallest useful operator chain, the runtime/executor
responsibilities, and the gaps that must be closed before a tiny decoder block
can run end to end.

The project-level target for this pipeline is:

```text
run a small Qwen-style decode path through lrrt and report benchmark numbers
```

The first benchmark should measure a fixed-shape, deterministic decode step:
one token enters a tiny decoder block, kernels are dispatched through the
Triton executor layer, outputs are validated against a CPU reference, and the
GPU execution path reports latency. Model fidelity can grow after this target
is stable.

## Stack Boundary

The intended stack is:

```text
model / compiler layer
  -> operator bundles: HSACO + manifest.json
  -> lightweight executor: buffer ownership, queueing, dependencies
  -> light-rocm-runtime: memory, module load, dispatch, synchronization
  -> HSA / ROCm
```

`light-rocm-runtime` should stay focused on low-overhead dispatch and
predictable resource management. It should not own tensor graph semantics,
token scheduling, model weights, tokenizer state, or compiler IR.

## Operator Status

The first pipeline should be driven by the operators needed for a small
decoder-style path, not by every example currently present in the repository.

| Pipeline need | Current status | Existing coverage | Gap |
| --- | --- | --- | --- |
| RMSNorm | ✅ | `rmsnorm` with FP32/FP16/BF16 input variants and FP32 reference validation | Not yet connected into the decoder-block integration path |
| Q/K/V projection | ✅/❌ | `matvec` covers one-vector FP32 projection | No batched projection, no tiled GEMM, no FP16/BF16 inputs yet |
| RoPE | ✅ | `rope` for FP32 vectors up to the current specialization limits | Connected in mini attention before K cache update; not yet connected to projection outputs |
| Attention score computation | ✅ | `attention_score` computes FP32 Q x K dot products with scaling | Covered in the mini attention integration path |
| Causal softmax | ✅ | `causal_softmax` covers FP32 future-token masking with query offsets | Covered in the mini attention integration path |
| Value aggregation | ✅ | `value_aggregation` computes FP32 weighted sums over V vectors | Covered in the mini attention integration path |
| Residual add | ✅ | `vector_add` is connected in the mini attention integration path | Needs reuse in a larger decoder-block path |
| Gated MLP activation | ✅ | `silu_mul` covers `SiLU(gate) * up` in FP32 | Not yet connected to MLP projection outputs |
| Output projection | ✅/❌ | `matvec` can stand in for a small FP32 output projection | Needs larger/batched projection support and lower precision |
| KV cache update/read | ✅ | `kv_cache_update` and `kv_cache_read` cover FP32 row-major `[max_tokens, head_dim]` cache writes and indexed reads; mini attention writes RoPE-applied K into cache and reads K/V through cache buffers | Still single-head row-major layout |
| Benchmark timing | ✅/❌ | runtime event timing and benchmark formatting exist | Not yet attached to the Qwen-style decode path |

## Decoder Block Shape

A minimal decoder-block-style pipeline can be described as:

```text
input hidden
  -> RMSNorm
  -> Q/K/V projections
  -> RoPE on Q and K
  -> attention scores
  -> causal softmax
  -> attention value projection
  -> residual add
  -> RMSNorm
  -> gated MLP projections
  -> SiLU * up
  -> output projection
  -> residual add
```

The current examples cover only part of this chain. The available `matvec`
example can stand in for a small projection, but it is not yet a complete GEMM
or batched matmul implementation.

## Current Integration Baseline

The current multi-kernel baseline is `triton_mini_attention`. It exercises the
important executor concerns before a full decoder block exists:

```text
Q/K/V-like vectors
  -> RoPE on Q and K
  -> KV cache update/read
  -> attention scores
  -> causal softmax
  -> value aggregation
  -> residual add
```

This is not a faithful transformer block. It is a controlled integration test
that exercises the same executor concerns:

- multiple bundle loads
- multiple device buffers
- repeated argument binding
- ordered launches on one queue
- CPU reference validation between or after stages
- predictable buffer lifetime

The next target should add the missing decoder-block pieces around this
baseline, especially projection coverage, RMSNorm placement, MLP projection
wiring, and benchmark timing.

## Executor Responsibilities

An executor above `lrrt::Bundle` should own:

- model buffer allocation and lifetime
- temporary workspace allocation from `workspace_bytes`
- binding tensor pointers and scalar dimensions into `KernargBuffer`
- selecting a bundle specialization based on shape and dtype
- queue selection and event dependency wiring
- launch ordering across multiple kernels
- final synchronization policy
- benchmark timing boundaries for executor-owned pipeline stages

The executor should not compile kernels, parse a graph IR, or hide the bundle
ABI. It should be a thin coordinator around already-generated bundles.

The current `executor/triton` helper is the first version of this layer. It
owns named bundle and buffer lookup, common launch argument binding, and
diagnostic wrapping for executor-level failures. It is intentionally smaller
than a graph runtime.

## Runtime Responsibilities

The runtime should continue to own:

- HSA initialization and shutdown
- device discovery and opening
- device allocation and host/device copies
- HSACO load and kernel symbol resolution
- queue-backed kernel launch
- event and synchronization primitives
- bundle manifest parsing and lightweight ABI validation

Runtime support for this pipeline should mean reliable dispatch and resource
management, not high-level model execution.

## Missing Pieces

The current operator examples leave several gaps before a Qwen-like path can be
meaningful:

- **Decoder-block executor shape**: `executor/triton` now provides reusable
  bundle, buffer, and launch helpers, but there is not yet a Qwen-style
  decoder-block runner that wires RMSNorm, projections, attention, MLP, and
  residuals together.
- **Residual add**: the mini attention path now covers attention output plus
  residual stream, but this should be reused in a larger decoder-block example.
- **KV cache layout**: mini attention now stores RoPE-applied K and raw V in
  cache buffers, but the layout is still a single-head row-major
  `[max_tokens, head_dim]` buffer.
- **FP16/BF16 matvec**: Qwen-style inference should use lower precision inputs
  with FP32 accumulation where appropriate.
- **Shape metadata**: the current manifest describes launch ABI, not tensor
  shape semantics. The executor must provide shape policy outside the runtime
  core.
- **Decode benchmark**: runtime timing primitives exist, but the Qwen-style
  integration path does not yet report per-stage or end-to-end decode latency.

## Suggested Milestones

### P0: Pipeline Documentation

- Keep this document as the high-level integration and benchmark sketch.
- Avoid adding graph semantics to the runtime core.

### P1: Thin Executor Prototype

- Status: mostly complete.
- `executor/triton` now provides reusable named bundle, named buffer, and
  manifest-driven launch helpers.
- `triton_mini_attention` and `triton_kv_cache` use this helper for multi-op
  executor-style examples.
- Keep further executor changes focused on repeated wiring, diagnostics,
  specialization selection, and benchmark boundaries.

### P2: Causal Softmax

- Status: mostly complete.
- Keep the Triton causal softmax bundle validated against a CPU reference.
- Connect masked softmax to attention score output in an executor path.
- Keep mask behavior in the operator example, not the runtime core.

### P3: Projection Coverage

- Extend `matvec` toward the smallest useful attention projection path.
- Decide whether the next step is batched matvec, tiled GEMV, or a small GEMM
  example.
- Add the projection behavior needed for Q/K/V, output projection, MLP gate/up,
  MLP down, and final logits in a tiny fixed-shape model.
- Prefer the smallest shape-specialized Triton kernels that make the decoder
  path realistic enough to benchmark.

### P4: Tiny Decoder Block

- Combine RMSNorm, projection, RoPE, causal softmax, value aggregation, MLP
  activation, and residual operations into one fixed-shape integration example.
- Treat it as an executor test, not as a general model runtime.
- Validate every stage, or at least the final hidden state, against a CPU
  reference.

### P5: Small Qwen-Style Decode Benchmark

- Add a deterministic tiny Qwen-style decode example with fixed weights,
  fixed shapes, and a CPU reference.
- Run one-token decode through lrrt using Triton-generated bundles.
- Measure end-to-end decode latency and, where useful, per-stage latency using
  runtime event timing.
- Report enough benchmark metadata to make results reproducible: target arch,
  dtype, hidden size, head count, head dimension, cache length, layer count,
  warmup iterations, and measured iterations.
- Keep tokenizer, sampling, and external model loading out of scope until the
  fixed-shape decode benchmark is stable.

## Non-Goals

- Do not implement tokenizer, sampling, or model loading in lrrt.
- Do not make `light-rocm-runtime` own a tensor graph or operator scheduler.
- Do not require the runtime core to understand Qwen-specific semantics.
- Do not hide bundle manifests behind a framework-specific API yet.
