# Mini Inference Pipeline Sketch

## Purpose

This document sketches how the current Triton bundle examples could become a
small Qwen-style inference path on top of `light-rocm-runtime`.

The goal is not to implement a model runtime in the core library. The goal is
to identify the smallest useful operator chain, the runtime/executor
responsibilities, and the gaps that must be closed before a tiny decoder block
can run end to end.

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
| RMSNorm | ✅ | `rmsnorm` with FP32/FP16/BF16 input variants and FP32 reference validation | Not yet connected into a multi-kernel executor path |
| Q/K/V projection | ✅/❌ | `matvec` covers one-vector FP32 projection | No batched projection, no tiled GEMM, no FP16/BF16 inputs yet |
| RoPE | ✅ | `rope` for FP32 vectors up to the current specialization limits | Connected in mini attention before K cache update; not yet connected to projection outputs |
| Attention score computation | ✅ | `attention_score` computes FP32 Q x K dot products with scaling | Covered in the mini attention integration path |
| Causal softmax | ✅ | `causal_softmax` covers FP32 future-token masking with query offsets | Covered in the mini attention integration path |
| Value aggregation | ✅ | `value_aggregation` computes FP32 weighted sums over V vectors | Covered in the mini attention integration path |
| Residual add | ✅ | `vector_add` is connected in the mini attention integration path | Needs reuse in a larger decoder-block path |
| Gated MLP activation | ✅ | `silu_mul` covers `SiLU(gate) * up` in FP32 | Not yet connected to MLP projection outputs |
| Output projection | ✅/❌ | `matvec` can stand in for a small FP32 output projection | Needs larger/batched projection support and lower precision |
| KV cache update/read | ✅ | `kv_cache_update` and `kv_cache_read` cover FP32 row-major `[max_tokens, head_dim]` cache writes and indexed reads; mini attention writes RoPE-applied K into cache and reads K/V through cache buffers | Still single-head row-major layout |

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

## Proposed First End-to-End Target

The first useful target should be smaller than a full Qwen block:

```text
hidden[hidden]
  -> rmsnorm_fp32
  -> matvec_fp32 for one projection
  -> softmax_fp32 over a small score row
  -> matvec_fp32 for one output projection
```

This target is not a faithful transformer block. It is a controlled integration
test that exercises the same executor concerns:

- multiple bundle loads
- multiple device buffers
- repeated argument binding
- ordered launches on one queue
- CPU reference validation between or after stages
- predictable buffer lifetime

Once this small chain is stable, RoPE, causal masking, and gated MLP pieces can
be added incrementally.

## Executor Responsibilities

An executor above `lrrt::Bundle` should own:

- model buffer allocation and lifetime
- temporary workspace allocation from `workspace_bytes`
- binding tensor pointers and scalar dimensions into `KernargBuffer`
- selecting a bundle specialization based on shape and dtype
- queue selection and event dependency wiring
- launch ordering across multiple kernels
- final synchronization policy

The executor should not compile kernels, parse a graph IR, or hide the bundle
ABI. It should be a thin coordinator around already-generated bundles.

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

- **Executor generalization**: the mini attention example connects attention
  score, causal softmax, value aggregation, and residual add, but it is still a
  fixed example rather than a reusable executor abstraction.
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

## Suggested Milestones

### P0: Pipeline Documentation

- Keep this document as the high-level integration sketch.
- Avoid adding graph semantics to the runtime core.

### P1: Thin Executor Prototype

- `triton_mini_attention` now contains a fixed-shape executor prototype.
- It loads a fixed list of bundles, owns named device buffers, applies RoPE to
  Q/K, fills valid K/V cache rows, launches three attention kernels plus a
  residual-add kernel in order on one queue, and validates the final buffer
  against a CPU reference.
- Keep the next step focused on generalizing only the parts that are repeated
  by additional integration examples.

### P2: Causal Softmax

- Keep the Triton causal softmax bundle validated against a CPU reference.
- Connect masked softmax to attention score output in an executor path.
- Keep mask behavior in the operator example, not the runtime core.

### P3: Projection Coverage

- Extend `matvec` toward the smallest useful attention projection path.
- Decide whether the next step is batched matvec, tiled GEMV, or a small GEMM
  example.

### P4: Tiny Decoder Block

- Combine RMSNorm, projection, RoPE, causal softmax, value aggregation, MLP
  activation, and residual operations into one fixed-shape integration example.
- Treat it as an executor test, not as a general model runtime.

## Non-Goals

- Do not implement tokenizer, sampling, or model loading in lrrt.
- Do not make `light-rocm-runtime` own a tensor graph or operator scheduler.
- Do not require the runtime core to understand Qwen-specific semantics.
- Do not hide bundle manifests behind a framework-specific API yet.
