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
one token enters a tiny decoder layer, kernels are dispatched through the
Triton executor layer, outputs are validated against a CPU reference in the
example path, and the GPU execution path reports latency in the benchmark path.
Model fidelity can grow after this target is stable.

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
| RMSNorm | ✅ | `rmsnorm` with FP32/FP16/BF16 input variants and FP32 reference validation; used by `triton_mini_decoder_layer` before attention and MLP | Still fixed-shape/specialization driven |
| Q/K/V projection | ✅/❌ | `matvec` covers one-vector FP32 projection and is wired into `triton_mini_decoder_layer` for Q/K/V | No batched projection or tiled GEMM; lower precision bundle variants exist for matvec tests but the decoder layer path is still FP32 |
| RoPE | ✅ | `rope` for FP32 vectors up to the current specialization limits; applied to Q and K projection outputs in the decoder layer path | Still single-head in the decoder layer path |
| Attention score computation | ✅ | `attention_score` computes FP32 Q x K dot products with scaling and is used in mini attention and mini decoder layer | Still single-head row-major cache layout |
| Causal softmax | ✅ | `causal_softmax` covers FP32 future-token masking with query offsets and is used in mini attention and mini decoder layer | Needs broader shape coverage for model-like cache lengths |
| Value aggregation | ✅ | `value_aggregation` computes FP32 weighted sums over V vectors and is used in mini attention and mini decoder layer | Still single-head |
| Residual add | ✅ | `vector_add` is used for attention residual and MLP residual in `triton_mini_decoder_layer` | Only FP32 path is wired into the decoder layer |
| Gated MLP activation | ✅ | `silu_mul` covers `SiLU(gate) * up` in FP32 and is wired into `triton_mini_mlp` and `triton_mini_decoder_layer` | Needs lower precision and larger shape coverage later |
| Output projection | ✅/❌ | `matvec` stands in for attention output projection and MLP down projection in the mini decoder layer | Needs larger/batched projection support and lower precision in the end-to-end path |
| KV cache update/read | ✅ | `kv_cache_update` and `kv_cache_read` cover FP32 row-major `[max_tokens, head_dim]` cache writes and indexed reads; mini decoder layer writes RoPE-applied K and raw V through cache buffers | Still single-head row-major layout |
| Benchmark timing | ✅ | `lrrt_triton_mini_decoder_layer_benchmark` reports round-trip and burst-queued latency for the fixed-shape mini decoder layer | Needs methodology cleanup, per-stage timing, and model metadata reporting |

## Decoder Layer Shape

The current fixed-shape mini decoder layer is:

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

`triton_mini_decoder_layer` wires this chain together using Triton-generated
bundles and the shared `lrrt::executor::triton::mini::DecoderLayer` helper.
The example validates the final hidden state against a CPU reference. The
available `matvec` bundle stands in for small projections, but it is not yet a
complete GEMM or batched matmul implementation.

## Current Integration Baseline

The current multi-kernel baseline has moved beyond `triton_mini_attention` to
`triton_mini_decoder_layer`.

`triton_mini_attention` remains the smallest attention-only pipeline:

```text
Q/K/V-like vectors
  -> RoPE on Q and K
  -> KV cache update/read
  -> attention scores
  -> causal softmax
  -> value aggregation
  -> residual add
```

`triton_mini_decoder_layer` adds the missing transformer-layer shape around
that baseline:

```text
RMSNorm
  -> Q/K/V projection
  -> RoPE + KV cache update
  -> attention score + causal softmax + value aggregation
  -> output projection + residual
  -> RMSNorm
  -> gate/up/down MLP projection with SiLU multiply
  -> residual
```

This is still not a faithful Qwen model runtime. It is a controlled integration
path that exercises the same executor concerns:

- multiple bundle loads
- multiple device buffers
- repeated argument binding
- ordered launches on one queue
- CPU reference validation between or after stages
- predictable buffer lifetime

The benchmark target is `lrrt_triton_mini_decoder_layer_benchmark`, enabled by
`LRRT_BUILD_BENCHMARKS=ON` and `LRRT_BUILD_TRITON_BENCHMARKS=ON`. It reuses the
shared mini decoder layer executor helper and reports round-trip and
burst-queued latency for deterministic synthetic inputs.

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
than a graph runtime. The mini decoder layer helper lives in this executor
area because it coordinates a Triton bundle pipeline without adding model
semantics to the runtime core.

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

The current mini decoder layer and benchmark leave several gaps before a
Qwen-like path can be meaningful:

- **Multi-head attention**: the mini decoder layer is still effectively
  single-head. Qwen-style execution needs a `num_heads x head_dim` layout for
  Q/K/V, RoPE, cache, scores, probabilities, and value aggregation.
- **KV cache layout**: mini attention now stores RoPE-applied K and raw V in
  cache buffers, but the layout is still a single-head row-major
  `[max_tokens, head_dim]` buffer.
- **Weight loading**: the current examples and benchmark use deterministic
  synthetic weights. A Qwen-style benchmark needs an external weight-loading
  path, even if the first format is a simple raw binary plus metadata file.
- **FP16/BF16 end-to-end path**: Qwen-style inference should use lower
  precision inputs with FP32 accumulation where appropriate. Some operator
  bundles have lower precision coverage, but the mini decoder layer still runs
  the end-to-end path as FP32.
- **Shape metadata**: the current manifest describes launch ABI, not tensor
  shape semantics. The executor must provide shape policy outside the runtime
  core.
- **Benchmark methodology**: the benchmark reports end-to-end round-trip and
  burst-queued latency, but the measurement boundaries and any future per-stage
  timing need to be documented carefully before comparing performance across
  implementations.

## Suggested Milestones

### P0: Pipeline Documentation

- Keep this document as the high-level integration and benchmark sketch.
- Avoid adding graph semantics to the runtime core.

### P1: Thin Executor Prototype

- Status: complete for the current mini pipeline.
- `executor/triton` now provides reusable named bundle, named buffer, and
  manifest-driven launch helpers.
- `triton_mini_attention`, `triton_kv_cache`, and
  `triton_mini_decoder_layer` use executor-style helpers for multi-op paths.
- `lrrt::executor::triton::mini::DecoderLayer` is shared by the mini decoder
  layer example and benchmark.
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

- Status: complete for a fixed-shape, synthetic, single-head mini decoder
  layer.
- `triton_mini_decoder_layer` combines RMSNorm, projection, RoPE, causal
  softmax, value aggregation, MLP activation, and residual operations.
- It is an executor-style example, not a general model runtime.
- It validates the final hidden state against a CPU reference.

### P5: Small Qwen-Style Decode Benchmark

- Status: partially complete.
- `lrrt_triton_mini_decoder_layer_benchmark` runs one fixed-shape mini decoder
  layer through lrrt using Triton-generated bundles.
- It currently uses deterministic synthetic inputs and weights, not real Qwen
  weights.
- It measures round-trip latency and burst-queued latency for the whole layer.
- Next steps are multi-head support, external weight loading, lower precision
  end-to-end execution, and clearer benchmark methodology.
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
