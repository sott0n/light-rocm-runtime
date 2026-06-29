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
| RoPE | ✅ | `rope` for FP32 vectors up to the current specialization limits; applied to Q and K projection outputs in the decoder layer path | Multi-head currently reuses the single-head kernel once per head |
| Attention score computation | ✅ | `attention_score` computes FP32 Q x K dot products with scaling and is used in mini attention and mini decoder layer | Multi-head currently dispatches score computation per head |
| Causal softmax | ✅ | `causal_softmax` covers FP32 future-token masking with query offsets and is used in mini attention and mini decoder layer | Needs broader shape coverage for model-like cache lengths |
| Value aggregation | ✅ | `value_aggregation` computes FP32 weighted sums over V vectors and is used in mini attention and mini decoder layer | Multi-head currently dispatches aggregation per head |
| Residual add | ✅ | `vector_add` is used for attention residual and MLP residual in `triton_mini_decoder_layer` | Only FP32 path is wired into the decoder layer |
| Gated MLP activation | ✅ | `silu_mul` covers `SiLU(gate) * up` in FP32 and is wired into `triton_mini_mlp` and `triton_mini_decoder_layer` | Needs lower precision and larger shape coverage later |
| Output projection | ✅/❌ | `matvec` stands in for attention output projection and MLP down projection in the mini decoder layer | Needs larger/batched projection support and lower precision in the end-to-end path |
| KV cache update/read | ✅ | `kv_cache_update` and `kv_cache_read` cover FP32 row-major `[max_tokens, head_dim]` cache writes and indexed reads; mini decoder layer uses a `[heads, keys, head_dim]` cache layout by dispatching update/read-like operations per head | Needs a layout policy that can survive real model weight/cache integration |
| Benchmark timing | ✅ | `lrrt_triton_mini_decoder_layer_benchmark` reports CPU round-trip, CPU burst, and HSA GPU-event burst latency with dtype, cache layout, queueing mode, QKV dimension, and estimated dispatch count metadata | Needs per-stage timing for deeper performance analysis |
| Weight bundle loading | ✅/❌ | `executor/triton/mini_decoder_weights.hpp` can load a mini decoder FP32 raw binary plus manifest into executor-owned weight vectors | This is a prototype format, not a Qwen checkpoint loader; lower precision and real checkpoint conversion are still missing |

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

## Mini Decoder Weight Bundle

The first weight-loading prototype is intentionally narrow. It is a mini
decoder-layer executor format, not a runtime-core ABI and not a Qwen checkpoint
reader.

The bundle is:

```text
weights.json
weights.bin
```

`weights.bin` stores tightly packed FP32 tensor bytes. `weights.json` describes
the shape and tensor offsets:

```json
{
  "format": "lrrt.mini_decoder_weights",
  "version": 1,
  "dtype": "f32",
  "data": "weights.bin",
  "keys": 16,
  "hidden": 768,
  "heads": 1,
  "head_dim": 64,
  "intermediate": 2048,
  "tensors": [
    {"name": "attention_norm_weight", "offset": 0, "count": 768}
  ]
}
```

The actual manifest must include all mini decoder layer weight tensors:
`attention_norm_weight`, `mlp_norm_weight`, `q_weight`, `k_weight`, `v_weight`,
`out_weight`, `gate_weight`, `up_weight`, and `down_weight`.

The Triton benchmark build also provides
`lrrt_triton_mini_decoder_weight_bundle`, which emits the deterministic
synthetic weight pattern in this format:

```bash
mkdir -p /tmp/lrrt-mini-weights
./build-triton-bench/lrrt_triton_mini_decoder_weight_bundle \
  /tmp/lrrt-mini-weights/weights.json 16 768 1 64 2048
```

The mini decoder layer benchmark can then consume that bundle:

```bash
./build-triton-bench/lrrt_triton_mini_decoder_layer_benchmark \
  20 --weights /tmp/lrrt-mini-weights/weights.json --valid-keys 7
```

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
burst-queued latency for deterministic synthetic inputs. The benchmark reports
FP32 dtype, `[heads, keys, head_dim]` cache layout, ordered single-queue
launching, `QKVDim`, and an estimated kernel dispatch count per layer. That
dispatch count matters because multi-head attention currently loops over heads
and reuses single-head kernels instead of using fused multi-head kernels.

The benchmark timing modes are intentionally simple:

- **Round trip**: measure one `executor.run()` followed by `synchronize()` for
  each iteration. This includes CPU submission overhead, GPU execution, and the
  final synchronization wait for that layer.
- **Burst interval**: submit `executor.run()` repeatedly and call
  `synchronize()` once at the end. This measures the average interval for a
  burst of queued layer submissions, but it still preserves queue ordering and
  dependencies.

Both modes currently use CPU `steady_clock` around executor calls. Runtime GPU
event timing can be added later for per-stage or GPU-only measurements.

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

- **Multi-head performance**: the mini decoder layer now supports a
  `num_heads x head_dim` layout, but it does so by dispatching existing
  single-head kernels once per head. This is useful for correctness and layout
  validation, not for final performance.
- **KV cache layout policy**: the mini decoder layer uses a
  `[heads, keys, head_dim]` cache layout. This is explicit enough for the
  current benchmark, but real Qwen weight/cache integration may require a
  different layout or a documented adapter.
- **Weight loading**: the executor now has a small FP32 raw-binary plus
  manifest loader for the mini decoder layer. The examples and benchmark still
  use deterministic synthetic weights by default, and a Qwen-style benchmark
  still needs checkpoint conversion into this or a later weight-bundle format.
- **FP16/BF16 end-to-end path**: Qwen-style inference should use lower
  precision inputs with FP32 accumulation where appropriate. Some operator
  bundles have lower precision coverage, but the mini decoder layer still runs
  the end-to-end path as FP32.
- **Shape metadata**: the current manifest describes launch ABI, not tensor
  shape semantics. The executor must provide shape policy outside the runtime
  core.
- **Per-stage timing**: the benchmark reports CPU and GPU-event burst timing,
  but it does not yet break timing down by pipeline stage.

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

- Status: complete for a fixed-shape, synthetic, multi-head mini decoder layer.
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
- It measures round-trip latency and burst-queued latency for the whole layer
  and prints enough shape/queueing metadata to interpret those numbers.
- Next steps are external weight loading, lower precision end-to-end execution,
  fused or batched multi-head kernels, and per-stage timing.
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
