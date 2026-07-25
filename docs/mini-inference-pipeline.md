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
| RoPE | ✅ | `rope` for FP32 vectors up to the current specialization limits; applied to Q and K projection outputs in the decoder layer path using `rope_theta` from converted Qwen config | Multi-head currently reuses the single-head kernel once per head |
| Attention score computation | ✅ | `attention_score` computes FP32 Q x K dot products with scaling and is used in mini attention and mini decoder layer | Multi-head currently dispatches score computation per head |
| Causal softmax | ✅ | `causal_softmax` covers FP32 future-token masking with query offsets and is used in mini attention and mini decoder layer | Needs broader shape coverage for model-like cache lengths |
| Value aggregation | ✅ | `value_aggregation` computes FP32 weighted sums over V vectors and is used in mini attention and mini decoder layer | Multi-head currently dispatches aggregation per head |
| Residual add | ✅ | `vector_add` is used for attention residual and MLP residual in `triton_mini_decoder_layer` | Only FP32 path is wired into the decoder layer |
| Gated MLP activation | ✅ | `silu_mul` covers `SiLU(gate) * up` in FP32 and is wired into `triton_mini_mlp` and `triton_mini_decoder_layer` | Needs lower precision and larger shape coverage later |
| Output projection | ✅/❌ | `matvec` stands in for attention output projection and MLP down projection in the mini decoder layer | Needs larger/batched projection support and lower precision in the end-to-end path |
| Token embedding | ✅/❌ | The Qwen converter can store selected `model.embed_tokens.weight` rows as initial hidden states for the mini stack benchmark | This is a converter-side prompt setup path, not a general device-side embedding lookup |
| Final norm | ✅ | `rmsnorm` is reused for `model.norm.weight` in the model tail path | Still FP32 and specialization driven |
| LM head logits | ✅/❌ | `matvec` can run `lm_head.weight` against the final normalized hidden state and produce vocabulary logits | This is a prototype one-vector logits path, not a tiled production GEMV |
| KV cache update/read | ✅ | `kv_cache_update` and `kv_cache_read` cover FP32 row-major `[max_tokens, head_dim]` cache writes and indexed reads; mini decoder layer uses a `[heads, keys, head_dim]` cache layout by dispatching update/read-like operations per head | Needs a layout policy that can survive real model weight/cache integration |
| Benchmark timing | ✅ | `lrrt_triton_mini_decoder_layer_benchmark` reports CPU round-trip, CPU burst, and HSA GPU-event burst latency with dtype, cache layout, queueing mode, QKV dimension, and estimated dispatch count metadata | Needs per-stage timing for deeper performance analysis |
| Weight bundle loading | ✅/❌ | `executor/qwen/weight_bundle.hpp` can load decoder layer and model tail FP32 raw binaries plus manifests; `tools/convert_qwen_layer.py` can convert local Qwen checkpoint tensors into those bundles | This is still a prototype FP32 format and does not cover full model/runtime semantics |

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
Qwen weight manifests are loaded through the framework-neutral
`lrrt::executor::qwen` bundle helpers so Triton and IREE executors can share
the same converted checkpoint format. The example validates the final hidden
state against a CPU reference. The
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
  "version": 2,
  "dtype": "f32",
  "data": "weights.bin",
  "keys": 16,
  "hidden": 768,
  "heads": 1,
  "kv_heads": 1,
  "head_dim": 64,
  "intermediate": 2048,
  "rope_theta": 10000.0,
  "tensors": [
    {"name": "attention_norm_weight", "offset": 0, "count": 768}
  ]
}
```

The actual manifest must include all mini decoder layer weight tensors:
`attention_norm_weight`, `mlp_norm_weight`, `q_weight`, `k_weight`, `v_weight`,
`q_bias`, `k_bias`, `v_bias`, `out_weight`, `gate_weight`, `up_weight`, and
`down_weight`. Version 2 added the Q/K/V projection biases; the converter
writes zero vectors when a checkpoint omits them.

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

## Qwen Tensor Mapping

The first checkpoint converter should target one decoder layer from a
Qwen2/Qwen2.5-style Hugging Face checkpoint. It should emit the current mini
decoder weight bundle format for one selected layer, converting tensors to
FP32 row-major bytes for the prototype path.

The prototype converter is `tools/convert_qwen_layer.py`. It reads a local
Hugging Face checkpoint directory, loads the required safetensors shards, and
writes one or more mini decoder layer bundles:

```bash
uv pip install -r tools/requirements.txt
python3 tools/convert_qwen_layer.py \
  --checkpoint-dir /path/to/qwen-checkpoint \
  --layer 0 \
  --keys 16 \
  --output /tmp/lrrt-qwen-layer0/weights.json
```

For a consecutive layer range, pass `--layer-count` and treat `--output` as a
directory:

```bash
python3 tools/convert_qwen_layer.py \
  --checkpoint-dir /path/to/qwen-checkpoint \
  --layer 0 \
  --layer-count 4 \
  --keys 16 \
  --output /tmp/lrrt-qwen-layers
```

That writes one independent bundle per layer:

```text
/tmp/lrrt-qwen-layers/
  layer_0/weights.json
  layer_0/weights.bin
  layer_1/weights.json
  layer_1/weights.bin
  ...
```

Pass `--bundle-directory` when a single selected layer should still be written
as an E2E-capable directory bundle with `layer_<index>/weights.json` plus
`model_tail/weights.json`.

The converter does not download model weights. The caller must provide a local
checkpoint directory containing `config.json` and `.safetensors` files. This is
still a narrow bridge into the mini decoder benchmark format, not a general
model loader.

`tools/run_qwen_benchmark.py` wraps this prototype E2E flow. It reuses an
existing bundle directory when all requested layer manifests and the model tail
manifest are present; otherwise it invokes the converter before running
`lrrt_triton_mini_decoder_layer_benchmark`:

```bash
python3 tools/run_qwen_benchmark.py \
  --checkpoint-dir /path/to/qwen-checkpoint \
  --bundle-dir /tmp/lrrt-qwen-full \
  --all-layers \
  --keys 4 \
  --token-ids 0,1,2 \
  --iterations 1 \
  --no-warmup
```

Use `--no-convert` to require an already-converted bundle directory, or
`--force-convert` to rewrite the bundle before running the benchmark. The
wrapper uses `uv run --with-requirements tools/requirements.txt` for conversion
when `uv` is available, and falls back to the current Python interpreter when
`uv` is not installed.

The current wrapper starts at layer 0 because the benchmark consumes
`layer_0..layer_N` directories.

`tools/run_qwen_e2e.py` is the full E2E check entry point. It wraps the same
converter and benchmark runner, but always requires the model tail and passes
`--e2e-check` and `--sync-stack` to make logits production, finite output
values, and synchronized layer handoff part of the success condition:

```bash
python3 tools/run_qwen_e2e.py \
  --checkpoint-dir /path/to/qwen-checkpoint \
  --bundle-dir /tmp/lrrt-qwen-e2e \
  --keys 4 \
  --token-ids 0,1,2
```

By default the E2E wrapper reads `num_hidden_layers` from the checkpoint config
and runs every layer. The path is:

```text
local Qwen checkpoint
  -> mini decoder layer bundles + model tail bundle
  -> lrrt Triton decoder stack
  -> synchronized device-to-device layer handoff
  -> final RMSNorm
  -> lm_head logits
  -> finite-logit/top-logit validation
```

This is still the mini FP32 runtime path rather than a tokenizer-integrated
chat loop. The input prompt is represented by explicit token ids whose
embedding rows initialize the first decoder layer.
The benchmark path can still use queued async handoff, but the E2E wrapper uses
the synchronized path so correctness checks do not depend on cross-queue timing.

The intended per-layer mapping is:

| Mini decoder tensor | Qwen checkpoint tensor | Expected shape in mini bundle | Conversion note |
| --- | --- | --- | --- |
| `attention_norm_weight` | `model.layers.{layer}.input_layernorm.weight` | `[hidden]` | Direct copy |
| `mlp_norm_weight` | `model.layers.{layer}.post_attention_layernorm.weight` | `[hidden]` | Direct copy |
| `q_weight` | `model.layers.{layer}.self_attn.q_proj.weight` | `[heads * head_dim, hidden]` | Direct copy when `q_proj` is stored as PyTorch linear `[out, in]` |
| `k_weight` | `model.layers.{layer}.self_attn.k_proj.weight` | `[kv_heads * head_dim, hidden]` | Direct copy for grouped-query attention |
| `v_weight` | `model.layers.{layer}.self_attn.v_proj.weight` | `[kv_heads * head_dim, hidden]` | Direct copy for grouped-query attention |
| `q_bias` | `model.layers.{layer}.self_attn.q_proj.bias` | `[heads * head_dim]` | Direct copy, or zeros when absent |
| `k_bias` | `model.layers.{layer}.self_attn.k_proj.bias` | `[kv_heads * head_dim]` | Direct copy, or zeros when absent |
| `v_bias` | `model.layers.{layer}.self_attn.v_proj.bias` | `[kv_heads * head_dim]` | Direct copy, or zeros when absent |
| `out_weight` | `model.layers.{layer}.self_attn.o_proj.weight` | `[hidden, heads * head_dim]` | Direct copy for the current matvec layout |
| `gate_weight` | `model.layers.{layer}.mlp.gate_proj.weight` | `[intermediate, hidden]` | Direct copy |
| `up_weight` | `model.layers.{layer}.mlp.up_proj.weight` | `[intermediate, hidden]` | Direct copy |
| `down_weight` | `model.layers.{layer}.mlp.down_proj.weight` | `[hidden, intermediate]` | Direct copy |

The current mini decoder layer intentionally does not consume these
checkpoint-level tensors yet:

| Qwen tensor or config field | Current handling |
| --- | --- |
| `model.embed_tokens.weight` | The converter stores selected token rows in `model_tail/weights.json` for the benchmark input |
| `model.norm.weight` | The converter stores it in `model_tail/weights.json` and the benchmark runs final RMSNorm |
| `lm_head.weight` | The converter stores it in `model_tail/weights.json` and the benchmark runs an FP32 matvec to produce logits |
| RoPE parameters such as `rope_theta` | The converter stores `rope_theta` in each layer bundle and the benchmark uses it to generate `cos` and `sin` tables |
| Output projection bias | Not supported; `o_proj.bias` is rejected when present |

The converter should derive bundle shape fields from model config and tensor
shapes:

| Bundle field | Source |
| --- | --- |
| `hidden` | `hidden_size` |
| `heads` | `num_attention_heads` |
| `kv_heads` | `num_key_value_heads`, defaulting to `heads` when absent |
| `head_dim` | `hidden_size / num_attention_heads`, unless the config exposes an explicit head dimension |
| `intermediate` | `intermediate_size` |
| `keys` | Benchmark/cache length chosen by the converter or command-line option |

Grouped-query attention is represented explicitly. The mini decoder executor
keeps separate Q-head and KV-head counts, stores the KV cache as
`[kv_heads, keys, head_dim]`, and maps each Q head to a KV head by contiguous
groups. The converter therefore keeps Q projection rows at
`heads * head_dim` while keeping K/V projection rows at
`kv_heads * head_dim`. It rejects only unsupported shapes where
`num_attention_heads` is not a multiple of `num_key_value_heads`.

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
FP32 dtype, `[kv_heads, keys, head_dim]` cache layout, ordered single-queue
launching, `QDim`/`KVDim`, and an estimated kernel dispatch count per layer. That
dispatch count matters because multi-head attention currently loops over heads
and reuses single-head kernels instead of using fused multi-head kernels.

When passed `--weights-dir <dir> --layers <count>`, the benchmark loads
`layer_0/weights.json` through `layer_<count - 1>/weights.json` and runs a
small decoder stack loop. This path copies each layer's output hidden state
directly into the next layer's device input buffer through the runtime
async device-to-device copy API. It records source-layer completion events,
queues copy events, and launches the next layer with explicit dependencies
instead of synchronizing at each handoff. It is useful for verifying
multi-layer bundle conversion and executor sequencing, but it is still a
prototype executor path rather than a general graph scheduler.

The benchmark timing modes are intentionally simple:

- **Round trip**: measure one `executor.run()` followed by `synchronize()` for
  each iteration. This includes CPU submission overhead, GPU execution, and the
  final synchronization wait for that layer.
- **Burst interval**: submit `executor.run()` repeatedly and call
  `synchronize()` once at the end. This measures the average interval for a
  burst of queued layer submissions, but it still preserves queue ordering and
  dependencies.
- **Stack round trip**: for `--weights-dir`, run each layer in order and pass
  hidden states between layers through queued async runtime device-to-device
  copies. The stack path synchronizes only after the queued layer/copy chain is
  submitted, so the measured time covers submission plus final completion wait.
- **Stack GPU burst**: for `--weights-dir`, place HSA event markers around one
  queued stack chain. The start marker is recorded on the first layer queue and
  the end marker is recorded on the final layer or model-tail queue, so this
  reports GPU-side elapsed time across the queued handoff path.

The benchmark still does not break timing down by individual stage or operator.

The IREE adapter path exposes a matching static-shape `qwen_decode_step` VMFB
export for the same one-token decode contract:

```text
input/query + K/V cache
  -> RoPE
  -> K/V cache update
  -> attention score + softmax + value aggregation
  -> FFN
  -> updated K/V cache + hidden result
```

`lrrt_iree_qwen_decode_e2e_smoke` calls that export twice through
`VmfbRunner`, passing the first step's K/V cache `buffer_view`s directly into
the second step. This is the current IREE+lrrt E2E skeleton: it proves the
VMFB can run through the lrrt HAL adapter and that decode-step state can stay
device-resident between token steps. It still uses deterministic stub tensors
and fixed `2x2`/three-token cache shapes; connecting Qwen checkpoint weights
and larger model dimensions remains future work.

The real-weight Qwen runner now has two decode paths. The fixed milestone path
uses separate one-, two-, and three-token VMFB exports:

```text
token 0 embedding
  -> 24x qwen_decode1_layer
  -> per-layer K/V outputs stay as IREE buffer_views

logits argmax token embedding
  -> 24x qwen_decode2_layer_kv_cache
  -> each layer consumes the matching token-0 K/V buffer_view
  -> updated 2-token K/V cache + hidden result
  -> qwen_decode1_tail logits

logits argmax token embedding
  -> 24x qwen_decode3_layer_kv_cache
  -> each layer consumes the matching 2-token K/V buffer_view
  -> updated 3-token K/V cache + hidden result
  -> qwen_decode1_tail logits
```

This path is exposed by `lrrt_iree_qwen_decode1_e2e --max-new-tokens 3` and uses
`tools/iree_qwen_decode1_layer.mlir`,
`tools/iree_qwen_decode2_layer_kv_cache.mlir`,
`tools/iree_qwen_decode3_layer_kv_cache.mlir`, and
`tools/iree_qwen_decode1_tail.mlir`. It runs the full Qwen 0.5B layer count and
checkpoint-derived weights through the lrrt IREE HAL adapter while preserving
per-layer K/V cache ownership in VM buffer views. The decode2 MLIR computes the
two-token attention path with grouped-query attention, QK scores, `1/sqrt(64)`
scaling, numerically stable softmax over the visible two-token cache, and V
aggregation before the output projection. The decode3 MLIR repeats the same
fixed-shape pattern with a `2x128xf32` input cache and a `3x128xf32` output
cache. The current shape is still fixed to a single current token with a
statically visible cache length; causal masking is implicit because each entry
point only materializes visible cache slots.

The preferred path uses static-shape
`qwen_decode_layer_kv_cache_max<N>.mlir` specializations generated by CMake
from `tools/iree_qwen_decode_layer_kv_cache.mlir.in`. The current supported
cache capacities are `8`, `16`, `32`, and `64`, each available with FP32,
FP16, or BF16 decoder/KV-cache tensors. The runner-facing contract is a
capacity- and precision-bearing bundle rather than an open-ended list of
sequence-length-specific entry points. Reduced-precision tails extend the
final hidden state to FP32 before final RMSNorm and the language-model head.
Final-norm and language-model-head weights are transposed and uploaded once at
startup, then retained for all generated tokens.

```text
token embedding
  -> initialize per-layer max_cache_tokens x 128 K/V cache buffer_views once

for step in 0..max_new_tokens:
  for each of 24 decoder layers:
    qwen_decode_layer_kv_cache_maxN(hidden,
                                    key_cache[layer],
                                    value_cache[layer],
                                    position=step,
                                    Q/K/V/O and MLP weights,
                                    Q/K/V biases,
                                    rope_theta)
    -> updated max_cache_tokens-token K/V cache + hidden result
  -> qwen_decode1_tail logits
  -> host argmax token embedding for the next step
```

This path is exposed as:

```text
lrrt_iree_qwen_decode1_e2e --max-new-tokens N --max-seq-len S \
  --bundle <bundle-dir> <weights-dir> [layers]
```

where `<bundle-dir>/manifest.json` records the layer VMFB, tail VMFB, export
names, the user-facing sequence capacity, and the physical cache tensor shape.
Create that bundle from compiled VMFB artifacts with:

```text
tools/write_iree_qwen_decode_bundle.py \
  --target gfx1101 \
  --layer-vmfb <qwen_decode_layer_kv_cache_max*.vmfb> \
  --tail-vmfb <qwen_decode1_tail.vmfb> \
  --out-dir <bundle-dir> \
  --sequence-capacity <max_seq_len> \
  --max-cache-tokens <8|16|32|64>
```

The direct VMFB form is still available for manual experiments:

```text
lrrt_iree_qwen_decode1_e2e --max-new-tokens N --max-cache-tokens <8|16|32|64> \
  <qwen_decode_layer_kv_cache_max*.vmfb> <tail.vmfb> <weights-dir> [layers]
```

The higher-level E2E wrapper can prepare both sides of this path. In IREE mode,
`tools/run_qwen_e2e.py` first reuses or converts the Qwen mini weight bundle,
then reuses or creates the IREE decode bundle, then invokes
`lrrt_iree_qwen_decode1_e2e` through the bundle form:

```text
python3 tools/run_qwen_e2e.py --iree \
  --checkpoint-dir /path/to/qwen-checkpoint \
  --bundle-dir /tmp/lrrt-qwen-full \
  --iree-decode-bundle-dir /tmp/lrrt-iree-qwen-decode-bundle \
  --max-seq-len 16 \
  --max-new-tokens 4
```

For tokenizer-backed input and output, pass a text prompt and a directory
containing a Hugging Face `tokenizer.json`:

```text
python3 tools/run_qwen_e2e.py --iree \
  --checkpoint-dir /path/to/qwen-checkpoint \
  --prompt "Hello" \
  --max-seq-len 16 \
  --max-new-tokens 4
```

With `--prompt`, `--checkpoint-dir` is also the default `--tokenizer-dir`.
Pass `--tokenizer-dir` explicitly when the tokenizer and checkpoint are in
different directories, or with `--token-ids` to decode only the generated
output. The wrapper prints the encoded input as `prompt_token_ids=[...]` and,
after the native runner's `generated_token_ids=[...]` summary, prints the
decoded output as `generated_text="..."`. `--prompt` and `--token-ids` are
mutually exclusive. Tokenization requires the `tokenizers` package from
`tools/requirements.txt`; tokenizer and text semantics remain in the Python
wrapper rather than the runtime or IREE HAL adapter.

For instruction-tuned Qwen checkpoints, use `--chat-user` instead of manually
constructing ChatML. The wrapper reads `chat_template` from
`tokenizer_config.json`, renders the system and user messages with an assistant
generation prompt, and prints the rendered value as `chat_prompt=...` before
tokenization:

```text
python3 tools/run_qwen_e2e.py --iree \
  --checkpoint-dir /path/to/Qwen2.5-0.5B-Instruct \
  --chat-system "You are helpful." \
  --chat-user "Say hello." \
  --max-seq-len 32 \
  --max-new-tokens 8
```

`--chat-user`, `--prompt`, and `--token-ids` are mutually exclusive.
`--chat-system` is optional and valid only with `--chat-user`; when it is
omitted, the checkpoint template controls its default system message. Chat
template rendering additionally requires `jinja2` from
`tools/requirements.txt`.

Generation stops early when the selected token matches `--eos-token-id`.
The wrapper infers that id from `generation_config.json`, `config.json`, or
`tokenizer_config.json` when those files are available, and passes an explicit
value through unchanged. The native runner always prints
`stop_reason=eos_token` or `stop_reason=max_new_tokens` after
`generated_token_ids=[...]`.

To check every generated token and its top logit against the local Hugging Face
Qwen implementation, add `--reference-check`:

```text
python3 tools/run_qwen_e2e.py --iree \
  --checkpoint-dir /path/to/qwen-checkpoint \
  --prompt "Hello" \
  --max-seq-len 8 \
  --max-new-tokens 2 \
  --reference-check
```

This optional check loads the checkpoint with `transformers` in FP32 eager
mode. Before inference it verifies representative prompt-embedding and layer-0
weights against the converted bundle so a different Qwen variant cannot produce
a misleading comparison. For each decode step it appends the IREE-generated
token to the Hugging Face input, requires the next top token to match exactly,
and checks the top logit with `--reference-logit-atol` (default `0.05`). It
prints one `reference_step=...` line per generated token and
`reference_check=passed steps=...` on success. `numpy`, `torch`, and
`transformers` are only required when this check is requested.

For a tokenizer-backed text generation regression, pin both the token IDs and
decoded text. For the official Qwen2.5-0.5B-Instruct checkpoint, the two-token
greedy result for the raw prompt `Hello` is:

```text
python3 tools/run_qwen_e2e.py --iree \
  --checkpoint-dir /path/to/qwen-checkpoint \
  --prompt "Hello" \
  --max-seq-len 8 \
  --max-new-tokens 2 \
  --expect-generated-token-ids 271,40 \
  --expect-generated-text $'\n\nI'
```

The command fails if either result changes and prints
`text_generation_regression=passed` when both match. Pinning token IDs makes the
regression independent of ambiguity in how adjacent tokenizer pieces are
rendered; pinning text also covers the complete tokenize, IREE generation, and
decode path.

Longer chat generation can pin the stop reason as well. For the official
Qwen2.5-0.5B-Instruct checkpoint, this 20-token chat prompt plus eight generated
tokens exercises the max32 VMFB and all eight autoregressive feedback steps:

```text
python3 tools/run_qwen_e2e.py --iree \
  --checkpoint-dir /path/to/Qwen2.5-0.5B-Instruct \
  --chat-system "You are helpful." \
  --chat-user "Say hello." \
  --max-seq-len 32 \
  --max-new-tokens 8 \
  --reference-check \
  --expect-generated-token-ids 9707,0,2585,646,358,7789,498,3351 \
  --expect-generated-text "Hello! How can I assist you today" \
  --expect-stop-reason max_new_tokens
```

To cover early EOS termination with the same official model and max32 bundle:

```text
python3 tools/run_qwen_e2e.py --iree \
  --checkpoint-dir /path/to/Qwen2.5-0.5B-Instruct \
  --chat-system "Reply only with OK." \
  --chat-user "Ready?" \
  --max-seq-len 32 \
  --max-new-tokens 8 \
  --expect-generated-token-ids 3925,151645 \
  --expect-generated-text "OK" \
  --expect-stop-reason eos_token
```

`--expect-stop-reason` accepts `max_new_tokens` or `eos_token`. It makes a
regression fail if the native runner stops for a different reason, including
when the token IDs and decoded text would otherwise pass.

To cross the 32-token boundary, the following 11-token prompt generates 40
tokens with the max64 specialization and checks every step against Hugging
Face FP32:

```text
python3 tools/run_qwen_e2e.py --iree \
  --checkpoint-dir /path/to/Qwen2.5-0.5B-Instruct \
  --prompt "Continue: 1, 2, 3," \
  --max-seq-len 64 \
  --max-new-tokens 40 \
  --reference-check \
  --expect-generated-token-ids 220,19,11,220,20,11,220,21,11,220,22,11,220,23,11,220,24,11,220,16,15,11,220,16,16,11,220,16,17,11,220,16,18,11,220,16,19,11,220,16 \
  --expect-generated-text " 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1" \
  --expect-stop-reason max_new_tokens
```

The 40 top tokens match exactly; the maximum top-logit absolute difference is
`5.9552002e-05`.

By default, the wrapper discovers available layer VMFB capacities and the tail
VMFB from the standard probe output paths under `build-iree-probe/` for
`--iree-target`:
`qwen_decode_layer_kv_cache_max<N>/qwen_decode_layer_kv_cache_max<N>_<target>.vmfb`
and `qwen_decode1_tail/qwen_decode1_tail_<target>.vmfb`. Use
`--iree-layer-vmfb` or `--iree-tail-vmfb` to override those paths explicitly.
The wrapper treats `--max-seq-len` as the user-facing inference limit and picks
the smallest discovered cache-capacity VMFB that can hold it; for example, if a
`max16` VMFB exists, `--max-seq-len 10` selects that VMFB and writes
`sequence_capacity: 10` and `max_cache_tokens: 16` to the bundle manifest. The
runner then validates `max_new_tokens <= max_seq_len <= sequence_capacity`.
`max_cache_tokens` remains the current static VMFB tensor extent and should not
be treated as the user-facing decode length. When an existing IREE decode bundle
is reused, the wrapper reads its manifest and rejects `--max-seq-len` values
larger than the recorded `sequence_capacity` before launching the runner.

Use `--no-convert` to require an existing weight bundle and
`--no-iree-bundle-write` to require an existing IREE decode bundle. Use
`--force-convert` or `--force-iree-bundle` when those artifacts should be
rewritten explicitly. When conversion is enabled, the IREE wrapper asks the
converter for a directory bundle and full token embeddings so a one-layer
decode run still has both `layer_0` and `model_tail` artifacts.

`--max-new-tokens` is the output length for this E2E check. It can be any
positive value up to `--max-seq-len`; values beyond the selected sequence
capacity are rejected before launching the runner. The runner keeps one
device-resident K/V cache pair per layer, passes the previous step's buffer
views directly into the next step, and only uses host readback for the current
`argmax(logits)` feedback. After generation it prints the generated ids,
excluding the prompt, as `generated_token_ids=[...]`. Real multi-step runs
require the tail bundle to include the selected embeddings; direct converter
calls should pass `--full-token-embeddings` to produce that format. The cache
ABI and the next longer-context direction are tracked in
`docs/iree-qwen-kv-cache-abi.md`.
The max-cache MLIR adds the checkpoint Q/K/V projection biases, then applies
Qwen's split-half `rotate_half` RoPE convention to Q and K using the current
position and each layer bundle's `rope_theta`.
The current IREE E2E path has been validated with the full 24-layer Qwen stack
through `--max-new-tokens 40 --max-seq-len 64`, crossing the former max32
boundary while matching all generated tokens against Hugging Face FP32.

The current `qwen_decode_step` input contract is fixed as follows:

| Index | Input | Static shape | Current meaning | Future Qwen source |
| --- | --- | --- | --- | --- |
| 0 | `input` | `2x2xf32` | Residual hidden state entering the decode step | token embedding or previous layer output |
| 1 | `query` | `2x2xf32` | Precomputed Q-like tensor consumed by RoPE and attention | output of `q_proj(input_norm)` |
| 2 | `old_key_cache_transposed` | `2x3xf32` | Previous K cache in `[head_dim, cache_tokens]` layout | long-lived per-layer K cache |
| 3 | `new_key` | `2xf32` | New K vector for the current token before RoPE | output of `k_proj(input_norm)` |
| 4 | `old_value_cache` | `3x2xf32` | Previous V cache in `[cache_tokens, head_dim]` layout | long-lived per-layer V cache |
| 5 | `new_value` | `2xf32` | New V vector for the current token | output of `v_proj(input_norm)` |
| 6 | `cos` | `2xf32` | RoPE cosine values for the current position | generated from Qwen RoPE config |
| 7 | `sin` | `2xf32` | RoPE sine values for the current position | generated from Qwen RoPE config |
| 8 | `w_gate` | `2x2xf32` | Stub gate projection weight for the FFN path | `mlp.gate_proj.weight` |
| 9 | `w_up` | `2x2xf32` | Stub up projection weight for the FFN path | `mlp.up_proj.weight` |
| 10 | `w_down` | `2x2xf32` | Stub down projection weight for the FFN path | `mlp.down_proj.weight` |

The output contract is:

| Index | Output | Static shape | Meaning |
| --- | --- | --- | --- |
| 0 | `key_cache_transposed` | `2x3xf32` | Updated K cache for the next token step |
| 1 | `value_cache` | `3x2xf32` | Updated V cache for the next token step |
| 2 | `hidden` | `2x2xf32` | Decode-step hidden result after attention and FFN |

Only the FFN weights are explicit inputs in this first IREE skeleton. Q/K/V and
attention-output projection weights are not connected yet; `query`, `new_key`,
and `new_value` are supplied as already-projected tensors. That keeps the
current VMFB small while leaving the input indices and cache ownership model
stable for the next real-weight integration step.

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
  `[kv_heads, keys, head_dim]` cache layout. This is explicit enough for the
  current benchmark, but real Qwen weight/cache integration may require a
  different layout or a documented adapter.
- **Weight loading**: the executor now has a small FP32 raw-binary plus
  manifest loader for the mini decoder layer. The examples and benchmark still
  use deterministic synthetic weights by default. A prototype Qwen converter
  can now write one local Hugging Face Qwen layer or a consecutive range of
  layers into this format, but it is FP32-only and supports grouped-query
  attention when the attention head count is an integer multiple of the KV head
  count.
- **Layer handoff**: the multi-layer benchmark path now keeps hidden-state
  handoff on the device through queued async runtime copy events. Each next
  layer launch depends on the copy event for the matching token position, so
  the host no longer synchronizes between per-token layer steps.
- **FP16/BF16 optimization**: the IREE Qwen path can now run decoder layers,
  hidden states, and KV caches in FP16 or BF16 with an FP32 model tail. The
  current batch-1 lowering does not materially outperform FP32, and the Triton
  mini decoder path remains FP32-only. Lower-precision accumulation policy,
  native 16-bit weight storage, and tuned/fused kernels remain future work.
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
- It can use deterministic synthetic inputs and weights, or load a local Qwen
  checkpoint converted into layer bundles plus a `model_tail` bundle.
- It can also load a directory of per-layer Qwen weight bundles and run a
  queued runtime-copy decoder stack loop for early multi-layer validation. When
  `model_tail/weights.json` is present, it initializes the first layer from one
  or more selected token embeddings and runs final RMSNorm plus lm_head to
  produce logits.
- It measures round-trip latency and burst-queued latency for the whole layer
  and prints enough shape/queueing metadata to interpret those numbers.
- Next steps are lower precision end-to-end execution, fused or batched
  multi-head kernels, broader token sequence handling, and per-stage timing.
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
