# Qwen with IREE and Triton

This guide runs a local Qwen2/Qwen2.5 Hugging Face checkpoint through the two
GPU executor paths in `light-rocm-runtime`:

- **IREE** provides the current end-to-end generation path: tokenizer input,
  prompt prefill, device-resident KV cache, greedy autoregressive decode, EOS
  handling, generated token IDs, and decoded text.
- **Triton** is the reference executor and benchmark path: converted checkpoint
  weights, the full decoder-layer stack, final logits validation, and detailed
  runtime timing.

Both paths are experimental, FP32-only Qwen 0.5B integrations rather than
general model servers.

## Current Capability

| Capability | IREE | Triton |
| --- | --- | --- |
| Local Qwen2/Qwen2.5 safetensors | Yes | Yes |
| Full 24-layer Qwen2.5-0.5B stack | Yes | Yes |
| Text and chat-template input | Yes | No; explicit token IDs |
| Prompt prefill | Yes, one token at a time | Fixed benchmark input |
| Device-resident KV cache | Yes | Yes |
| Multi-token autoregressive generation | Yes, greedy | No |
| Generated text output | Yes | No |
| Final logits validation | Yes | Yes |
| Hugging Face step-by-step reference check | Optional | No |
| Layer and runtime benchmark output | Basic progress timing | Yes |

## Prerequisites

- Linux with an AMD GPU supported by ROCm
- ROCr/HSA development files and an accessible `/dev/kfd`
- CMake, a C++17 compiler, Git, and `uv`
- A local Qwen2/Qwen2.5 Hugging Face checkpoint containing `config.json`,
  tokenizer files, and `.safetensors`

Check GPU access before building:

```sh
rocminfo | head
ls -l /dev/kfd
```

Set the checkpoint once for the commands below:

```sh
export QWEN_CHECKPOINT=/path/to/Qwen2.5-0.5B-Instruct
export QWEN_ARTIFACTS="${QWEN_ARTIFACTS:-$HOME/.cache/light-rocm-runtime/qwen}"
mkdir -p "$QWEN_ARTIFACTS"
```

The converter never downloads model weights. Point `QWEN_CHECKPOINT` at an
already-downloaded checkpoint directory.

## Shared Weight Bundle

Both executors consume the same converted layer-weight format:

```text
<bundle>/
  layer_0/weights.json
  layer_0/weights.bin
  ...
  layer_23/weights.json
  layer_23/weights.bin
  model_tail/weights.json
  model_tail/weights.bin
```

An IREE conversion includes the full token-embedding table required for
autoregressive feedback and can also be reused by Triton. Store persistent
bundles outside `/tmp` if they must survive a reboot.

## IREE

### 1. Build the IREE tools

Initialize the pinned IREE source and build `iree-compile` and
`iree-run-module`:

```sh
git submodule update --init third_party/iree
tools/build_iree_tools.sh --jobs 2
```

The tool build is intentionally separate from the lrrt build and may take a
while the first time.

### 2. Build the lrrt IREE adapter and Qwen runner

The commands below target the development GPU, `gfx1101`. Change
`LRRT_AMDGPU_TARGET` for a different GPU.

```sh
cmake -S . -B build-iree/adapter \
  -DLRRT_ENABLE_IREE_ADAPTER=ON \
  -DLRRT_IREE_ROOT=third_party/iree \
  -DLRRT_IREE_COMPILE_EXECUTABLE="$PWD/build-iree-tools/tools/iree-compile" \
  -DLRRT_IREE_RUN_MODULE_EXECUTABLE="$PWD/build-iree-tools/tools/iree-run-module" \
  -DLRRT_AMDGPU_TARGET=gfx1101

cmake --build build-iree/adapter -j2
```

Compile the max8, max16, max32, and max64 Qwen decode modules plus the model
tail. These tests write VMFB files under `build-iree-probe/`:

```sh
ctest --test-dir build-iree/adapter --output-on-failure \
  -R 'lrrt_iree_qwen_decode_layer_kv_cache_max(8|16|32|64)_vmfb_probe|lrrt_iree_qwen_decode1_tail_vmfb_probe'
```

### 3. Generate text

This command converts the checkpoint on the first run, creates an IREE decode
bundle, executes every checkpoint layer, and decodes two output tokens:

```sh
uv run --with-requirements tools/requirements.txt \
  python tools/run_qwen_e2e.py --iree \
  --checkpoint-dir "$QWEN_CHECKPOINT" \
  --bundle-dir "$QWEN_ARTIFACTS/weights" \
  --iree-decode-bundle-dir "$QWEN_ARTIFACTS/iree-max8" \
  --prompt "Hello" \
  --max-seq-len 8 \
  --max-new-tokens 2
```

The important final records are:

```text
step 1 top_token=<id> logit=<value> vocab=<size>
step 2 top_token=<id> logit=<value> vocab=<size>
generated_token_ids=[<id>,<id>]
stop_reason=max_new_tokens
iree_qwen summary generated_tokens=2 elapsed_ms=<time>
generated_text="<decoded text>"
```

For an instruction-tuned checkpoint, apply its chat template:

```sh
uv run --with-requirements tools/requirements.txt \
  python tools/run_qwen_e2e.py --iree \
  --checkpoint-dir "$QWEN_CHECKPOINT" \
  --bundle-dir "$QWEN_ARTIFACTS/weights" \
  --iree-decode-bundle-dir "$QWEN_ARTIFACTS/iree-max32" \
  --chat-system "You are helpful." \
  --chat-user "Say hello." \
  --max-seq-len 32 \
  --max-new-tokens 8
```

`--prompt`, `--chat-user`, and `--token-ids` are mutually exclusive.
`--chat-system` is valid only with `--chat-user`.

### 4. Reuse existing artifacts

After the first run, prevent accidental conversion or VMFB bundle replacement:

```sh
uv run --with-requirements tools/requirements.txt \
  python tools/run_qwen_e2e.py --iree \
  --checkpoint-dir "$QWEN_CHECKPOINT" \
  --bundle-dir "$QWEN_ARTIFACTS/weights" \
  --iree-decode-bundle-dir "$QWEN_ARTIFACTS/iree-max32" \
  --prompt "Continue: 1, 2, 3," \
  --max-seq-len 32 \
  --max-new-tokens 8 \
  --no-convert \
  --no-iree-bundle-write
```

The prompt-token count plus `--max-new-tokens` must fit within
`--max-seq-len`. The wrapper selects the smallest compiled max8/max16/max32/max64
VMFB capable of holding the requested sequence when it creates a new IREE
bundle. An existing bundle must already have sufficient capacity.

### 5. Diagnose progress

The normal output reports module initialization, weight loading, prefill,
decode, and total elapsed time. Add layer-level progress when a run appears to
stall:

```text
--iree-verbose-layers
```

This adds weight-loading and execution records such as:

```text
iree_qwen startup phase=weights layer=12/24 status=complete
step 3 layer 12/24 complete
```

### 6. Compare with Hugging Face

Add `--reference-check` to compare every generated top token and logit with a
local Hugging Face FP32 eager run:

```sh
uv run --with-requirements tools/requirements.txt \
  --with torch --with transformers \
  python tools/run_qwen_e2e.py --iree \
  --checkpoint-dir "$QWEN_CHECKPOINT" \
  --bundle-dir "$QWEN_ARTIFACTS/weights" \
  --iree-decode-bundle-dir "$QWEN_ARTIFACTS/iree-max8" \
  --prompt "Hello" \
  --max-seq-len 8 \
  --max-new-tokens 2 \
  --no-convert \
  --no-iree-bundle-write \
  --reference-check
```

This check loads the model in PyTorch as well as the IREE runner and therefore
needs additional host memory.

### 7. Development performance snapshot

The following numbers are a diagnostic baseline, not a performance guarantee.
They were measured on the development `gfx1101` GPU at commit `339175a`, using
the full 24-layer Qwen2.5-0.5B-Instruct model, an already-converted weight
bundle, and an already-compiled max64 VMFB bundle. Each row is the median of
three fresh-process runs:

```sh
build-iree/adapter/lrrt_iree_qwen_decode1_e2e \
  --max-new-tokens 2 \
  --max-seq-len 8 \
  --prompt-token-ids 23526,25,220 \
  --bundle /path/to/iree-max64 \
  /path/to/weights \
  24
```

| Phase | Median | Observed range |
| --- | ---: | ---: |
| Module initialization | 701.947 ms | 700.584–702.348 ms |
| Weight loading | 6456.584 ms | 6248.650–6481.618 ms |
| Three-token prefill host submission | 105.180 ms | 104.719–108.417 ms |
| First generated-token completion | 1439.319 ms | 1433.094–1649.590 ms |
| Next generated-token completion | 1438.157 ms | 1433.602–1630.368 ms |
| Prefill and two-token decode loop | 2940.477 ms | 2938.859–3350.780 ms |

The median measured process time for module initialization, weight loading, and
the decode loop was approximately 10.1 seconds. After the first generated
token, this experiment completed approximately 0.70 generated tokens/second.
The generated IDs were `[16,13]` in every run.

IREE work is submitted asynchronously. Consequently,
`phase=prefill status=submitted` measures host submission rather than completed
GPU prefill. The first generated-token completion synchronizes the dependent
work and includes outstanding prefill cost; it is not a standalone
time-to-first-token measurement. Use a GPU profiler and a longer run before
drawing optimization conclusions.

## Triton

### 1. Build the Triton benchmark path

Triton bundle generation uses the Python version selected by
`LRRT_TRITON_PYTHON`, which defaults to Python 3.13:

```sh
cmake -S . -B build-triton-bench \
  -DLRRT_BUILD_BENCHMARKS=ON \
  -DLRRT_BUILD_TRITON_BENCHMARKS=ON \
  -DLRRT_AMDGPU_TARGET=gfx1101

cmake --build build-triton-bench -j2
```

To select another Python installation, add:

```text
-DLRRT_TRITON_PYTHON=/path/to/python
```

### 2. Run the correctness-oriented E2E stack

`tools/run_qwen_e2e.py` defaults to Triton when `--iree` is absent. It converts
or reuses the weight bundle, runs the complete layer stack, executes final
RMSNorm and the language-model head, and checks that the logits are finite:

```sh
python3 tools/run_qwen_e2e.py \
  --checkpoint-dir "$QWEN_CHECKPOINT" \
  --bundle-dir "$QWEN_ARTIFACTS/weights" \
  --layers 24 \
  --keys 4 \
  --token-ids 0,1,2
```

The Triton path treats `--token-ids` as explicit embedding rows for its fixed
input rather than as a text prompt. It does not currently perform tokenizer
decode or autoregressive text generation.

If the bundle was already produced by the IREE command, add `--no-convert`:

```sh
python3 tools/run_qwen_e2e.py \
  --bundle-dir "$QWEN_ARTIFACTS/weights" \
  --layers 24 \
  --keys 4 \
  --token-ids 0,1,2 \
  --no-convert
```

### 3. Run the Triton benchmark

The benchmark wrapper reports decoder-stack timing and can reuse the same
bundle:

```sh
python3 tools/run_qwen_benchmark.py \
  --bundle-dir "$QWEN_ARTIFACTS/weights" \
  --layers 24 \
  --keys 4 \
  --token-ids 0,1,2 \
  --iterations 10 \
  --no-warmup \
  --no-convert
```

To convert a fresh bundle as part of the benchmark instead:

```sh
python3 tools/run_qwen_benchmark.py \
  --checkpoint-dir "$QWEN_CHECKPOINT" \
  --bundle-dir "$QWEN_ARTIFACTS/triton-weights" \
  --all-layers \
  --keys 4 \
  --token-ids 0,1,2 \
  --iterations 10 \
  --no-warmup
```

Useful benchmark options include:

- `--layer-sweep` for multiple layer counts
- `--trace-setup` for setup allocation/copy diagnostics
- `--trace-run` for per-run allocation/copy diagnostics
- `--sync-stack` for synchronized inter-layer correctness runs

## Current Limitations

The Qwen paths demonstrate correct integration, but are not production
inference servers:

- The implemented model shape is the FP32 Qwen2/Qwen2.5 0.5B configuration;
  BF16, FP16, quantized weights, and other model sizes are not supported.
- IREE runs one sequence at a time and submits prompt tokens sequentially.
  There is no batched prefill, continuous batching, or multi-request scheduler.
- IREE generation is greedy. Temperature, top-k, top-p, beam search, and
  user-selectable sampling seeds are not implemented.
- IREE KV-cache capacity is selected from statically compiled VMFB variants.
  The checked-in build flow currently provides capacities up to 64 tokens.
- Compiled VMFB and Triton artifacts are GPU-target-specific; rebuild them when
  changing the configured AMDGPU target.
- IREE progress times are coarse host-side diagnostics and include asynchronous
  queue effects. Triton has the more detailed benchmark path.
- Triton validates fixed token-ID inputs and logits, but does not currently
  provide tokenizer-driven autoregressive text generation.

## Artifact and Build Directory Summary

| Path | Contents | Reusable by |
| --- | --- | --- |
| `build-iree-tools/` | `iree-compile` and `iree-run-module` | IREE |
| `build-iree/adapter/` | lrrt IREE runner and tests | IREE |
| `build-iree-probe/` | compiled Qwen VMFB specializations | IREE |
| `build-triton-bench/` | Triton bundles and benchmark executables | Triton |
| `$QWEN_ARTIFACTS/weights/` | converted checkpoint weights | IREE and Triton |
| `$QWEN_ARTIFACTS/iree-max*/` | IREE manifest and copied VMFBs | IREE |

`$QWEN_ARTIFACTS` defaults to a user cache outside the repository. Do not
commit generated model weights or VMFB artifacts.

## Troubleshooting

### GPU is not visible

Run `rocminfo` and inspect permissions on `/dev/kfd` and `/dev/dri`. Container
or sandbox execution must expose those devices.

### No IREE Qwen decode VMFBs were found

Run the IREE VMFB probe CTest command from the build section. Also verify that
the configured `LRRT_AMDGPU_TARGET` matches the target suffix under
`build-iree-probe/`.

### The IREE sequence does not fit

Increase `--max-seq-len` and create a new IREE decode bundle, or point
`--iree-decode-bundle-dir` at one created with enough capacity. The currently
compiled maximum is 64 tokens.

### Text input reports a missing Python package

Run the wrapper through:

```sh
uv run --with-requirements tools/requirements.txt \
  python tools/run_qwen_e2e.py ...
```

### A converted bundle is incomplete or stale

Use `--force-convert` to rewrite checkpoint weights and
`--force-iree-bundle` to rewrite an IREE decode bundle. These operations replace
the selected generated artifacts, so verify their paths first.

## Further Reading

- [Mini inference pipeline](../../docs/mini-inference-pipeline.md)
- [IREE Qwen KV-cache ABI](../../docs/iree-qwen-kv-cache-abi.md)
- [IREE tool build](../../docs/iree-tool-build.md)
- [Triton compiler support](../../docs/compiler-support.md)
