# LRRT Stateful Graph

## Objective

LRRT Stateful Graph is an optional execution layer above the small LRRT core.
It targets workloads with repeated, data-dependent GPU control flow, beginning
with autoregressive Qwen decode. The layer keeps mutable execution state and
dependency counters on the device so that one host launch can execute multiple
conditional graph steps.

This extends the project stack without adding model semantics to the core
dispatch API:

```text
compiler-generated stage kernels
  -> Stateful Graph: task ABI, device state, conditional scheduling
  -> LRRT core: memory, code objects, queues, dispatch, events
  -> ROCr or light-rocr backend
```

The intended performance claim is not merely that LRRT has a thinner graph
API. The target is a device-resident execution model that HIP Graph does not
currently expose on AMD GPUs: a graph loop whose predicate is produced and
consumed on the GPU without a host round trip.

## Prototype v3

The prototype defines a 64-byte device `State`, a 32-byte `Task`, and a 32-byte
`Program` descriptor. Each Program selects a DAG of up to 64 tasks from a
shared task table and names its normal and conditional successors. Tasks can
publish a branch value in device state, allowing a Program transition to skip
or enter a subgraph without host intervention. A fixed set of persistent
workgroups repeatedly:

1. scan for a task whose dependencies are complete;
2. atomically claim and execute that task;
3. publish completion so dependent tasks become ready;
4. let the final completing worker evaluate device-side branch and loop
   predicates; and
5. atomically publish the selected Program phase or return.

Claimed and completed masks remain pairs of 32-bit atomic words so the 32-task
fast path stays intact. Each bounded execution phase owns a separate 20-byte
`MaskState`, including a one-word atomic completion counter. The counter elects
exactly one transition owner without requiring a coherent snapshot across both
halves of a 64-task mask. Mask storage is therefore proportional to the maximum
number of Program transitions, not model weights. Program index and phase
generation are packed into one atomic 32-bit word. The completing worker
publishes them together, preventing workers from combining a new Program with
an old mask generation. Since phase mask storage is never reused during a
launch, stale workers can only touch an obsolete phase and no grid-wide
transition barrier is required. The benchmark exercises a 64-task dependency
chain through bit 63 before running the timed Qwen graph.

The accompanying benchmark compares one resident launch with an equivalent
sequence of dependency-ordered host launches using kernels from the same
HSACO. Worker count and task count are independent: a small resident worker
pool can execute a larger task DAG without a device-wide barrier.
The resident kernel validates task bounds and rejects cyclic dependency graphs
before entering the scheduling loop.

The worker pool must still satisfy resource-aware admission: every persistent
worker must be concurrently resident. The benchmark limits the pool to 20
small workgroups, while tasks are scheduled onto that pool dynamically.

The prototype also uses a fixed 256-thread worker width from the Stateful Graph
ABI. LRRT launches raw kernel argument buffers and deliberately does not
synthesize HIP-specific hidden `blockDim` or `gridDim` arguments, so resident
kernels must not depend on those fields.

## Qwen executor specialization

The scheduler ABI carries an opaque execution-context pointer, keeping model
buffer semantics out of the generic graph state and task descriptors. All
Qwen-specific shapes, opcodes, contexts, kernels, and benchmark wiring live
under `executor/qwen` or Qwen-named benchmark sources; the generic
`executor/stateful_graph` layer has no Qwen dependency. The decoder
specialization uses the
Qwen2.5-0.5B dimensions `hidden=896`, `heads=14`, `kv_heads=2`, `head_dim=64`,
and `intermediate=4864`.

The persistent dependency scheduler is implemented by the reusable device
template in `executor/stateful_graph/device_scheduler.hpp`. A specialization
supplies only context validation, task validation, and task execution
callbacks. The Qwen kernel therefore owns its fused model operations and
shared-memory scratch while the generic scheduler owns task claiming,
dependency masks, Program transitions, conditional branches, and termination.

The corresponding host runtime API is
`executor/stateful_graph/resident_executor.hpp`. `ResidentExecutor` validates
and uploads the generic task and Program tables, owns the device State and
per-phase masks, binds an opaque workload context, initializes each bounded
run, and dispatches the workload-specialized resident kernel. The Qwen
benchmark uses this API rather than managing scheduler buffers and constructing
`ResidentArgs` itself.

For fixed workloads, `compile_resident_plan()` converts a topologically ordered
task DAG into contiguous dependency-independent stages. The reusable compiled
scheduler in `executor/stateful_graph/compiled_scheduler.hpp` assigns those
tasks directly to persistent workers, removing runtime task scans and atomic
claims. All resident workers rendezvous at each stage boundary; the last arrival
performs the same device-side Program transition, conditional branch, and
termination logic as the dynamic scheduler. This fast path requires all
launched workgroups to be concurrently resident. The dynamic scheduler remains
available for irregular graphs and validates cyclic-graph rejection in the
benchmark.

The resident megakernel executes this FP32 dependency graph:

```text
attention RMSNorm
  -> 8 x fused Q/K/V projection over one packed output space
  -> fused Q/K RoPE + K/V cache update
  -> 7 x two-head grouped-query attention with online softmax
  -> 8 x output projection + attention residual add
  -> MLP RMSNorm
  -> 20 x fused gate projection + up projection + SiLU multiply
  -> 10 x down projection
  -> fused residual add + advance device state
  -> [final layer only] final RMSNorm
  -> worker-matched fused LM head projection + partition-local Top-1
  -> reduce partition-local candidates to the sampled token and logit
  -> record token, test EOS/limit, and select the next token embedding
```

The decoder layer is represented as two device-linked Programs: a 25-task
attention DAG followed by a 32-task FFN DAG. Together they contain the same 57
tasks, while each phase remains below the scheduler's 64-task limit. The Q/K/V
task maps a packed 1,152-row output range onto the three checkpoint matrices
and fuses bias addition. This avoids three separately scheduled projection
kinds while retaining parallel range tasks. A single successor applies Qwen's
half-split RoPE to Q and K in place, using the device-resident token position,
and writes K/V directly into token-major `[capacity, 128]` caches. Seven
attention tasks process two query heads each and map seven query
heads to each of the two KV heads. Each wave owns one 64-element head and uses
online softmax without block-wide barriers or a materialized score/probability
buffer. Eight output-projection tasks write disjoint hidden ranges and fuse the
attention residual add, so MLP RMSNorm directly consumes the post-attention
hidden state.

Matvec tasks use wave-coalesced weight reads, and each persistent workgroup
contains 256 threads to expose eight wave32 wavefronts. Gate, up, and activation
are fused so the gate and up intermediates never enter device memory. The final
FFN task adds the MLP residual, advances layer state, and publishes whether the
token reached its final layer.

The opaque context can also be an array of layer contexts. After each completed
FFN phase, the resident scheduler selects
`contexts[state.step % layer_count]`; weights and KV-cache slices remain
layer-local while activation scratch is shared. A Qwen-specific device
`DecodeState` explicitly owns token position, KV length, sampled token/logit,
generated-token count, EOS token, and stop reason. Non-final layers branch
directly back to Attention;
the final layer enters a third Program containing final RMSNorm, LM head, and
Top-1 reduction. The LM head never materializes the full logits vector: each
of at most 32 tasks emits one local token/logit candidate and the successor
reduces only those candidates. The resident path matches its partition count
to the admitted worker count, avoiding a second partition on only a subset of
workers. Its final task records the sampled token, tests
EOS and the maximum-token limit, and, when execution continues, copies the
selected full-vocabulary embedding row into the shared hidden buffer. It also
advances token position and KV length before returning to Attention. The tail
Program then evaluates the device predicate, so the complete autoregressive
loop remains inside one kernel launch.

The host-driven and HIP Graph baselines invoke the same arithmetic in nine
kernels per decoder layer plus four kernels per token for final RMSNorm,
partitioned LM head, Top-1 reduction, and embedding feedback. Their LM head
retains the independently faster 32-way split instead of inheriting the
resident worker count. The HIP Graph path loads the same HSACO as LRRT and
records that fixed kernel sequence as one graph submission. By default the
benchmark uses deterministic model-shaped decoder weights and a 4,096-row
synthetic LM head.
Pass `--weights` with an existing Qwen layer bundle manifest to run both paths
with its checkpoint attention norm, Q/K/V weights and biases, output projection,
`mlp_norm`, gate, up, and down tensors. Pass `--tail` to load an existing
`lrrt.mini_model_tail_weights` manifest:

```sh
./build/lrrt_qwen_stateful_graph_benchmark 64 20 20 \
  --weights /path/to/qwen-bundle/layer_0/weights.json \
  --tail /path/to/qwen-bundle/model_tail/weights.json
```

To exercise a full checkpoint decoder stack, point the benchmark at the bundle
root and select the layer count:

```sh
./build/lrrt_qwen_stateful_graph_benchmark 4 20 20 \
  --weights-dir /path/to/qwen-bundle --layers 24
```

When `--weights-dir` contains `model_tail/weights.json`, the benchmark loads it
automatically.

Four tokens through 24 layers use one compiled resident kernel, one HIP Graph
submission containing 880 kernel nodes, or 880 host-driven LRRT submissions.
The HIP Graph baseline is enabled for the ROCr backend when CMake finds the HIP
headers and `libamdhip64`; other builds retain the three LRRT paths. Absolute
timings remain sensitive to GPU clocks and concurrent load, so repeat the
measurement on the target system before making a whole-model speedup claim.

The benchmark measures each available path once per trial in an interleaved
forward-and-reverse rotation cycle. This balances paths across measurement
positions instead of timing all trials for one path before moving to the next.
LRRT paths use LRRT events and HIP Graph uses HIP events on its graph stream.
Both mean and median device event time and synchronized wall time are reported;
initialization and input reset remain outside the timed region for every path.
The result, decode state, generated tokens, and KV cache from HIP Graph are
checked against the host-driven LRRT path after measurement.

Pass `--csv` to preserve every measured invocation in execution order for
offline analysis and reproducibility:

```sh
./build/lrrt_qwen_stateful_graph_benchmark 4 20 20 \
  --weights-dir /tmp/lrrt-qwen-full --layers 24 \
  --csv /tmp/qwen-stateful-trials.csv
```

The CSV contains one row per path per measured trial with the one-based
`trial` and `position`, a stable path identifier, and raw `device_ns` and
`wall_ns` values. Warmup launches are excluded. Reading rows in file order
reconstructs the forward-and-reverse rotation used by the interleaved
measurement.

Pass `--profile-resident` to run a separate compiled-resident scheduler probe
after the normal interleaved measurement. The probe uses the same compiled
plan, worker count, stage barriers, Program transitions, layer branches, and
autoregressive termination path. It retains the residual-advance and embedding
feedback control tasks but skips the matrix, attention, normalization, and LM
head arithmetic. One warmup plus five measured probe launches report a
control-only scheduler/barrier floor. The benchmark subtracts that floor from
the uninstrumented compiled-resident mean to estimate the task-execution share:

```sh
./build/lrrt_qwen_stateful_graph_benchmark 4 20 20 \
  --weights-dir /tmp/lrrt-qwen-full --layers 24 --profile-resident
```

On the RX 7800 XT full-checkpoint run, the probe measured a 1.191 ms floor,
0.55% of the 214.837 ms compiled-resident mean, across 880 stage
synchronizations and 196 Program transitions. That run had substantially lower
and more variable GPU clocks than the result recorded below, so its absolute
model times are not a replacement benchmark result. The useful conclusion is
the within-run ratio: the control-only floor was far smaller than the resident
gap to HIP Graph. The next optimization target is therefore task execution
inside the persistent megakernel, including occupancy and memory/compute
efficiency, rather than the software scheduler alone. This probe is a
counterfactual lower bound, not a direct hardware stall counter; removing
arithmetic also changes cache and occupancy interactions.

### Full-checkpoint measurement

The full-weight comparison uses the official `Qwen/Qwen2.5-0.5B-Instruct`
checkpoint: all 24 decoder layers, the 151,936-row embedding table, final
RMSNorm, and tied LM head are converted to the existing FP32 bundle format.
The measured model file had SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
The conversion and benchmark commands were:

```sh
python3 tools/convert_qwen_layer.py \
  --checkpoint-dir /path/to/Qwen2.5-0.5B-Instruct \
  --all-layers --keys 4 --token-ids 0 --full-token-embeddings \
  --output /tmp/lrrt-qwen-full

./build/lrrt_qwen_stateful_graph_benchmark 4 20 20 \
  --weights-dir /tmp/lrrt-qwen-full --layers 24 \
  --csv /tmp/qwen-fair-lm-partitions.csv
```

With 20 resident workers, the resident LM head uses 20 partitions while the
host-driven and HIP Graph baselines use 32. On the gfx1101 AMD Radeon RX 7800
XT reference system, one 20-trial interleaved run produced:

| Path | Device mean | Device p50 | Wall mean | Wall p50 |
| --- | ---: | ---: | ---: | ---: |
| Dynamic resident graph | 180.942 ms | 171.402 ms | 181.058 ms | 171.468 ms |
| Compiled resident graph | 177.727 ms | 169.763 ms | 177.861 ms | 169.810 ms |
| Host-driven LRRT | 167.601 ms | 161.237 ms | 169.464 ms | 163.057 ms |
| HIP Graph | 164.748 ms | 157.581 ms | 164.772 ms | 157.604 ms |

The compiled-resident, host-driven, and HIP Graph correctness checks produced
matching graph state, hidden values, Q/K/V values, KV caches, final normalized
values, decode state, and generated token sequence. The compiled resident path
was `0.93x` the HIP Graph device and wall throughput in this run: its mean
device time was 12.979 ms, or 7.9%, longer. All 20 paired trials favored HIP
Graph; the paired device-time gap ranged from 8.595 ms to 25.506 ms. This is
the comparison baseline for further optimization: partition tuning for the
resident path may not reduce the baseline's independently selected
parallelism. The first-token hidden input remains deterministic benchmark data
rather than a tokenized prompt, while subsequent autoregressive steps use
checkpoint embedding rows selected entirely on the device.

The loader accepts the existing `lrrt.mini_decoder_weights` and
`lrrt.mini_model_tail_weights` FP32 formats. Autoregressive feedback requires
the model-tail bundle to contain every token embedding in vocabulary order.
Bundles whose hidden, head, KV-head, head-dimension, intermediate dimensions,
or embedding table do not match the specialized megakernel are rejected.

The resident kernel uses 1,072 bytes of group memory, 44 SGPRs, and 73 VGPRs on
gfx1101, with no private segment, dynamic stack, or register spills.

## Qwen path

Device-resident decode state, embedding feedback, EOS termination, and
resource-aware worker admission are now implemented. The benchmark compares
the whole loop against HIP Graph and host-driven LRRT using the same HSACO,
weights, buffer layout, and arithmetic. The full-checkpoint comparison is now
recorded above with independently selected LM-head partition counts. The next
milestone is a persistent-specific optimization that closes the measured
12.979 ms device-time gap without weakening the HIP Graph baseline.
