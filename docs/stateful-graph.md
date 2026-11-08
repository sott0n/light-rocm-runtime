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
  -> 32 x down projection
  -> fused residual add + advance device state
  -> [final layer only] final RMSNorm
  -> worker-matched fused LM head projection + partition-local Top-1
  -> reduce partition-local candidates to the sampled token and logit
  -> record token, test EOS/limit, and select the next token embedding
```

The decoder layer is represented as two device-linked Programs: a 25-task
attention DAG followed by a 54-task FFN DAG. Together they contain 79
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
workers. The Qwen HSACO uses the compiler's default WGP mode and limits the
launch to 20 persistent workgroups. The generic resource calculation permits
more workgroups, but it does not prove simultaneous residency for the
grid-wide software stage barrier. Experimental CU-mode launches and larger
WGP-mode launches caused non-deterministic hangs and GPU resets, so the Qwen
benchmark rejects worker counts above the conservative limit before launch.
Its final task records the sampled token, tests
EOS and the maximum-token limit, and, when execution continues, copies the
selected full-vocabulary embedding row into the shared hidden buffer. It also
advances token position and KV length before returning to Attention. The tail
Program then evaluates the device predicate, so the complete autoregressive
loop remains inside one kernel launch.

The host-driven and HIP Graph baselines invoke the same arithmetic in nine
kernels per decoder layer plus four kernels per token for final RMSNorm,
partitioned LM head, Top-1 reduction, and embedding feedback. Their LM head
owns an independent task table and retains its 32-way split even when resident
admission selects fewer workers. The HIP Graph path loads the same HSACO as
LRRT and records that fixed kernel sequence as one graph submission. By
default the benchmark requests 20 resident workers, then clamps that request
through the device/kernel resource limit. It uses deterministic model-shaped
decoder weights and a 4,096-row synthetic LM head.
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

For an operator breakdown and an explicit roofline latency floor, pass
`--profile-roofline`. This mode also runs the scheduler probe, then runs five
additional host-driven profiles with a GPU event after every stage. It reports
the event-marker time separately and subtracts each marker packet from its
preceding stage interval. The uninstrumented resident, host, and HIP Graph
measurements still run first and are not included in this profile.

The same option also runs dedicated instrumented resident and HIP Graph paths.
The resident kernel reads the constant-rate GPU wall clock when each compiled
stage completes. Its intervals include the compiled barrier and Program
transition leading into the next stage. The HIP Graph profile records the
earliest workgroup start and latest workgroup completion for every kernel node;
the reported graph-gap total contains time outside those kernel envelopes.
Both paths aggregate the same 13 operator classes and report minimum effective
bandwidth, bandwidth utilization when a hardware limit is supplied, and the
largest resident-over-HIP penalties. These are separate five-trial diagnostic
passes; the normal comparison kernels contain no stage instrumentation.

The optional hardware limits are explicit inputs because LRRT device
properties do not expose a reliable active-clock FP32 rate or DRAM bandwidth:

```sh
./build/lrrt_qwen_stateful_graph_benchmark 4 20 20 \
  --weights-dir /tmp/lrrt-qwen-full --layers 24 \
  --profile-roofline \
  --peak-fp32-tflops 37.32 --memory-bandwidth-gbps 624
```

The operation model counts FP32 add, multiply, and FMA work, but does not
assign synthetic FLOP counts to `exp` or `rsqrt`. The byte model is an
optimistic DRAM minimum: every weight is streamed once, while activation and
KV-head reuse are assumed ideal. Consequently, the reported roof time is a
latency floor, not a prediction. Supplying no hardware limits still reports
stage time, modeled work, arithmetic intensity, and effective minimum
bandwidth. The peak and bandwidth values must describe the target device and
clock mode; they are not inferred from the device name.

On the same RX 7800 XT and full-checkpoint shape, one profile using the limits
above measured the following dominant stages. `Roof` is
`max(FLOPs/peak, minimum-bytes/bandwidth)` and `gap` is measured stage time
minus that floor:

| Stage | Measured | Minimum traffic | Effective minimum bandwidth | Roof | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| QKV projection | 14.400 ms | 0.398 GB | 27.6 GB/s | 0.637 ms | 13.763 ms |
| attention output | 11.368 ms | 0.309 GB | 27.2 GB/s | 0.496 ms | 10.872 ms |
| gate + up + SiLU | 24.421 ms | 3.349 GB | 137.1 GB/s | 5.367 ms | 19.053 ms |
| down projection | 46.212 ms | 1.676 GB | 36.3 GB/s | 2.685 ms | 43.526 ms |
| LM head + local Top-1 | 18.091 ms | 2.178 GB | 120.4 GB/s | 3.491 ms | 14.600 ms |
| all stages | 119.618 ms | 7.915 GB | 66.2 GB/s | 12.685 ms | 106.933 ms |

The modeled workload is 3.956 GFLOP at 0.500 FLOP/byte overall, far below the
59.81 FLOP/byte ridge point from the supplied limits. Its theoretical floor is
therefore bandwidth, not FP32 compute. The uninstrumented compiled resident
time was 113.077 ms, 8.91 times the 12.685 ms stage roof, while the independent
control-only scheduler/barrier floor was 0.971 ms, or 0.86% of resident time.
The instrumented interval was 121.281 ms, comprising a 119.618 ms stage sum
and 1.662 ms of event-marker packets.

This separates two optimization classes. Eliminating graph nodes and tiny
stage launches is persistent-specific, but it cannot recover the dominant
matrix time. The initial profile found under-filled memory-bound matvec work
shared by both execution paths: the down projection scheduled only 10
workgroups and reached about 36 GB/s while 32-way LM-head work reached about
120 GB/s.

Increasing the down projection from 10 to 32 partitions raised its available
parallelism from 80 to 256 wave32 wavefronts. A repeat of the same 20-trial
full-checkpoint profile on the experimental 32-worker CU-mode build produced:

| Metric | 10 partitions | 32 partitions | Change |
| --- | ---: | ---: | ---: |
| Down-projection stage | 46.212 ms | 17.063 ms | 63.1% shorter |
| Down effective minimum bandwidth | 36.3 GB/s | 98.2 GB/s | 2.71x |
| Down roof efficiency | 5.81% | 15.74% | 2.71x |
| All profiled stages | 119.618 ms | 94.272 ms | 21.2% shorter |
| Compiled resident | 113.077 ms | 96.978 ms | 14.2% shorter |
| HIP Graph | 114.990 ms | 99.013 ms | 13.9% shorter |

For that completed run, correctness remained identical across compiled
resident, host-driven, and HIP Graph execution. The modeled 12.685 ms bandwidth
floor did not change, so the overall compiled-resident distance to that floor
improved from 8.91x to 7.65x.
The resident scheduler/barrier probe rose from 0.971 ms to 1.151 ms because the
compiled plan contains 22 more tasks per layer, but it remains only 1.19% of
resident time. The result confirms that maintaining enough independent waves
to hide streaming-weight latency is more important than reducing scheduler
work at this point. These timing results motivate retaining 32 down-projection
tasks, but they do not establish that the 32-worker CU-mode launch is safe.

A later direct resident-versus-HIP profile on the supported 20-worker WGP
configuration localized the remaining gap. The run used one generated token,
all 24 checkpoint layers, three uninstrumented comparison trials, and five
trials per diagnostic path:

| Stage | Resident | HIP Graph | Resident penalty | Resident min BW | HIP min BW |
| --- | ---: | ---: | ---: | ---: | ---: |
| QKV projection | 3.398 ms | 3.362 ms | 0.036 ms | 29.3 GB/s | 29.6 GB/s |
| attention output | 2.624 ms | 2.587 ms | 0.037 ms | 29.5 GB/s | 29.9 GB/s |
| gate + up + SiLU | 5.887 ms | 5.864 ms | 0.023 ms | 142.2 GB/s | 142.8 GB/s |
| down projection | 7.587 ms | 3.842 ms | 3.746 ms | 55.2 GB/s | 109.0 GB/s |
| LM head + local Top-1 | 7.098 ms | 4.478 ms | 2.621 ms | 76.7 GB/s | 121.6 GB/s |
| all stages | 27.003 ms | 20.382 ms | 6.621 ms | 73.3 GB/s | 97.1 GB/s |

The corresponding uninstrumented means were 27.108 ms for compiled resident
and 21.066 ms for HIP Graph. QKV, output projection, and gate/up were already
at parity between execution models. Down projection has 32 tasks, but only 20
resident workgroups can execute at once, so 12 workers receive a second task
after their first; the HIP kernel launches 32 independent workgroups. The LM
head similarly uses 20 resident partitions versus 32 baseline partitions.
Together these two streaming matvec stages explain essentially all of the
measured resident deficit. The next concurrency target is therefore more
simultaneous matvec waves within the safe 20-workgroup resident launch, before
software-prefetch changes to the inner loop.

The compiled Qwen workload now provides a stage-level execution hook for this
case. Its down-projection implementation redistributes the complete 896-row
output range over all 160 resident wave32 wavefronts instead of assigning 32
task partitions to 20 workgroups and making 12 workgroups run a second task
serially. Each wave accumulates two independent rows at once. This reuses the
activation load and exposes two weight streams to the memory pipeline without
increasing the persistent grid beyond the supported 20-workgroup limit. The
host-driven and HIP Graph kernels keep their original 32-workgroup arithmetic
path.

A same-session A/B run used 16 generated tokens, all 24 checkpoint layers,
three uninstrumented trials per path, and five diagnostic trials per profile.
Only the resident down-projection stage hook was disabled for the control run:

| Metric | Original task assignment | Resident stage specialization | Change |
| --- | ---: | ---: | ---: |
| Down-projection stage | 135.212 ms | 59.804 ms | 55.8% shorter |
| Down effective minimum bandwidth | 49.6 GB/s | 112.1 GB/s | 2.26x |
| Down resident / HIP Graph | 2.066x | 0.907x | resident became 9.3% shorter |
| Compiled resident | 524.952 ms | 439.167 ms | 16.3% shorter |
| Compiled resident / HIP Graph speed | 0.77x | 0.93x | 20.8% relative gain |

The specialized compiled kernel remains spill-free at 51 SGPRs, 79 VGPRs,
and 1,060 bytes of fixed group memory, so the optimization does not weaken the
resource assumptions behind the existing WGP20 admission limit. LM head is now
the dominant persistent-only deficit.

### Historical CU-mode full-checkpoint measurement

The full-weight comparison uses the official `Qwen/Qwen2.5-0.5B-Instruct`
checkpoint: all 24 decoder layers, the 151,936-row embedding table, final
RMSNorm, and tied LM head are converted to the existing FP32 bundle format.
The measured model file had SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
The conversion and benchmark commands on that experimental revision were:

```sh
python3 tools/convert_qwen_layer.py \
  --checkpoint-dir /path/to/Qwen2.5-0.5B-Instruct \
  --all-layers --keys 4 --token-ids 0 --full-token-embeddings \
  --output /tmp/lrrt-qwen-full

./build/lrrt_qwen_stateful_graph_benchmark 4 20 32 \
  --weights-dir /tmp/lrrt-qwen-full --layers 24 \
  --csv /tmp/qwen-cu32.csv
```

The current benchmark rejects this 32-worker command; it is retained only to
identify the historical measurement. Although one 20-trial interleaved run
completed on the gfx1101 AMD Radeon RX 7800 XT reference system, fresh-boot
retests produced CPC VM faults, unrecoverable MES hangs, and GPU resets. The
run produced:

| Path | Device mean | Device p50 | Wall mean | Wall p50 |
| --- | ---: | ---: | ---: | ---: |
| Dynamic resident graph | 151.688 ms | 145.050 ms | 151.750 ms | 145.111 ms |
| Compiled resident graph | 150.051 ms | 143.737 ms | 150.113 ms | 143.804 ms |
| Host-driven LRRT | 154.827 ms | 149.695 ms | 156.588 ms | 151.429 ms |
| HIP Graph | 151.996 ms | 146.075 ms | 152.018 ms | 146.096 ms |

For the completed run, the compiled-resident, host-driven, and HIP Graph
correctness checks produced
matching graph state, hidden values, Q/K/V values, KV caches, final normalized
values, decode state, and generated token sequence. The compiled resident path
had a 1.3% lower mean device and wall time than HIP Graph in this run. The
paired device-time difference ranged from 6.892 ms in favor of resident to
3.906 ms in favor of HIP Graph, with resident ahead in 15 of 20 trials. The
overlapping distribution makes this practical parity, not a performance-win
claim. A same-configuration 20-worker control run left resident 7.5% behind
HIP Graph, but the unsafe 32-worker result cannot be used as a supported
performance comparison. The first-token hidden input remains deterministic
benchmark data rather than a tokenized prompt, while subsequent autoregressive
steps use checkpoint embedding rows selected entirely on the device.

The loader accepts the existing `lrrt.mini_decoder_weights` and
`lrrt.mini_model_tail_weights` FP32 formats. Autoregressive feedback requires
the model-tail bundle to contain every token embedding in vocabulary order.
Bundles whose hidden, head, KV-head, head-dimension, intermediate dimensions,
or embedding table do not match the specialized megakernel are rejected.

The experimental CU-mode dynamic resident kernel used 1,072 bytes of group
memory, 44 SGPRs, and 73 VGPRs on gfx1101. Its compiled resident kernel used
1,060 bytes, 50 SGPRs, and 75 VGPRs. Neither used a private segment, dynamic
stack, or register spills.

## Qwen path

Device-resident decode state, embedding feedback, EOS termination, and
resource-aware worker admission are now implemented. The benchmark compares
the whole loop against HIP Graph and host-driven LRRT using the same HSACO,
weights, buffer layout, and arithmetic. The full-checkpoint comparison is now
recorded above with independent resident and baseline task tables. The
benchmark now uses WGP mode with a conservative 20-worker limit; CU-mode and
larger launches are unsupported after reproducing GPU-reset failures on a
fresh boot. The 32-way down-projection task split remains because it increased
streaming-weight parallelism independently of resident worker count. The next
milestone is to establish repeated full-stack stability at this limit, then
optimize the memory-bound task bodies without weakening the HIP Graph baseline.
