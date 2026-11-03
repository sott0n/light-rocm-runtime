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
  -> 32 x fused LM head projection + partition-local Top-1
  -> reduce 32 candidates to the sampled token and logit
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
reduces only those candidates. Its final task records the sampled token, tests
EOS and the maximum-token limit, and, when execution continues, copies the
selected full-vocabulary embedding row into the shared hidden buffer. It also
advances token position and KV length before returning to Attention. The tail
Program then evaluates the device predicate, so the complete autoregressive
loop remains inside one kernel launch.

The host-driven baseline invokes the same arithmetic in nine launches per
decoder layer plus four launches per token for final RMSNorm, partitioned LM
head, Top-1 reduction, and embedding feedback. By default the benchmark uses
deterministic model-shaped decoder weights and a 4,096-row synthetic LM head.
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

Four tokens through 24 layers plus the synthetic LM head use one compiled
resident launch versus 880 host launches. Absolute timings remain sensitive to
GPU clocks and concurrent load, and the full-vocabulary 24-layer checkpoint
comparison must be rerun before making a whole-model speedup claim.

The benchmark measures the dynamic resident, compiled resident, and host-driven
paths once per trial in an interleaved six-permutation cycle. This balances each
path across the first, middle, and last measurement position instead of timing
all trials for one path before moving to the next. Both mean and median device
event time and synchronized wall time are reported; initialization and input
reset remain outside the timed region for every path.

The loader accepts the existing `lrrt.mini_decoder_weights` and
`lrrt.mini_model_tail_weights` FP32 formats. Autoregressive feedback requires
the model-tail bundle to contain every token embedding in vocabulary order.
Bundles whose hidden, head, KV-head, head-dimension, intermediate dimensions,
or embedding table do not match the specialized megakernel are rejected.

The resident kernel uses 1,072 bytes of group memory, 44 SGPRs, and 73 VGPRs on
gfx1101, with no private segment, dynamic stack, or register spills.

## Qwen path

Device-resident decode state, embedding feedback, EOS termination, and
resource-aware worker admission are now implemented. The next milestone is a
whole-loop comparison against HIP Graph and host-driven LRRT using the same
full-checkpoint weights and arithmetic.
