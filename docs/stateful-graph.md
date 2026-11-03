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

## Prototype v2

The prototype defines a 64-byte device `State`, a 32-byte `Task`, and a 16-byte
`Program` descriptor. Each Program selects a DAG of up to 64 tasks from a
shared task table and names the next Program. A fixed set of persistent
workgroups repeatedly:

1. scan for a task whose dependencies are complete;
2. atomically claim and execute that task;
3. publish completion so dependent tasks become ready;
4. let the final completing worker evaluate a device-side predicate; and
5. atomically publish the next Program phase or return.

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

## Qwen decoder megakernel

The scheduler ABI carries an opaque execution-context pointer, keeping model
buffer semantics out of the generic graph state and task descriptors. The Qwen
decoder specialization defines its context separately and uses the
Qwen2.5-0.5B dimensions `hidden=896`, `heads=14`, `kv_heads=2`, `head_dim=64`,
and `intermediate=4864`.

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
task adds the MLP residual and updates the predicate, which terminates the
repeated graph loop without host intervention.

The opaque context can also be an array of layer contexts. After each completed
FFN phase, the resident scheduler selects
`contexts[state.step % layer_count]`; weights and KV-cache slices remain
layer-local while activation scratch is shared. The token position is
`state.step / layer_count`, and only the final layer's residual task advances
the token predicate. Consequently, a 24-layer decoder stack and multiple token
iterations remain inside one kernel launch rather than returning to the host
between layers.

The host-driven baseline invokes the same arithmetic in nine stage launches
per step. By default the benchmark uses deterministic model-shaped weights,
isolating execution and scheduling performance while preserving the real FFN
dimensions and memory footprint. Pass `--weights` with an existing Qwen layer
bundle manifest to run both paths with its checkpoint attention norm, Q/K/V
weights and biases, output projection, `mlp_norm`, gate, up, and down tensors:

```sh
./build/lrrt_stateful_graph_benchmark 64 20 20 \
  --weights /path/to/qwen-bundle/layer_0/weights.json
```

To exercise a full checkpoint decoder stack, point the benchmark at the bundle
root and select the layer count:

```sh
./build/lrrt_stateful_graph_benchmark 4 20 20 \
  --weights-dir /path/to/qwen-bundle --layers 24
```

The host-driven comparison launches the same nine arithmetic stages for every
layer and token. Four tokens through 24 layers therefore use one resident
launch versus 864 host launches. On the local gfx1101 system, the barrier-free
multi-Program path completed that synthetic workload 1.20x faster in both
device and wall time. A longer 64-token, two-layer run traversed 256 Program
phases in one launch and measured 1.08x faster than 1,152 host launches.
Absolute timings remain sensitive to GPU clocks and concurrent load, and the
real 24-layer checkpoint comparison must be rerun before making a whole-model
speedup claim.

The loader accepts the existing `lrrt.mini_decoder_weights` FP32 format and
rejects bundles whose hidden, head, KV-head, head-dimension, or intermediate
dimensions do not match the specialized megakernel.

The resident kernel uses 1,040 bytes of group memory, 42 SGPRs, and 72 VGPRs on
gfx1101, with no private segment, dynamic stack, or register spills.

## Qwen path

The next milestones are:

1. add final RMSNorm, LM head, and sampling as additional device-linked
   Programs;
2. make KV length, sampled token, EOS, and embedding selection explicit
   device-resident decode state;
3. add resource-aware worker admission from HSACO metadata; and
4. compare a whole decode loop against HIP Graph and host-driven LRRT.
