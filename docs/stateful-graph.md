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

## Prototype v1

The prototype defines a 64-byte device `State` and a 32-byte `Task`
descriptor. Up to 64 tasks express their dependencies as 64-bit masks. A fixed
set of persistent workgroups repeatedly:

1. scan for a task whose dependencies are complete;
2. atomically claim and execute that task;
3. publish completion so dependent tasks become ready;
4. let the final completing worker evaluate a device-side predicate; and
5. reset the task masks for the next epoch or return.

The masks are stored as two 32-bit atomic words so the original 32-task fast
path remains intact. Epoch transitions use a device-side barrier across every
admitted persistent worker before either word is reset. This prevents a worker
from combining a claimed mask from one epoch with a completed mask from the
next. The benchmark exercises a 64-task dependency chain through bit 63 before
running the timed FFN graph.

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
  -> MLP RMSNorm
  -> 20 x fused gate projection + up projection + SiLU multiply
  -> 10 x down projection
  -> fused residual add + advance device state
```

The attention input and FFN stages consume 41 of the scheduler's 64 task slots,
leaving 23 slots for RoPE, KV-cache, attention, and output-projection work. The
Q/K/V task maps a packed 1,152-row output range onto the three checkpoint
matrices and fuses bias addition. This avoids three separately scheduled
projection kinds while retaining parallel range tasks. Matvec tasks use
wave-coalesced weight reads, and each persistent workgroup contains 256 threads
to expose eight wave32 wavefronts. Gate, up, and activation are fused so the
gate and up intermediates never enter device memory. The Q/K/V results are
materialized for the next attention milestone; until attention output and its
residual are connected, the FFN still consumes the pre-attention hidden state.
The final task updates the predicate, which terminates the repeated graph loop
without host intervention.

The host-driven baseline invokes the same arithmetic in six stage launches per
step. By default the benchmark uses deterministic model-shaped weights,
isolating execution and scheduling performance while preserving the real FFN
dimensions and memory footprint. Pass `--weights` with an existing Qwen layer
bundle manifest to run both paths with its checkpoint attention norm, Q/K/V
weights and biases, `mlp_norm`, gate, up, and down tensors:

```sh
./build/lrrt_stateful_graph_benchmark 64 20 20 \
  --weights /path/to/qwen-bundle/layer_0/weights.json
```

The loader accepts the existing `lrrt.mini_decoder_weights` FP32 format and
rejects bundles whose hidden, head, KV-head, head-dimension, or intermediate
dimensions do not match the specialized megakernel.

The resident kernel uses 1,032 bytes of group memory, 64 SGPRs, and 39 VGPRs on
gfx1101, with no private segment, dynamic stack, or register spills.

## Qwen path

The next milestones are:

1. add RoPE, KV-cache, attention, and output-projection stage specializations;
2. keep token position, KV length, sampled token, and EOS in device state;
3. add resource-aware worker admission from HSACO metadata; and
4. compare a whole decode loop against HIP Graph and host-driven LRRT.
