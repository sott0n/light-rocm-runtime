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
descriptor. Up to 32 tasks express their dependencies as bitmasks. A fixed
set of persistent workgroups repeatedly:

1. scan for a task whose dependencies are complete;
2. atomically claim and execute that task;
3. publish completion so dependent tasks become ready;
4. let the final completing worker evaluate a device-side predicate; and
5. reset the task masks for the next epoch or return.

The accompanying benchmark compares one resident launch with an equivalent
sequence of dependency-ordered host launches using kernels from the same
HSACO. Worker count and task count are independent: a small resident worker
pool can execute a larger task DAG without a device-wide barrier.
The resident kernel validates task bounds and rejects cyclic dependency graphs
before entering the scheduling loop.

The worker pool must still satisfy resource-aware admission: every persistent
worker must be concurrently resident. The benchmark limits the pool to 16
small workgroups, while tasks are scheduled onto that pool dynamically.

The prototype also uses a fixed worker width from the Stateful Graph ABI.
LRRT launches raw kernel argument buffers and deliberately does not synthesize
HIP-specific hidden `blockDim` or `gridDim` arguments, so resident kernels must
not depend on those fields.

## Qwen path

The synthetic range task is only a way to isolate scheduling cost. The planned
task vocabulary is stage-sized rather than elementwise:

```text
attention stage
output projection + residual + norm stage
MLP stage
sampling stage
advance-token stage
```

Stage implementations may be individually fused megakernels, but their
selection, dependencies, loop state, and early exit remain runtime concerns.
The next milestones are:

1. replace the synthetic opcode with two heterogeneous stage opcodes;
2. lower one Qwen layer into the task ABI;
3. keep token position, KV length, sampled token, and EOS in device state; and
4. compare a whole decode loop against HIP Graph and host-driven LRRT.
