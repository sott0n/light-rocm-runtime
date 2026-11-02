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

## Prototype v0

The first prototype defines a 64-byte device `State` and a 32-byte `Task`
descriptor. A fixed set of persistent workgroups repeatedly:

1. execute disjoint tasks;
2. arrive at a device-wide epoch counter;
3. let the final arriving worker advance graph state;
4. evaluate a predicate stored in device memory; and
5. either begin the next epoch or return.

The accompanying benchmark compares one resident launch with an equivalent
sequence of host-driven launches using kernels from the same HSACO.

The v0 grid barrier assumes every worker workgroup is concurrently resident.
The benchmark therefore launches only four small workgroups on the fixed
gfx1101 target. A production scheduler must add resource-aware admission and
must never launch more persistent workers than can be resident together.

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
2. represent dependencies with per-task event epochs instead of a full-grid
   barrier after every task;
3. lower one Qwen layer into the task ABI;
4. keep token position, KV length, sampled token, and EOS in device state; and
5. compare a whole decode loop against HIP Graph and host-driven LRRT.
