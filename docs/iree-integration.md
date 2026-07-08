# IREE Integration Investigation

## Purpose

This document defines the investigation path for connecting IREE to
`light-rocm-runtime` after the first Triton bundle milestone. The goal is not
to replace IREE's runtime or to embed IREE's graph execution stack into lrrt.
The goal is to understand whether IREE-produced AMD GPU dispatches can be
executed through a narrow lrrt-backed HAL adapter.

The target direction is:

```text
IREE compiler
  -> VMFB / HAL executable artifacts
  -> lrrt-backed IREE HAL adapter
  -> lrrt dispatcher and resource manager
  -> HSA / ROCm
```

## Current IREE ROCm Shape

IREE's documented ROCm path compiles programs for the HIP HAL target. A typical
compile command selects a HIP device and a ROCm target architecture:

```bash
iree-compile \
  --iree-hal-target-device=hip \
  --iree-rocm-target=gfx1101 \
  model.mlir -o model_rocm.vmfb
```

The resulting deployable artifact is normally run by IREE's own runtime through
the HIP HAL driver:

```bash
iree-run-module \
  --device=hip \
  --module=model_rocm.vmfb \
  --function=entry \
  --input="..."
```

The important observation for lrrt is that IREE already has the concepts that a
runtime needs: executable code, entry points, workgroup sizes, dynamic shared
memory, buffers, command submission, and synchronization. The integration
should therefore focus on implementing a narrow lrrt-backed device/adapter
under IREE's HAL-level contract, not on duplicating IREE's compiler,
frontend import layers, or high-level graph semantics.

## Integration Scope

The integration direction is a HAL-style lrrt adapter. Other connection models
are out of scope for this effort.

The target is to let IREE continue owning the VMFB, graph-level execution, HAL
command structure, and frontend integration, while lrrt provides the low-level
AMD GPU execution backend:

- device selection maps to `lr_device_open`
- buffers map to `lr_malloc`, `lr_free`, and copy APIs
- executable loading maps to `lr_module_load_hsaco`
- entry point lookup maps to `lr_kernel_get`
- command dispatch maps to `lr_launch_on_queue_with_dependencies`
- fences or timeline points map to lrrt events and queue synchronization

This is the most seamless user-facing path because the caller should still run
an IREE-compiled module through IREE's runtime shape, with lrrt sitting under
the HAL boundary as the dispatcher and resource manager.

This is also the largest integration path. It requires understanding enough of
IREE's HAL contracts to map executable loading, buffer lifetimes, command
submission, dependency handling, and synchronization onto lrrt without pulling
IREE semantics into the lrrt C runtime core.

The following connection models are explicit non-goals for this IREE work:

- exporting IREE dispatches into standalone lrrt bundles
- adding a VMFB-to-manifest conversion path
- using IREE only as a side-by-side reference runtime
- bypassing IREE's runtime shape to call generated kernels manually

## Recommended Path

The first useful result is a tiny IREE-compiled module that can be invoked
through an IREE-shaped runtime path while dispatching through lrrt. The adapter
should be narrow and experimental at first: one device, one queue model, one or
two static-shape dispatches, explicit resource ownership, and clear unsupported
feature errors.

The runtime core should only gain APIs if the investigation proves that a
dispatch-level contract needs them. Candidate runtime-level gaps are:

- richer explicit dependency validation
- queue/event primitives that better match compiler-generated schedules
- workspace allocation patterns that can be reused across dispatches
- diagnostics for target architecture or kernarg layout mismatches

Graph ordering, tensor shape propagation, dynamic dispatch regions, parameter
loading, and model invocation remain outside the lrrt core.

## Investigation Questions

The first investigation should answer these questions:

| Question | Why it matters |
| --- | --- |
| Can the generated HSACO be extracted from an IREE ROCm VMFB or intermediate artifact? | lrrt currently loads HSACO directly. |
| Where are entry point names and workgroup sizes stored? | The adapter must map them into lrrt launch configuration. |
| Where is dynamic shared memory recorded? | lrrt launch config needs this value. |
| How are buffer and scalar arguments packed into kernargs? | The adapter must bind arguments safely without owning graph semantics. |
| Does IREE require runtime-side shape or workload calculation? | If yes, that logic belongs in an adapter, not in runtime core. |
| What synchronization model does the compiled program expect? | lrrt events/queues may need adapter-level mapping. |
| What is the smallest HAL surface needed for one static dispatch? | This defines the first adapter prototype. |

## Prototype Milestones

### I0: Tooling baseline

- Install or build an IREE compiler with ROCm target support.
- Confirm the local ROCm target, for example `gfx1101`.
- Compile a tiny MLIR or ONNX-imported operator with
  `--iree-hal-target-device=hip`.
- Run the compiled VMFB with `iree-run-module --device=hip` to establish a
  baseline outside lrrt.

The local tool build strategy is tracked in
[IREE Tool Build Strategy](iree-tool-build.md).

The first local compile probe is tracked by `tools/iree_compile_probe.sh`. It
uses `tools/iree_minimal_mul.mlir` and emits `executable-configurations` and
`executable-targets` MLIR into `build-iree-probe/`. These intermediate forms
are useful before a full VMFB baseline exists because they expose the first
adapter-relevant anchors:

- `hal.executable` and `hal.executable.variant`
- `hal.executable.export`
- target architecture and ROCm backend attributes
- pipeline binding layout
- workgroup size and subgroup size
- lowered `llvm.func` kernel symbol
- `stream.cmd.dispatch`

On the current development machine, full VMFB serialization requires a recent
LLD. The system `ld.lld` is version 14 and fails with
`lld: error: unknown abi version`. The probe therefore prefers
`/opt/rocm/llvm/bin/lld` when present, which allows `--try-vmfb` to serialize a
VMFB and run it through IREE's HIP runtime baseline.

### I1: HAL contract mapping

- Identify the minimal IREE HAL interfaces needed to load an executable,
  allocate buffers, submit one dispatch, and wait for completion.
- Map each required HAL concept to an existing lrrt API.
- Record missing lrrt runtime capabilities as adapter requirements, not core
  changes, unless they are generally useful for dispatch/resource management.

The current adapter skeleton mapping is tracked in
[IREE HAL Adapter Mapping](iree-hal-mapping.md).

### I2: Artifact inspection

- Inspect the VMFB or compiler intermediates.
- Identify where HSACO bytes, entry point names, workgroup sizes, and dispatch
  metadata are stored.
- Use this to validate the adapter's executable-loading assumptions.

### I3: Single-dispatch adapter prototype

- Load one IREE-generated executable through the lrrt-backed adapter.
- Bind inputs through the adapter's buffer path.
- Submit one static-shape dispatch through lrrt.
- Validate output against the IREE runtime result or a CPU reference.

### I4: Compiler support decision

- Decide whether the adapter belongs in this repository, a separate
  `executor/iree` layer, or a companion project.
- Document unsupported IREE features explicitly.

## Non-goals

- Do not implement an IREE compiler pass in the runtime core.
- Do not parse VMFB in the C runtime core.
- Do not implement IREE VM, Flow, Stream, or full HAL semantics inside lrrt.
- Do not make lrrt responsible for frontend import from PyTorch, ONNX, JAX, or
  TensorFlow.
- Do not add high-level tensor graph semantics to lrrt for the first IREE
  prototype.

## References

- IREE ROCm deployment guide:
  <https://iree.dev/guides/deployment-configurations/gpu-rocm/>
- IREE HIP HAL driver design:
  <https://iree.dev/developers/design-docs/hip-hal-driver/>
