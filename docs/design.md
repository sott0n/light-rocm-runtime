# Runtime Design

## Purpose

`light-rocm-runtime` is a small execution substrate for compiler-generated AMD
GPU workloads. It loads pre-built AMDGPU code objects, manages low-level device
resources, submits kernel dispatches, and exposes the synchronization primitives
needed by an executor.

The runtime is intentionally below tensor, operator, graph, and model semantics.
It is also intentionally below the HIP runtime API: the implementation uses
ROCr/HSA directly and exposes a small project-specific C ABI.

The intended stack is:

```text
compiler or model layer
  -> generated kernels and scheduling metadata
  -> executor: tensors, arenas, workspaces, queues, dependencies
  -> light-rocm-runtime: devices, memory, code objects, dispatch, events
  -> ROCr/HSA
  -> AMDGPU kernel driver and GPU
```

## Design Goals

### Keep the runtime boundary small

The core API should contain only mechanisms required to execute a compiled
kernel:

- initialize and enumerate devices
- allocate and copy device memory
- load an HSACO and resolve a kernel symbol
- create queues and events
- submit kernel dispatches
- express dependencies and wait for completion

Features that require tensor shapes, dtypes, graph analysis, model structure, or
operator knowledge belong above this boundary.

### Make resource ownership predictable

Every low-level resource created through the C API has an explicit owner and
destruction operation. The C++ API adds RAII wrappers without introducing a
second ownership model.

Destruction is conservative. Operations such as `lr_free`,
`lr_module_destroy`, and `lr_shutdown` wait for work that could still reference
the resource. This favors deterministic lifetimes and debuggability over fully
stream-ordered destruction.

The detailed ownership boundary is documented in
[Resource Ownership Policy](resource-ownership.md).

### Keep submission asynchronous, but dependencies explicit

A successful `lr_launch` means that a dispatch was enqueued; it does not mean
that the kernel completed. Completion and asynchronous execution errors are
observed through events or synchronization.

The default queue provides a convenient ordered path. Additional queues allow
independent work to overlap, but cross-queue ordering must be represented with
events and explicit dependency APIs. The runtime implements these dependencies
with HSA signals and barrier packets rather than host waits where possible.

### Preserve the compiler/runtime ABI boundary

The runtime consumes a compiled code object and packed kernargs. It does not
lower operators or infer a launch schedule.

The optional `lrrt::bundle` layer adds a small manifest containing the kernel
symbol, kernarg layout, launch shape, shared-memory requirement, and workspace
metadata. The manifest is a dispatch ABI, not a graph or tensor IR. See
[Bundle Manifest Schema](manifest-schema.md).

### Keep performance costs visible

The runtime avoids hiding graph transformations or model policies behind the
launch API. It exposes event timing, queue synchronization, and lrrt-local
memory counters so an executor can distinguish submission, execution,
synchronization, allocation, and transfer costs.

These facilities are diagnostic. They are not a replacement for ROCm system
profilers or global VRAM accounting.

## Responsibility Boundaries

| Layer | Owns |
| --- | --- |
| Compiler | Kernel generation, target selection, kernel ABI, static launch metadata, operator lowering |
| Bundle producer | HSACO packaging and consistency between the manifest and code-object metadata |
| Executor | Tensor and weight ownership, arena layout, workspace reuse, KV cache, queue selection, dependency wiring, graph/model scheduling |
| Runtime core | HSA lifecycle, device handles, queues, signals/events, device allocations, copies, code-object loading, dispatch packets, synchronization |
| ROCr/HSA | Agents, memory regions, executable loading, AQL queues and packets, signal operations, interaction with the AMDGPU driver |

The boundary is deliberate: adding a feature to the runtime is appropriate only
when it can be expressed without model- or operator-specific policy.

## Execution Model

### Devices and queues

`lr_device_open` lazily creates a default HSA queue. User queues are separate
HSA multi-producer queues. Work submitted to one lrrt queue is kept
completion-ordered. Work on different queues is unordered unless connected by
an event dependency.

`lr_queue_synchronize` waits for one queue. `lr_synchronize` is a device-wide
drain and therefore includes every lrrt queue and pending asynchronous copy for
that device.

### Dispatch

The runtime resolves these values from the loaded kernel:

- kernel object
- kernarg segment size
- fixed group segment size
- private segment size

For every dispatch it allocates or reuses a kernarg buffer, zero-initializes the
full buffer, copies the caller-provided argument bytes, builds an HSA kernel
dispatch packet, and associates a completion signal. Signals and kernarg
buffers are recycled after completion.

`lr_launch_config_t::grid` is the HSA total grid size in work-items. It is not
the HIP block count or Triton program count. `block` is the HSA workgroup size.
The bundle layer is responsible for converting its constrained launch
expression into this representation.

### Copies and events

Synchronous copies drain pending device work before calling `hsa_memory_copy`.
Asynchronous copies use `hsa_amd_memory_async_copy` and report completion
through an event.

The convenience APIs add dependencies between default-queue dispatches and
asynchronous copies. APIs with `_with_dependencies` use only the dependencies
provided by the caller. Explicit user queues do not acquire implicit
dependencies.

An event is device-scoped and owns one HSA signal. Re-recording or destroying
an event may wait or drain the device if queued packets still reference that
signal.

## Current Limitations

The following limitations describe the current implementation contract. Some
are intentional non-goals; others identify areas that may be expanded without
changing the core design.

### Platform and compatibility

- Linux, the AMDGPU driver stack, and an installed ROCr/HSA runtime are the
  supported execution environment.
- A build without HSA headers or `hsa-runtime64` produces a stub library. The
  API can be compiled and basic validation can run, but GPU operations return
  `LR_ERROR_NOT_SUPPORTED`.
- The runtime is not HIP API compatible. It does not provide the complete HIP
  device, stream, graph, module, memory, callback, or interoperability surface.
- The runtime loads pre-built HSACO images only. It has no source compiler, JIT,
  linker, or kernel cache.
- HSACO ISA compatibility with the selected GPU is not proactively negotiated
  by the core runtime. Bundle consistency tests can compare a manifest target
  with code-object metadata, but deployment must still provide a compatible
  artifact.

### Runtime lifecycle and host concurrency

- Initialization is process-global and not reference-counted. Only one active
  lrrt runtime lifecycle is supported; independently constructed C++ `Runtime`
  objects must not be treated as isolated runtime instances.
- Runtime registries and submission state are protected by a process-global
  mutex. This simplifies handle validation and lifetime safety, but host-side
  API calls do not scale independently across devices or queues.
- Queues are created with a fixed requested capacity of 1024 packets, reduced
  to the device maximum. When tracked work reaches queue capacity, the current
  recovery path may drain the entire device rather than applying fine-grained
  backpressure to one producer.
- Queue priorities, callbacks, cooperative dispatch policy, and per-thread
  default queue semantics are not exposed.

### Device and multi-GPU support

- The core can enumerate multiple GPU agents, but it does not implement
  topology-aware selection, load balancing, peer access, peer copies, or
  cross-device event dependencies.
- Allocations, modules, queues, kernels, and events are tied to one device.
  Device-to-device copies currently mean two registered allocations on the
  same lrrt device, not a peer-to-peer transfer.
- The experimental IREE HAL driver exposes only lrrt device index 0 even when
  the core runtime can enumerate more devices.

### Memory model

- Device allocations use the first suitable allocatable coarse-grained global
  HSA region selected for an agent. There is no public memory-pool selection or
  placement policy.
- Pinned-host allocations are device-scoped. The runtime locks ordinary host
  memory with `hsa_amd_memory_lock`, retains both the host and agent mappings,
  and translates registered host subranges for asynchronous H2D and D2H copies.
  It does not expose the agent mapping for zero-copy kernel access.
- There is no managed/unified memory API, fine-grained shared allocation,
  virtual memory API, external allocation import/export, or IPC handle support.
- There is no runtime allocation pool or stream-ordered allocation/free API.
  `lr_free` performs a device-wide drain before releasing memory.
- The allocation registry tracks only memory allocated by `lr_malloc`. Copy
  APIs reject device pointers outside registered lrrt allocations, even if
  another ROCm component created otherwise valid GPU-accessible memory.
- `lr_free` accepts only the exact base pointer returned by `lr_malloc`; copy
  APIs may use validated subranges.
- `lr_host_malloc` allocations have runtime-managed pinning and lifetime.
  Arbitrary pageable host pointers remain accepted for compatibility, but the
  API does not guarantee useful copy/compute overlap for them.
- Memory statistics cover only allocations and copies submitted through lrrt.
  They do not report available VRAM, allocations owned by other runtimes,
  compiler scratch, LDS, or kernel private-segment usage.

### Synchronization and scheduling

- Synchronous copies, allocation destruction, module destruction, and shutdown
  introduce broad implicit synchronization. They are safe lifetime boundaries,
  not high-throughput stream-ordered operations.
- Implicit copy/dispatch dependencies apply to the default convenience path.
  Work submitted to explicit queues requires explicit event dependencies.
- Events are reusable completion/timing objects, but they may be re-recorded
  only after safe reuse. There are no public timeline semaphores, external
  semaphores, host callbacks, or wait-any operation.
- Dependencies are device-local. The runtime does not detect higher-level data
  hazards; it only orders the events supplied by the caller or implied by the
  default path.
- The runtime does not schedule kernels based on occupancy, queue load, copy
  engines, NUMA placement, or graph-level lifetime information. Those policies
  remain executor responsibilities.

### Kernel and launch ABI

- The core accepts already-packed raw kernarg bytes. It does not interpret
  argument types, tensor metadata, address spaces, or compiler-specific hidden
  arguments.
- The core verifies that the supplied byte count does not exceed the kernel's
  kernarg segment, but it cannot prove that offsets, alignment, pointer values,
  or argument meanings match the code object.
- The bundle layer provides stronger manifest validation, but the current
  schema supports only version 1 and a constrained one-dimensional grid
  expression. It does not describe general dynamic shapes or multidimensional
  launch derivation.
- Workspace size is metadata for an executor. The runtime and `Bundle::launch`
  do not allocate, reuse, or automatically bind workspace memory.
- Fixed group/private segment sizes are read from the kernel descriptor and
  dynamic shared memory is added at dispatch. The runtime does not perform a
  complete hardware-capability or occupancy validation before submission.
- There is no graph capture, command-buffer optimization, kernel fusion, or
  automatic batching in the core runtime.

### Errors and observability

- Most HSA failures are collapsed to `LR_ERROR_RUNTIME`; the public status does
  not preserve the native HSA status, failing queue, or dispatch identity.
- Kernel execution failures can be asynchronous and may be observed only by a
  later event wait, queue synchronization, device synchronization, or
  destruction operation that drains work.
- Event elapsed time and individual operation duration are available for
  completed markers and copies, but the runtime does not expose a full tracing
  stream, hardware counters, or per-kernel occupancy information.
- Profiling is enabled on every lrrt-created queue. There is currently no API to
  disable it or select profiling features.

### Compiler integrations

- Triton and IREE support is optional and does not form part of the default
  runtime build.
- The bundle manifest is a repository-local dispatch contract, not a stable
  cross-project standard.
- The IREE HAL adapter is a correctness-oriented experimental bridge, not a
  complete or performance-equivalent HIP/ROCm HAL. Its detailed supported and
  unsupported behavior is tracked in
  [IREE HAL Adapter Coverage](iree-hal-coverage.md).
- Model execution examples do not move tensor, tokenizer, weight-layout, or KV
  cache semantics into the runtime core. Those remain executor/compiler
  concerns even when an example runs end to end.

## Intentional Non-Goals

Unless the project scope changes explicitly, the runtime should not:

- implement a compiler, graph optimizer, or operator library
- replace rocBLAS, hipBLASLt, MIOpen, RCCL, or other ROCm libraries
- become a general tensor or model runtime
- own model weights, KV cache policy, token scheduling, or tensor arenas
- reproduce the full HIP API
- bypass ROCr with a direct `/dev/kfd` ioctl backend
- infer workload scheduling policy from kernel names or argument contents

These exclusions preserve a small boundary that can be used by different
compiler and executor layers.

## Criteria for Extending the Core

A feature belongs in the runtime core when all of the following are true:

1. It is useful across compiler and model frontends.
2. It can be expressed using byte buffers, executable objects, queues, events,
   and device-level capabilities.
3. It does not require tensor, operator, graph, or model semantics.
4. Its ownership and synchronization behavior can be stated precisely.
5. It does not silently add broad synchronization or unpredictable allocation
   to the normal dispatch path.

Features that fail these criteria should live in the bundle, executor, or
compiler integration layer.
