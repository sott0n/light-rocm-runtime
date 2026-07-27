# Resource Ownership Policy

`light-rocm-runtime` aims to be a low-overhead dispatcher and predictable
resource manager. Resource ownership must stay explicit so compiler and
executor layers can build larger pipelines without turning the runtime core into
a model runtime.

This document defines where memory and lifetime policy belongs.

## Runtime Core

The C runtime owns low-level AMD GPU resources created through the public C ABI:

- HSA initialization and shutdown
- opened device state and default queues
- user-created queues and events
- loaded HSACO modules and kernel handles
- allocations returned by `lr_malloc`
- pinned host allocations returned by `lr_host_malloc`
- copies submitted through `lr_memcpy` and `lr_memcpy_async`
- per-device memory statistics for lrrt-managed work

The runtime tracks allocations returned by `lr_malloc` in its allocation
registry. `lr_free` only accepts exact pointers returned by `lr_malloc` for the
same device. Copy APIs accept pointers inside registered allocations when the
requested byte range stays within the allocation bounds.

Pinned host allocations have a separate registry keyed by the host-visible
base pointer. Each entry retains the device, size, and HSA agent mapping
returned by `hsa_amd_memory_lock`. Asynchronous H2D and D2H copies translate
registered host subranges to that agent mapping. `lr_host_free` accepts only the
exact base pointer and drains the allocation's device before unlocking and
freeing it.

Asynchronous copies also accept pageable host pointers for compatibility. The
runtime temporarily locks and maps the requested host range, keeps that mapping
owned by the completion event, and unlocks it when the event is synchronized or
otherwise drained. This per-copy mapping is the overhead that persistent pinned
host allocations avoid.

The runtime intentionally does not infer tensor ownership, tensor shape, dtype,
strides, model weights, KV cache layout, or graph lifetime. It only validates
the resources it directly owns.

## C++ Wrapper

The C++ wrapper owns convenience lifetime management around the C ABI:

- `lrrt::Runtime` calls `lr_init` and `lr_shutdown`
- `lrrt::Queue` calls `lr_queue_create` and `lr_queue_destroy`
- `lrrt::Event` calls `lr_event_create` and `lr_event_destroy`
- `lrrt::Module` calls `lr_module_load_hsaco` and `lr_module_destroy`
- `lrrt::DeviceBuffer` calls `lr_malloc` and `lr_free`
- `lrrt::PinnedHostBuffer` calls `lr_host_malloc` and `lr_host_free`

These wrappers do not introduce a second ownership model. They are RAII helpers
for the same C runtime resources.

`lrrt::DeviceBuffer::view` is a non-owning view over an existing device pointer.
It must not free memory. This is useful for executor-owned arena slices, but the
underlying allocation must outlive every view.

## Executor Layer

Executor layers own higher-level resource policy:

- named tensor buffers
- arena packing and buffer offsets
- temporary workspace allocation from bundle metadata
- model-specific weight layout
- KV cache layout
- queue selection for pipeline stages
- event dependency wiring across kernels and queues
- benchmark phase boundaries

The current Triton executor uses `lrrt::DeviceBuffer` for allocations and
`DeviceBuffer::view` for arena slices. The arena policy belongs to the executor
because it depends on the generated bundle pipeline and model layout. The
runtime only sees one allocation plus subrange pointers used for copies and
kernargs.

## User Responsibilities

Users of the C ABI must keep these rules:

- call `lr_init` before using runtime resources
- pass `lr_free` only exact pointers returned by `lr_malloc`
- pass `lr_host_free` only exact pointers returned by `lr_host_malloc`
- free an allocation on the same device that allocated it
- keep allocations, modules, queues, and events alive until queued work using
  them has completed or until the runtime API drains pending work during
  destruction
- pass only valid byte ranges for device pointers used by copy APIs
- use `lr_synchronize`, `lr_queue_synchronize`, or events when explicit
  completion is required
- do not modify an H2D source or read a D2H destination until its asynchronous
  copy event completes

Users of the C++ wrapper should keep the owning object alive for as long as
non-owning handles or views may use it. For example, a `DeviceBuffer::view`
created from an arena allocation must not outlive the owning arena buffer.

## Memory Statistics

`lr_memory_stats_t` reports only work submitted through lrrt:

- live and peak bytes for `lr_malloc` allocations
- live and peak bytes for pinned host allocations
- device and pinned-host allocated and freed bytes
- device and pinned-host allocation and free counts
- copy byte counts by direction
- copy call count

The counters are not a system VRAM query. They do not include allocations made
by HIP, PyTorch, Triton, IREE, another runtime, or kernel-side scratch memory.

`lr_reset_memory_stats` preserves the current live byte count and sets the peak
to that live value. Accumulated allocation, free, and copy totals are cleared so
a benchmark can measure one phase after setup allocations are already live.

## Future Memory Pool Policy

A memory pool can be added to the runtime core only if it remains below tensor
semantics. A suitable runtime-level pool would manage generic byte allocations,
device ownership, and synchronization safety.

The following should remain outside the runtime core:

- model-specific arena layouts
- dtype-specific packing
- operator workspace planning
- KV cache growth policy
- graph-level lifetime analysis

Those policies belong in compiler, bundle, or executor layers because they
depend on the workload above the runtime.
