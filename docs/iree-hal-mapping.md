# IREE HAL Adapter Mapping

## Purpose

This document records the current contract between IREE HAL concepts and the
experimental lrrt-backed adapter skeleton in `executor/iree`. It is intentionally
about the adapter boundary, not the runtime core. It explains how IREE HAL
concepts map onto lrrt concepts; it is not the status checklist for every
implemented or unsupported entry point.

The adapter should translate IREE HAL-level operations into lrrt C API calls
while keeping IREE VM, Flow, Stream, graph scheduling, and frontend semantics
outside `light-rocm-runtime`.

For the current implementation matrix, including test coverage and unsupported
features, see [IREE HAL Adapter Coverage](iree-hal-coverage.md).

## Current Skeleton

The current skeleton has two layers and is compiled only when
`LRRT_ENABLE_IREE_ADAPTER=ON`:

- a small C++ adapter layer that names the lrrt-owned concepts used by the
  existing metadata/HSACO dispatch probes
- a minimal native IREE HAL driver factory named `lrrt`, registered through
  `executor/iree/registration/driver_module.h`, plus an
  `executor/iree/registration/init.h` registration layer that mirrors IREE's
  `iree_hal_register_all_available_drivers(registry)` shape for user-provided
  drivers

The native HAL driver is built as a reusable CMake target,
`lrrt::iree_hal_driver`, when the required IREE runtime static libraries are
available. Repo-local smoke tools link that target instead of compiling the
driver implementation directly, keeping the registration and driver/device
implementation as adapter-owned code rather than smoke-runner-local code.

The native driver factory is the first step toward a seamless
`iree-run-module --device=lrrt` path. The low-level module registration
function keeps the same shape as IREE's built-in driver modules and accepts an
explicit registry. `lrrt_iree_hal_register_all_available_drivers(registry)` is
the lrrt-owned aggregate registration entry point for callers that manage their
own IREE registry. The higher-level `lrrt_iree_hal_register_all` helper is
idempotent and registers the same factory with `iree_hal_driver_registry_default`
so smoke runners and future embedded tooling entry points do not need to know
the registry plumbing. At this stage the driver can be created by name and
queried for one placeholder device. It can also create a minimal lrrt-backed
`iree_hal_device_t` with a
stable id, host allocator, empty capabilities, and a device-owned HAL allocator
skeleton. The allocator opens the first lrrt device and can allocate
device-local `iree_hal_buffer_t` objects backed by `lr_malloc`. Those buffers
release their lrrt allocation with `lr_free` when the HAL buffer is destroyed.
The buffer vtable supports HAL map/unmap/invalidate/flush through an
adapter-owned host shadow buffer; it does not expose a direct host-visible
mapping of the GPU allocation. External memory import/export and virtual memory
remain unsupported. The native device can create an executable cache that
recognizes lrrt-owned HSACO formats, prepares a raw HSACO into an lrrt module,
and lazily resolves exported function names into lrrt kernels. It can submit a
narrow direct `queue_dispatch` path for static workgroup counts, explicit
workgroup sizes, lrrt-owned HAL semaphore lists, no inline constants, and
pointer-only storage-buffer bindings. It also has a minimal command buffer path:
`iree_hal_command_buffer_update_buffer`, `fill_buffer`, `copy_buffer`, and
`dispatch` record ordered commands, and `queue_execute` replays them through
synchronous lrrt memory copies and the same lrrt launch path. Transfer and
dispatch records can resolve indirect buffer slots from the submission binding
table. The semaphore support is host-side timeline ordering for the current
synchronous queue operations, not a native device semaphore implementation.
Indirect dispatch parameters and general IREE ABI packing remain explicitly
unsupported until they are backed by lrrt runtime objects.

Set `LRRT_IREE_TRACE=1` to enable adapter-owned trace logging on `stderr`.
Tracing is opt-in and silent by default. It records the major HAL boundary
events that are useful when debugging the seamless IREE path: driver and device
creation, executable cache preparation, function lookup, buffer allocation,
queue transfer and dispatch submissions, command buffer replay, and host-side
semaphore waits/signals. The trace is diagnostic only; it must not change queue
ordering, synchronization, or normal test output.

| IREE HAL concept | Current adapter skeleton | lrrt mapping |
| --- | --- | --- |
| driver factory | `lrrt_iree_hal_driver_module_register` | Registers the `lrrt` HAL driver factory with an explicit IREE registry |
| user driver aggregate registration | `lrrt_iree_hal_register_all_available_drivers` | Registers all lrrt-provided HAL drivers with a caller-owned IREE registry |
| default registration entry point | `lrrt_iree_hal_register_all` | Idempotently registers the `lrrt` factory with IREE's default HAL driver registry for tooling-style execution |
| native HAL driver | `lrrt_iree_hal_driver_t` | Creates the minimal native HAL device for `default` |
| native HAL device | `lrrt_iree_hal_device_t` | Lifetime/id/query skeleton with an owned HAL allocator |
| native HAL allocator | `lrrt_iree_hal_allocator_t` | Owns the lrrt runtime/device lifetime needed by HAL buffers |
| native HAL buffer | `lrrt_iree_hal_buffer_t` | Wraps an `iree_hal_buffer_t` around an `lr_malloc` device allocation |
| HAL buffer mapping | `map_range`, `unmap_range`, `invalidate_range`, `flush_range` | Uses an adapter-owned host shadow allocation and `lr_memcpy` to synchronize with the lrrt device allocation |
| HAL queue update/copy | `queue_update`, `queue_copy` | Waits on lrrt-owned semaphore lists, uses synchronous `lr_memcpy` for host-to-device update and device-to-device copy, then signals completion semaphores |
| HAL queue read/write | `import_file`, `queue_read`, `queue_write` | Wraps IREE file handles and performs synchronous file-to-buffer / buffer-to-file transfers through lrrt-visible HAL buffers |
| HAL queue dispatch | `queue_dispatch` | Waits on lrrt-owned semaphore lists, packs direct HAL storage-buffer bindings as raw device pointers, submits one static dispatch with `lr_launch`, then signals completion semaphores |
| HAL command buffer execution | `command_buffer_update_buffer`, `command_buffer_fill_buffer`, `command_buffer_copy_buffer`, `command_buffer_dispatch`, `queue_execute` | Records ordered transfer and static dispatch commands, then replays them with synchronous `lr_memcpy` / `lr_launch`, including indirect binding-table resolution and lrrt-owned semaphore wait/signal lists |
| HAL semaphore | `create_semaphore`, queue wait/signal lists | Provides host-side timeline semaphore ordering for synchronous lrrt queue operations |
| native HAL executable cache | `lrrt_iree_hal_executable_cache_t` | Recognizes `rocm-hsaco` / `amdgpu-hsaco` formats and prepares raw HSACO with `lr_module_load_hsaco` |
| native HAL executable | `lrrt_iree_hal_executable_t` | Owns the prepared lrrt module and lazily resolves function names with `lr_kernel_get` |
| device / driver instance | `lrrt::executor::iree::Device` | Owns `lrrt::Runtime`, opens one `lrrt::Device`, owns one default `CommandQueue` |
| command queue / submit path | `CommandQueue` | Owns `lrrt::Queue`; dispatches with `lr_launch_on_queue_with_dependencies` |
| device allocation | `Buffer` | Owns `lrrt::DeviceBuffer`; allocates with `lr_malloc` through the C++ wrapper |
| host-device transfer | `Buffer::write`, `Buffer::read` | Uses `lr_memcpy` through `lrrt::copy_to_device` and `lrrt::copy_to_host` |
| executable object | `Executable` | Owns `lrrt::Module`; loads HSACO bytes with `lr_module_load_hsaco` |
| executable entry point | `Executable::entry_point` | Resolves a symbol with `lr_kernel_get` |
| fence / wait handle | `Fence` | Owns `lrrt::Event`; records on a queue and waits with `lr_event_synchronize` |
| unsupported HAL feature | `UnsupportedFeature` | Explicit adapter-level error, not a runtime fallback |

## Metadata Contract

The first probe does not ask lrrt to parse VMFB files. Instead,
`tools/iree_metadata_summary.py` extracts the adapter-relevant HAL/executable
anchors from IREE's `executable-targets` MLIR into a JSON summary. The
`executor/iree` layer now has matching metadata structs that describe the
contract an eventual HAL adapter should receive before it calls the lrrt
dispatcher.

The current `tools/iree_compile_probe.sh --try-vmfb` output for
`tools/iree_minimal_mul.mlir` has this shape:

| Summary field | Current value | Adapter metadata | lrrt use |
| --- | --- | --- | --- |
| `target` | `gfx1101` | `ExecutableMetadata::target` | Validate that the code object matches the selected AMD GPU. |
| `executables[0].executable` | `simple_mul_dispatch_0` | `ExecutableMetadata::executable` compatibility field | Names the HAL executable that owns the dispatch. |
| `executables[0].variant` | `rocm_hsaco_fb` | `ExecutableMetadata::variant` compatibility field | Identifies the ROCm HSACO variant that should provide the loaded image. |
| `executables[0].exports[0].symbol` | `simple_mul_dispatch_0_elementwise_4_f32` | `ExportMetadata::symbol` | Entry point name passed to `lr_kernel_get`. |
| `executables[0].exports[0].ordinal` | `0` | `ExportMetadata::ordinal` | Stable export index for adapter-side lookup within the owning executable. |
| `executables[0].exports[0].workgroup_size` | `[32, 1, 1]` | `ExportMetadata::workgroup_size` | Converted to `lr_launch_config_t::block`. |
| `executables[0].exports[0].subgroup_size` | `32` | `ExportMetadata::subgroup_size` | Validation/diagnostic metadata; lrrt does not schedule waves directly. |
| `executables[0].exports[0].bindings` | three `storage_buffer` bindings | `BindingMetadata` | Defines the buffer argument order that the adapter must pack into kernargs. |
| `executables[0].exports[0].kernel.symbol` | same as export symbol | `KernelMetadata::symbol` | Cross-checks the lowered LLVM kernel symbol. |
| `executables[0].exports[0].kernel.attributes` | `rocdl.kernel`, workgroup attributes | `KernelMetadata::attributes` | Confirms this is a ROCm kernel and records compiler launch constraints. |
| `executables[0].exports[0].dispatch` | executable, variant, symbol | `DispatchMetadata` | Connects the Stream dispatch site back to the executable export. |

The metadata structs are intentionally plain data. The adapter has a narrow
JSON loader for the metadata summary schema, but the structs do not load HSACO,
calculate tensor shapes, or own IREE runtime semantics. Their job is to make
the adapter boundary explicit:

```text
IREE executable metadata JSON
  -> parse_executable_metadata_json
  -> ExecutableMetadata / ExportMetadata / BindingMetadata
  -> lrrt::Module, lrrt::Kernel, lr_launch_config_t, packed kernargs
  -> CommandQueue::dispatch
```

The parser flattens all `executables[*].exports[*]` into
`ExecutableMetadata::exports` for the existing adapter helpers. When a VMFB has
more than one HAL executable, `ExportMetadata::dispatch.executable` records the
specific executable that owns each export.

For the current minimal multiply probe, the binding layout is:

| Binding index | Type | Flags | Expected role |
| --- | --- | --- | --- |
| `0` | `storage_buffer` | `ReadOnly`, `Indirect` | input buffer A |
| `1` | `storage_buffer` | `ReadOnly`, `Indirect` | input buffer B |
| `2` | `storage_buffer` | `Indirect` | output buffer |

The adapter still needs a real IREE HAL binding step to turn IREE buffer views
and dispatch records into these metadata and kernarg values. That logic belongs
in `executor/iree`, not in the C runtime core.

The current adapter helper can pack contiguous `storage_buffer` bindings marked
`Indirect` into the pointer-only kernarg layout used by the `simple_mul` probe.
It also converts IREE-style workgroup counts into lrrt's total-grid launch
configuration using the export workgroup size.

## Buffer Mapping Semantics

The lrrt HAL driver keeps buffer storage simple: every HAL buffer owns a device
allocation from `lr_malloc`. Mapping does not return a pointer into that device
allocation. Instead, `map_range` allocates or reuses a host shadow buffer owned
by `lrrt_iree_hal_buffer_t` and returns a span inside that shadow allocation.

The current synchronization rules are:

| Operation | Current behavior |
| --- | --- |
| `map_range(..., READ, ...)` / `iree_hal_buffer_map_read` | Copies the requested device range into the host shadow with `lr_memcpy(..., LR_MEMCPY_DEVICE_TO_HOST)` before returning the mapped span. |
| `map_range(..., WRITE, ...)` / `iree_hal_buffer_map_write` | Returns a writable host shadow span. The write becomes visible to the device when the mapping is unmapped. |
| `unmap_range` for a writable mapping | Copies the written range from the host shadow back to the device allocation with `lr_memcpy(..., LR_MEMCPY_HOST_TO_DEVICE)`. |
| `invalidate_range` | Refreshes the host shadow range from device memory. |
| `flush_range` | Copies the host shadow range to device memory if a shadow allocation exists; otherwise it is a no-op. |

This is a correctness bridge for IREE tooling and small smoke tests. It is not
a zero-copy host-visible allocation model, and it should not be treated as a
performance feature. Queue transfers and dispatches still operate on the lrrt
device allocation.

## Dispatch Contract

The skeleton dispatch path is deliberately narrow:

```text
Executable::entry_point(name)
  -> lrrt::Kernel
CommandQueue::dispatch(kernel, config, args, args_size, wait_fences)
  -> lr_launch_on_queue_with_dependencies
```

The adapter caller must provide:

- the resolved kernel entry point
- `lr_launch_config_t` with total grid, workgroup size, and dynamic shared
  memory bytes
- a packed kernarg buffer matching the generated code object ABI
- optional wait fences that belong to the same adapter device

The adapter does not currently infer dispatch workgroup counts from VM bytecode,
pack general IREE ABI arguments, or parse VMFB metadata. Those responsibilities
belong to the next real IREE HAL binding step.

The native HAL driver now has the first equivalent C-level path for direct
queue dispatch and transfer/dispatch command buffer execution:

```text
iree_hal_device_queue_dispatch
  -> executable function index
  -> stored lr_kernel_t
  -> direct iree_hal_buffer_ref_list_t bindings
  -> pointer-only kernarg buffer
  -> lr_launch

iree_hal_command_buffer_dispatch
  -> record executable function index + dispatch config + buffer refs
iree_hal_command_buffer_update_buffer/fill_buffer/copy_buffer
  -> record transfer payload + direct or indirect buffer refs
iree_hal_device_queue_execute
  -> resolve indirect refs from iree_hal_buffer_binding_table_t
  -> replay transfer commands with lr_memcpy
  -> replay dispatch commands with pointer-only kernarg buffer and lr_launch
```

This path is intentionally stricter than full IREE HAL semantics. It requires
the caller to provide explicit `workgroup_size`, uses the static
`workgroup_count`, requires lrrt-owned HAL semaphores for queue wait/signal
lists, rejects constants, rejects indirect workgroup parameters, and rejects
indirect argument modes. Command buffer transfer replay is synchronous and
minimal: update copies host bytes into the command buffer at record time, copy
uses device-to-device `lr_memcpy`, and fill materializes a temporary host
pattern buffer before a host-to-device copy. Command-buffer event and collective
commands are still explicit unsupported paths. This keeps the first adapter
dispatch test tied to the actual lrrt dispatcher without pretending to support
the full VMFB execution contract yet.

## Unsupported Feature Policy

The adapter should reject unsupported HAL features explicitly at the adapter
boundary. Unsupported behavior should not silently fall back to a different
runtime path, and it should not leak IREE VM, Flow, Stream, graph scheduling, or
tensor semantics into the lrrt C runtime core.

The unsupported surface currently falls into these categories:

| Category | Boundary rule |
| --- | --- |
| Device topology and queue policy | Expose only the lrrt-backed device and queue semantics that lrrt can actually provide. |
| External ownership | Reject external memory and semaphore import/export until lrrt has a clear ownership model for those handles. |
| Direct host-visible mapping | Keep the host-shadow mapping bridge explicit; do not present it as zero-copy GPU mapping. |
| Dynamic dispatch semantics | Reject dynamic workgroup parameters, general scalar ABI packing, and shape-driven dispatch policy until the adapter has real metadata support. |
| HAL side channels | Reject channels, events, host calls, collectives, profiling, and capture paths until they are intentionally mapped to lrrt primitives. |
| Runtime-core leakage | Keep VMFB parsing, VM invocation, graph scheduling, and tensor semantics outside `include/lrrt/lrrt.h`. |

The exact current status for individual HAL entry points is tracked in
[IREE HAL Adapter Coverage](iree-hal-coverage.md). This mapping document should
describe why a concept maps or does not map to lrrt; the coverage document
should say whether the current implementation supports it today.

## Runtime Core Boundary

The adapter may request new lrrt C APIs only when they are generally useful for
low-overhead dispatch or predictable resource management. It should not add
IREE-specific types or VM/HAL concepts to `include/lrrt/lrrt.h`.

Likely acceptable future runtime gaps:

- clearer queue/event dependency validation
- target architecture or code object diagnostics
- reusable workspace allocation support
- explicit queue/event primitives needed by compiler-generated schedules

Non-runtime responsibilities:

- IREE VM invocation
- HAL executable metadata parsing
- shape refinement
- tensor graph ownership
- frontend import from PyTorch, ONNX, JAX, or TensorFlow
