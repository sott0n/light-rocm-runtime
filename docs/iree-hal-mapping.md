# IREE HAL Adapter Mapping

## Purpose

This document records the current contract between IREE HAL concepts and the
experimental lrrt-backed adapter skeleton in `executor/iree`. It is intentionally
about the adapter boundary, not the runtime core.

The adapter should translate IREE HAL-level operations into lrrt C API calls
while keeping IREE VM, Flow, Stream, graph scheduling, and frontend semantics
outside `light-rocm-runtime`.

## Current Skeleton

The current skeleton has two layers and is compiled only when
`LRRT_ENABLE_IREE_ADAPTER=ON`:

- a small C++ adapter layer that names the lrrt-owned concepts used by the
  existing metadata/HSACO dispatch probes
- a minimal native IREE HAL driver factory named `lrrt`, registered through
  `lrrt_iree_hal_driver_module_register`

The native driver factory is the first step toward a seamless
`iree-run-module --device=lrrt` path. At this stage it can be registered in an
IREE HAL driver registry, created by name, and queried for one placeholder
device. It can also create a minimal lrrt-backed `iree_hal_device_t` with a
stable id, host allocator, empty capabilities, and a device-owned HAL allocator
skeleton. The allocator currently exposes lifetime/query methods only:
compatibility reports `IREE_HAL_BUFFER_COMPATIBILITY_NONE`, buffer allocation
returns `IREE_STATUS_UNIMPLEMENTED`, and virtual memory reports unavailable.
Executable cache, command buffers, semaphores, and queue submission APIs also
remain explicitly unsupported until they are backed by lrrt runtime objects.

| IREE HAL concept | Current adapter skeleton | lrrt mapping |
| --- | --- | --- |
| driver factory | `lrrt_iree_hal_driver_module_register` | Registers the `lrrt` HAL driver factory with an IREE registry |
| native HAL driver | `lrrt_iree_hal_driver_t` | Creates the minimal native HAL device for `default` |
| native HAL device | `lrrt_iree_hal_device_t` | Lifetime/id/query skeleton with an owned HAL allocator |
| native HAL allocator | `lrrt_iree_hal_allocator_t` | Lifetime/query skeleton only; real buffer allocation is not implemented yet |
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
| `executable` | `simple_mul_dispatch_0` | `ExecutableMetadata::executable` | Names the HAL executable that owns the dispatch. |
| `variant` | `rocm_hsaco_fb` | `ExecutableMetadata::variant` | Identifies the ROCm HSACO variant that should provide the loaded image. |
| `exports[0].symbol` | `simple_mul_dispatch_0_elementwise_4_f32` | `ExportMetadata::symbol` | Entry point name passed to `lr_kernel_get`. |
| `exports[0].ordinal` | `0` | `ExportMetadata::ordinal` | Stable export index for adapter-side lookup. |
| `exports[0].workgroup_size` | `[32, 1, 1]` | `ExportMetadata::workgroup_size` | Converted to `lr_launch_config_t::block`. |
| `exports[0].subgroup_size` | `32` | `ExportMetadata::subgroup_size` | Validation/diagnostic metadata; lrrt does not schedule waves directly. |
| `exports[0].bindings` | three `storage_buffer` bindings | `BindingMetadata` | Defines the buffer argument order that the adapter must pack into kernargs. |
| `exports[0].kernel.symbol` | same as export symbol | `KernelMetadata::symbol` | Cross-checks the lowered LLVM kernel symbol. |
| `exports[0].kernel.attributes` | `rocdl.kernel`, workgroup attributes | `KernelMetadata::attributes` | Confirms this is a ROCm kernel and records compiler launch constraints. |
| `exports[0].dispatch` | executable, variant, symbol | `DispatchMetadata` | Connects the Stream dispatch site back to the executable export. |

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

## Initial Unsupported Features

The first adapter prototype should reject these features explicitly:

- multiple devices
- multiple independent hardware queues as a required semantic
- external memory import/export
- host-visible mapped device allocations
- timeline semaphores
- full IREE command buffer optimization
- dynamic shape dispatch policy
- VMFB parsing in the lrrt runtime core
- graph-level scheduling or tensor semantics

Unsupported features should fail at the adapter boundary using
`UnsupportedFeature` or an equivalent IREE status once real IREE HAL types are
introduced.

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
