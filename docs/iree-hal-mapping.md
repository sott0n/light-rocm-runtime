# IREE HAL Adapter Mapping

## Purpose

This document records the current contract between IREE HAL concepts and the
experimental lrrt-backed adapter skeleton in `executor/iree`. It is intentionally
about the adapter boundary, not the runtime core.

The adapter should translate IREE HAL-level operations into lrrt C API calls
while keeping IREE VM, Flow, Stream, graph scheduling, and frontend semantics
outside `light-rocm-runtime`.

## Current Skeleton

The current skeleton is a small C++ layer that names the adapter-owned concepts
before binding directly to real IREE HAL types. It is compiled only when
`LRRT_ENABLE_IREE_ADAPTER=ON`.

| IREE HAL concept | Current adapter skeleton | lrrt mapping |
| --- | --- | --- |
| device / driver instance | `lrrt::executor::iree::Device` | Owns `lrrt::Runtime`, opens one `lrrt::Device`, owns one default `CommandQueue` |
| command queue / submit path | `CommandQueue` | Owns `lrrt::Queue`; dispatches with `lr_launch_on_queue_with_dependencies` |
| device allocation | `Buffer` | Owns `lrrt::DeviceBuffer`; allocates with `lr_malloc` through the C++ wrapper |
| host-device transfer | `Buffer::write`, `Buffer::read` | Uses `lr_memcpy` through `lrrt::copy_to_device` and `lrrt::copy_to_host` |
| executable object | `Executable` | Owns `lrrt::Module`; loads HSACO bytes with `lr_module_load_hsaco` |
| executable entry point | `Executable::entry_point` | Resolves a symbol with `lr_kernel_get` |
| fence / wait handle | `Fence` | Owns `lrrt::Event`; records on a queue and waits with `lr_event_synchronize` |
| unsupported HAL feature | `UnsupportedFeature` | Explicit adapter-level error, not a runtime fallback |

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

The adapter does not currently infer launch dimensions, pack IREE ABI
arguments, or parse VMFB metadata. Those responsibilities belong to the next
real IREE HAL binding step.

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
