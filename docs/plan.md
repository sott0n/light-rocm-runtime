# Plan

## Goal

Build a small ROCm-oriented runtime that can run AMD GPU kernels without pulling
in the full HIP runtime surface.

The runtime should first prove a single end-to-end path:

1. initialize the runtime
2. find an AMD GPU agent
3. allocate GPU-accessible memory
4. load a `.hsaco` code object
5. resolve a kernel symbol
6. dispatch the kernel
7. synchronize
8. copy the result back

## Non-goals for the first version

- Full HIP API compatibility
- Graphs
- Stream priority
- Peer access
- Full multi-GPU scheduling
- Math/library replacement
- Direct `/dev/kfd` ioctl backend

## Milestones

### M0: Repository scaffold

- Public C API in `include/lrrt/lrrt.h`
- CMake build
- Stub runtime library
- `vector_add` example source layout

### M1: HSA-backed vector add

- Initialize/shutdown HSA
- Select the first GPU agent
- Create one HSA queue
- Allocate/copy memory through HSA-compatible mechanisms
- Load a `.hsaco`
- Dispatch `vector_add`
- Validate output on CPU

### M2: Runtime structure

- Split device, memory, module, and queue internals
- Add stable error reporting
- Add basic device properties
- Add GPU-free unit tests for argument validation and lifecycle behavior

### M3: Practical minimum

- Async launch handle or stream-like queue object
- Multiple kernels per module
- Kernel argument packing helper
- Better diagnostics for missing ROCm/HSA pieces

### M4: Optional compatibility layer

- Consider a small HIP-like wrapper only after the native runtime API is stable
- Keep it as a separate header/library so it does not define the core design
