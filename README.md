# light-rocm-runtime

A small experimental runtime for launching AMD GPU kernels from ROCm code
objects.

The first target is intentionally narrow:

- Linux with AMDGPU and an installed ROCm stack
- load a pre-built `.hsaco`
- allocate device memory
- copy host/device buffers
- launch one kernel
- synchronize and copy results back

This project is not trying to replace the full HIP runtime. The initial
implementation will use the HSA/ROCr API directly and keep a small public C ABI.

## Build

```sh
cmake -S . -B build
cmake --build build
```

The current runtime is a scaffold. HSA-backed execution will be added behind the
same public API.

## Layout

```text
include/lrrt/lrrt.h      public C API
src/runtime.cpp          runtime state and API entry points
examples/vector_add/     first end-to-end target
docs/plan.md             milestone plan
```
