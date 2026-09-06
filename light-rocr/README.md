# light-rocr

`light-rocr` is the self-authored low-level AMDGPU runtime stack used by
`light-rocm-runtime`. It will contain the AMDHSA code-object loader, runtime
primitives, and interchangeable libhsakmt and direct-KFD transports.

The first component is a host-only ELF loader. It has no dependency on ROCr,
libhsakmt, KFD, a GPU, or LLVM libraries.

The initial parser accepts the AMDGPU-HSA ABI versions observed in the current
artifact corpus: version 4 from Clang/Triton and version 3 from IREE.

## Standalone build

```sh
cmake -S light-rocr -B build-light-rocr
cmake --build build-light-rocr
ctest --test-dir build-light-rocr --output-on-failure
```

Inspect an AMDHSA code object:

```sh
./build-light-rocr/light-rocr-inspect-hsaco \
  ./build/vector_add_kernel.hsaco
```

The main project can include this subproject with
`-DLRRT_ENABLE_LIGHT_ROCR=ON`.

## Component boundary

```text
loader/       bounded AMDHSA ELF parsing and LoadPlan construction
runtime/      device, memory, queue, signal, executable, and dispatch logic
transport/    libhsakmt and direct public-KFD implementations
arch/         GPU-generation-specific layouts and operations
```

Only `loader/` exists in the first implementation unit. Empty component
directories are added when they receive their first implementation.
