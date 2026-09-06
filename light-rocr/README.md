# light-rocr

`light-rocr` is the self-authored low-level AMDGPU runtime stack used by
`light-rocm-runtime`. It will contain the AMDHSA code-object loader, runtime
primitives, and interchangeable libhsakmt and direct-KFD transports.

The first component is a host-only ELF and AMDHSA metadata loader. It has no
dependency on ROCr, libhsakmt, KFD, a GPU, libelf, or LLVM libraries.

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

The inspector reports checked `PT_LOAD` records and, when an AMDGPU metadata
note is present, the metadata version, target ISA, kernel inventory, ELF
`.kd` symbol addresses, code entry offsets, segment sizes, wavefront size, and
raw kernel-descriptor resource flags. It reports both the metadata kernarg
alignment and the public alignment, whose minimum is normalized to 16 bytes to
match the observed ROCr executable property.

The initial executable subset is intentionally narrow: metadata version 1.2,
the two observed canonical `gfx1101` target strings, string-keyed MessagePack
maps, ELF64 dynamic symbols discoverable through `PT_DYNAMIC` and `DT_HASH`,
and 64-byte AMDHSA kernel descriptors. Section headers are not required.
Unknown MessagePack types, malformed/duplicate notes or dynamic tables,
inconsistent metadata and descriptors, reserved descriptor bits, and
out-of-range symbols or code entries are rejected. ELF objects without an
AMDGPU metadata note remain parseable and produce an empty kernel inventory; a
later load-plan API decides whether an executable operation requires kernels.

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
