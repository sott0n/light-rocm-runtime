# IREE Artifact Extraction Notes

## Purpose

This note records how the current IREE ROCm probe can produce an artifact that
`light-rocm-runtime` can load through `lr_module_load_hsaco`. The important
result is that the IREE baseline VMFB and the lrrt-loadable HSACO should be
treated as two related but separate artifacts.

## Observed Artifact Shapes

The default probe path produces a VMFB that IREE's HIP HAL runtime can execute:

```sh
tools/iree_compile_probe.sh --run-baseline
```

The VMFB is a polyglot zip archive. For the current `simple_mul` probe it
contains:

```text
module.fb
simple_mul_dispatch_0_rocm_hsaco_fb.fb
```

`module.fb` is IREE VM bytecode and host-side HAL command logic. The
`simple_mul_dispatch_0_rocm_hsaco_fb.fb` member is an IREE executable variant
payload for the ROCm backend.

With the default ROCm container, that variant payload is a HIP executable
flatbuffer. It contains a `HIP1` header and wraps the AMDGPU code object. IREE's
HIP runtime expects this shape when loading the VMFB.

## Raw HSACO Path

For lrrt, the useful artifact is the raw HSACO ELF image, not the HIP executable
flatbuffer. IREE can emit that directly from the generated
`executable-targets` MLIR:

```sh
build-iree-tools/tools/iree-compile \
  --compile-mode=hal-executable \
  --iree-rocm-target=gfx1101 \
  --iree-rocm-container-type=hsaco \
  build-iree-probe/minimal_mul_gfx1101_executable_targets.mlir \
  -o build-iree-probe/minimal_mul_gfx1101.hsaco
```

`tools/iree_compile_probe.sh --emit-hsaco` now runs this step after generating
the metadata summary:

```sh
tools/iree_compile_probe.sh --emit-hsaco
```

The emitted file is an AMDGPU HSA ELF code object. On the current development
machine it has:

```text
OS/ABI: AMDGPU - HSA
ABI Version: 3
Machine: EM_AMDGPU
Flags: gfx1101
```

It exports the same symbol recorded in the metadata summary:

```text
simple_mul_dispatch_0_elementwise_4_f32
```

That symbol is the value the adapter should pass to `lr_kernel_get` after
loading the HSACO through `lr_module_load_hsaco`.

## Why Not Use Raw-HSACO VMFB As The Baseline

IREE also accepts `--iree-rocm-container-type=hsaco` when compiling a full VMFB,
and the resulting zip member is a raw ELF image. However, that VMFB is not a
valid IREE HIP runtime baseline in the current setup: `iree-run-module
--device=hip` expects the ROCm executable payload to have the HIP executable
flatbuffer header, and fails when the payload is raw HSACO.

Therefore the validation split is:

| Artifact | Producer | Consumer | Purpose |
| --- | --- | --- | --- |
| default VMFB | `tools/iree_compile_probe.sh --run-baseline` | IREE HIP HAL runtime | Reference execution and expected output. |
| metadata JSON | `tools/iree_metadata_summary.py` | lrrt IREE adapter layer | Dispatch metadata contract. |
| raw HSACO | `tools/iree_compile_probe.sh --emit-hsaco` | lrrt module loader | Code object loading experiment. |

## Adapter Implication

The first lrrt-backed prototype should not parse VMFB in the runtime core. It
can start from three explicit inputs:

- raw HSACO bytes
- `ExecutableMetadata` from the metadata summary
- caller-provided buffers and packed kernargs

This keeps the lrrt runtime focused on low-overhead dispatch and predictable
resource management while leaving VMFB ownership and full HAL semantics to the
future adapter layer.
