# IREE Tool Build Strategy

## Purpose

This document defines how this repository should obtain IREE tools for adapter
validation. The lrrt build should use the IREE submodule for headers, but it
should not build IREE tools as part of the default lrrt build.

The tools are needed for later validation stages:

- `iree-compile` to compile a tiny program for the ROCm/HIP target
- `iree-run-module` to establish a baseline through IREE's own HIP HAL runtime

## Build Boundary

`light-rocm-runtime` owns:

- the lrrt C/HSA runtime
- the opt-in `executor/iree` adapter skeleton
- CMake probing for IREE headers and already-built tools

IREE owns:

- building `iree-compile`
- building `iree-run-module`
- compiler target backends such as ROCm
- runtime HAL drivers such as HIP

The lrrt CMake build must not automatically configure or build the IREE
submodule. IREE is large and may build bundled LLVM and target backends. Tool
builds should happen explicitly in a separate build directory through the helper
script or equivalent manual commands.

## Recommended Local Build

Initialize the pinned IREE submodule:

```sh
git submodule update --init third_party/iree
```

Build the required tools with the helper script:

```sh
tools/build_iree_tools.sh
```

By default this uses `third_party/iree`, writes to `build-iree-tools`, enables
the ROCm compiler backend and HIP HAL driver, and builds only `iree-compile`
and `iree-run-module`.

The equivalent manual configure step is:

```sh
cmake -S third_party/iree -B build-iree-tools \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_BENCHMARKS=OFF \
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_ROCM=ON \
  -DIREE_HAL_DRIVER_HIP=ON \
  -DIREE_ROCM_TEST_TARGET_CHIP=
```

Build only the tools needed by the first adapter validation path:

```sh
cmake --build build-iree-tools --target iree-compile iree-run-module -j16
```

On the development machine, `-j32` reached the same build graph but crashed the
host clang 17 frontend while compiling bundled LLVM. `-j16` completed the tool
build with the Ninja generator and is the recommended local setting.

Then configure lrrt with the adapter enabled:

```sh
cmake -S . -B build-iree \
  -DLRRT_ENABLE_IREE_ADAPTER=ON \
  -DLRRT_IREE_ROOT=third_party/iree
```

Add `build-iree-tools/tools` to `PATH` or pass the explicit tool paths in the
environment used by future validation tests. The current adapter skeleton only
requires headers for metadata/HSACO probes. The native HAL driver registration
test also uses the IREE runtime static libraries produced by
`tools/build_iree_tools.sh`; if those libraries are absent, that specific test
is skipped while the header-only adapter tests remain available.

## Why Not Build IREE From lrrt CMake

Do not call `add_subdirectory(third_party/iree)` from the lrrt build.

Reasons:

- it would make `cmake --build build` much heavier
- IREE may build bundled LLVM and many compiler/runtime targets
- ROCm backend and HIP driver options require explicit local choices
- default lrrt users should not need IREE tools
- later validation can depend on prebuilt tool paths without changing runtime
  core behavior

Use `tools/build_iree_tools.sh` when the tools are needed. The explicit script
keeps the dependency visible while avoiding hidden work in the lrrt configure or
build graph.

## Validation Direction

The first real validation should use the built tools as external executables:

1. compile a tiny static-shape program with `iree-compile`
2. run it with `iree-run-module --device=hip` as the IREE baseline
3. use the lrrt-backed adapter path for the same dispatch once the HAL binding
   exists
4. compare outputs against a CPU reference or the IREE HIP baseline

Until the HAL binding exists, lrrt should only check that the adapter can find
IREE headers and optionally report available tools.

The current compile probe is:

```sh
tools/iree_compile_probe.sh
```

It writes ignored artifacts under `build-iree-probe/` and prints the HAL
executable metadata anchors that matter to the lrrt adapter investigation. Use
`--try-vmfb` when checking whether the local IREE/LLD toolchain can serialize a
full VMFB. The probe automatically prefers `/opt/rocm/llvm/bin/lld` when it is
available; pass `--lld-dir` to use a different LLD directory.

Use `--run-baseline` when the local GPU should also run the serialized VMFB
through IREE's HIP HAL runtime:

```sh
tools/iree_compile_probe.sh --run-baseline
```

This option implies `--try-vmfb`, calls `iree-run-module --device=hip`, and
checks the default `tools/iree_minimal_mul.mlir` output:

```text
4xf32=10 40 90 160
```

The baseline run writes stdout/stderr to
`build-iree-probe/minimal_mul_<target>_baseline.log`. A mismatch or runner
failure makes the probe exit non-zero so it can be used as a local validation
step before comparing an lrrt-backed adapter path.

When lrrt is configured with `LRRT_ENABLE_IREE_ADAPTER=ON` and both
`iree-compile` and `iree-run-module` are found, the same baseline is also
available as an opt-in CTest:

```sh
ctest --test-dir build-iree/adapter --output-on-failure -R lrrt_iree_baseline_probe
```

The probe also writes `minimal_mul_<target>_metadata.json` by running
`tools/iree_metadata_summary.py` on the generated `executable-targets` MLIR.
This gives later adapter work a stable, reviewable view of the exported
dispatch metadata without parsing VMFB files in the lrrt runtime core. The
summary schema is documented in
[IREE Metadata Summary Schema](iree-metadata-schema.md).

Use `--emit-hsaco` when the lrrt adapter investigation needs the matching raw
AMDGPU code object:

```sh
tools/iree_compile_probe.sh --emit-hsaco
```

This writes `minimal_mul_<target>.hsaco` under `build-iree-probe/` by compiling
a raw-HSACO VMFB with `--iree-rocm-container-type=hsaco` and extracting its
ROCm executable payload. The artifact split is documented in
[IREE Artifact Extraction Notes](iree-artifact-extraction.md).

## Assumptions

- The pinned submodule remains the source of truth for IREE headers.
- `iree-compile` and `iree-run-module` are optional for skeleton builds.
- Tool build directories such as `build-iree-tools` are local artifacts and
  should not be tracked.
- Full IREE package production is out of scope for this repository.
