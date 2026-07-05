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
cmake --build build-iree-tools --target iree-compile iree-run-module -j2
```

Then configure lrrt with the adapter enabled:

```sh
cmake -S . -B build-iree \
  -DLRRT_ENABLE_IREE_ADAPTER=ON \
  -DLRRT_IREE_ROOT=third_party/iree
```

Add `build-iree-tools/tools` to `PATH` or pass the explicit tool paths in the
environment used by future validation tests. The current adapter skeleton only
requires headers, so missing tools are reported as CMake status messages.

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

## Assumptions

- The pinned submodule remains the source of truth for IREE headers.
- `iree-compile` and `iree-run-module` are optional for skeleton builds.
- Tool build directories such as `build-iree-tools` are local artifacts and
  should not be tracked.
- Full IREE package production is out of scope for this repository.
