# SparseWave Executor Integration

## Responsibility Boundary

SparseWave owns sparse operator semantics, storage-format lowering, work
mapping, and AMDGPU code generation. The SparseWave executor owns tensor
buffers, workspaces, and multi-kernel scheduling. The lrrt core owns only
HSACO loading, memory, queues, dispatch, and synchronization.

SparseWave integration must not add CSR formats, SpMM semantics, attention
semantics, or compiler passes to the runtime core.

## Dependency Policy

The compatible SparseWave source revision is pinned as an opt-in submodule at
`third_party/sparsewave`. Normal lrrt builds do not initialize or build it.
SparseWave-enabled builds may use the pinned source or an explicitly selected
external checkout through `LRRT_SPARSEWAVE_ROOT`.

SparseWave's own LLVM submodule is a compiler build dependency. It is not
recursively initialized by the default lrrt build.

Updating the pinned SparseWave revision is an integration change. Bundle
generation and end-to-end executor tests must pass before the submodule pointer
is updated.

## Compiler Build

Initialize the pinned compiler and its LLVM dependency explicitly:

```sh
git submodule update --init third_party/sparsewave
git -C third_party/sparsewave submodule update --init --recursive
```

Build SparseWave with its pinned LLVM revision:

```sh
cmake -G Ninja \
  -S third_party/sparsewave/externals/llvm-project/llvm \
  -B build-sparsewave/llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=sparsewave \
  -DLLVM_EXTERNAL_SPARSEWAVE_SOURCE_DIR="$PWD/third_party/sparsewave" \
  -DLLVM_TARGETS_TO_BUILD="AMDGPU;Native" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-sparsewave/llvm \
  --target sparsewave-opt sparsewave-bundle check-sparsewave
```

This builds `build-sparsewave/llvm/bin/sparsewave-opt` and
`build-sparsewave/llvm/bin/sparsewave-bundle`, then runs the compiler's
regression suite. Tests that require PyTorch or MLIR's HIP runtime wrapper are
optional compiler-project features; lrrt executor correctness is covered by
the separate end-to-end paths in this repository.

## SpMM Executor Build

Enable the opt-in executor after building SparseWave:

```sh
cmake -S . -B build-sparsewave/e2e \
  -DLRRT_ENABLE_SPARSEWAVE_EXECUTOR=ON \
  -DLRRT_SPARSEWAVE_BUILD_DIR="$PWD/build-sparsewave/llvm"
cmake --build build-sparsewave/e2e --target lrrt_sparsewave_spmm
ctest --test-dir build-sparsewave/e2e -R lrrt_sparsewave_spmm_e2e \
  --output-on-failure
```

The build uses SparseWave's locked PyTorch environment to capture
`torch.sparse.mm` with `torch.export` and import the raw graph through
torch-mlir. The raw Torch MLIR is passed directly to `sparsewave-bundle`, which
owns Torch-to-SparseWave lowering, dynamic-CSR bare-pointer preparation, HSACO
generation, and manifest verification. The runtime integration does not
rewrite MLIR or supply an NNZ specialization.

The first build uses `uv` to install the exact PyTorch and torch-mlir versions
from SparseWave's lockfile into the build directory. Set
`LRRT_SPARSEWAVE_ROOT` and `LRRT_SPARSEWAVE_BUILD_DIR` to use an explicitly
selected external compiler checkout and its matching build tree.

## Bundle Contract

The compiler produces a standard lrrt bundle:

```text
bundle/
  manifest.json
  kernels.hsaco
```

The bundle follows `docs/manifest-schema.md`. SparseWave-specific operator,
tensor, storage-format, and scheduling information stays outside the runtime
manifest unless it is required to dispatch a kernel safely.

For every kernel, the producer supplies:

- the AMDGPU target and raw HSACO code object;
- a stable logical name and the exact code-object symbol;
- explicit argument names, logical types, offsets, and sizes;
- the complete kernarg segment size;
- block dimensions and a manifest-supported grid expression;
- dynamic shared-memory and executor workspace requirements.

The producer derives logical argument types from compiler IR. HSACO metadata
alone cannot distinguish all same-sized by-value types.

## Initial Executor Scope

The first supported configuration is deliberately narrow:

- `gfx1101` with Wave32;
- CSR with 32-bit row offsets and column indices;
- FP32 values, dense operands, and output;
- compiler-specialized tensor shapes;
- CSR SpMM before the multi-dispatch SparseAttention path.

Broader targets, data types, index widths, formats, and dynamic shapes require
their own bundle and executor compatibility coverage.

## Revision Updates

The submodule pin is the compiler compatibility boundary. A SparseWave revision
update must regenerate its bundles and pass the supported SpMM and
SparseAttention end-to-end tests. Separate metadata compatibility tooling is
not maintained for routine builds.

When an update intentionally changes the generated kernel ABI or launch shape,
review the produced manifest and code-object metadata as part of that update.
If the new launch cannot be represented by the current manifest, extend and
test the manifest contract before updating the executor.
