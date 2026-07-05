# Compiler Support Plan

## Direction

`light-rocm-runtime` should remain a small runtime for executing AMD GPU code
objects. Compiler layers should produce GPU code objects and launch metadata,
while the runtime provides the execution substrate:

- load AMDGPU `.hsaco` code objects
- resolve kernel symbols
- allocate and copy device memory
- launch kernels from structured metadata
- synchronize execution

Compiler support should therefore mean supporting a small kernel bundle ABI. It
should not mean embedding a full ML framework, graph compiler, or operator
runtime into the core library.

## Compiler Targets

### Triton first

Triton is the first compiler target because it maps well to individual ML
kernels and operators. It is a practical path for experimenting with kernels
such as reductions, elementwise fusion, normalization, small GEMM variants, and
attention-related building blocks.

The first Triton integration target is:

1. compile a Triton kernel for the local AMDGPU target
2. extract or materialize the generated HSACO
3. describe the kernel launch in a manifest
4. load the HSACO with `lrrt::Module`
5. resolve the kernel with `lrrt::Kernel`
6. launch it with `lrrt::launch`

This keeps Triton responsible for code generation and keeps
`light-rocm-runtime` responsible for execution.

The in-repository Triton examples are opt-in through
`LRRT_ENABLE_TRITON_EXAMPLES=ON`. They use `uv` to install the pinned Triton
dependency from `examples/triton/requirements.txt` and compile bundles with
Python 3.13 by default through `LRRT_TRITON_PYTHON`. The default lrrt build does
not create Triton bundle targets or download Triton dependencies.

### IREE next

IREE is the second compiler target because it is better suited for whole graph
and model lowering. It can represent larger programs than a single custom
kernel and has an existing ROCm/HIP path.

The first IREE integration target is exploratory:

- study IREE's ROCm/HIP output artifacts
- identify the dispatch metadata needed by an external executor
- evaluate whether the mapping should be a lightweight lrrt executor or a
  HAL-style adapter
- keep IREE's graph, scheduling, and framework integration responsibilities out
  of the lrrt core

## Kernel Bundle ABI

A compiler-produced bundle should be small and explicit:

```text
bundle/
  kernels.hsaco
  manifest.json
```

The manifest should describe the execution contract that the runtime needs:

```json
{
  "manifest_version": 1,
  "target": "gfx1101",
  "kernels": [
    {
      "name": "rmsnorm",
      "symbol": "rmsnorm_kernel",
      "code_object": "kernels.hsaco",
      "args": [
        {"name": "x", "type": "ptr", "offset": 0, "size": 8},
        {"name": "weight", "type": "ptr", "offset": 8, "size": 8},
        {"name": "out", "type": "ptr", "offset": 16, "size": 8},
        {"name": "n", "type": "i32", "offset": 24, "size": 4}
      ],
      "kernarg_size": 32,
      "block": [256, 1, 1],
      "grid": ["ceil_div(n, 256) * 256", 1, 1],
      "workspace_bytes": 0
    }
  ]
}
```

The initial manifest does not need to be a general graph IR. It only needs to
describe enough information for an executor to bind arguments and dispatch
kernels safely.
Workspace requirements should stay in the manifest as dispatch metadata. A
higher-level executor can allocate and bind that temporary memory while the
runtime core remains focused on loading, dispatch, synchronization, and resource
handles.

### Schema reference

The current manifest schema is documented separately in
[Bundle Manifest Schema](manifest-schema.md). Keep schema-level validation
rules there so the roadmap can stay focused on compiler integration direction.

## Milestones

### M0: Hand-written bundle

- Build a hand-written HIP kernel into `kernels.hsaco`.
- Write a matching `manifest.json` by hand.
- Load the bundle from a small example and dispatch one kernel through lrrt.

### M1: Manifest-driven launcher

- Add the `lrrt::bundle` C++ layer that reads the manifest.
- Resolve kernel symbols from manifest entries.
- Bind arguments explicitly from host-side values and `DeviceBuffer` handles.
- Keep manifest parsing separate from the C/HSA runtime core.

### M2: Triton-generated HSACO

- Compile a simple Triton kernel for the development target.
- Package the generated HSACO with a hand-written manifest.
- Run the Triton-generated kernel through the same manifest-driven lrrt
  launcher.

### M3: Qwen-oriented operator set

- Define the smallest useful operator set for a Qwen-style inference path.
- Start with standalone kernels such as RMSNorm, elementwise activation,
  residual add, RoPE, and small matmul experiments.
- Treat each operator as one or more kernel bundle entries.

### M4: IREE investigation

- Compile a small model or operator graph through IREE's ROCm/HIP path.
- Inspect generated code objects and dispatch metadata.
- Prototype a narrow lrrt-backed HAL-style adapter.
- Decide whether IREE integration belongs in this repository or a companion
  compiler/executor project.

The detailed investigation plan is tracked in
[IREE Integration Investigation](iree-integration.md).

## Non-goals

- Do not implement a compiler inside the runtime core.
- Do not absorb PyTorch, Triton, or IREE runtime responsibilities.
- Do not make the runtime own high-level tensor graph semantics yet.
- Do not make the kernel bundle ABI depend on a single frontend.
- Do not require async launch, streams, or graph execution for the first bundle
  prototype.
