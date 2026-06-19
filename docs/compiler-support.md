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
  "target": "gfx1101",
  "kernels": [
    {
      "name": "rmsnorm",
      "symbol": "rmsnorm.kd",
      "args": [
        {"name": "x", "type": "ptr"},
        {"name": "weight", "type": "ptr"},
        {"name": "out", "type": "ptr"},
        {"name": "n", "type": "i32"}
      ],
      "block": [256, 1, 1],
      "grid": ["ceil_div(n, 256)", 1, 1],
      "workspace_bytes": 0
    }
  ]
}
```

The initial manifest does not need to be a general graph IR. It only needs to
describe enough information for an executor to bind arguments and dispatch
kernels safely.

### Current manifest schema

The current Triton examples use a narrow JSON schema that describes one or more
independent kernel entries. The schema is intentionally launch-oriented: it
describes the code object, symbol, kernarg layout, and launch shape needed by an
lrrt-based executor.

Top-level fields:

- `target`: AMDGPU target architecture, such as `gfx1101`. This must match the
  code object target reported by the HSACO metadata.
- `kernels`: array of kernel entries. Each entry describes one dispatchable
  kernel inside a code object.

Kernel entry fields:

- `name`: stable logical name used by examples or an executor.
- `symbol`: kernel entry symbol to resolve with `lrrt::Module::kernel`. The
  generated HSACO may also contain the descriptor symbol with a `.kd` suffix,
  but the lrrt lookup path accepts the entry symbol name.
- `code_object`: relative path to the HSACO file inside the bundle directory.
- `args`: ordered kernarg ABI description. Each argument includes:
  - `name`: logical argument name.
  - `type`: compact type label such as `ptr`, `i32`, or `fp32`.
  - `offset`: byte offset in the packed kernarg buffer.
  - `size`: byte size in the packed kernarg buffer.
  - `optional`: optional boolean for compiler-internal arguments that may be
    bound to null by small examples, such as Triton scratch pointers.
- `kernarg_size`: total byte size of the packed kernarg buffer.
- `block`: HSA workgroup size `[x, y, z]` used as `lr_launch_config_t.block`.
- `grid`: manifest expression for deriving the HSA total grid size
  `[x, y, z]`.
- `shared_memory_bytes`: optional dynamic shared-memory requirement passed to
  `lr_launch_config_t.shared_memory_bytes`. Kernels that perform multi-warp
  reductions may require this even when the HSACO fixed group segment is zero.
- `triton`: optional producer metadata such as Triton version, block size, and
  number of warps. This is useful for debugging and consistency checks, but it
  is not required by the runtime core.
- `workspace_bytes`: optional per-dispatch workspace requirement. The current
  examples use `0`.

Important launch convention: lrrt's `grid` is the HSA total grid size, not the
Triton program count. For the current Triton examples, the manifest writes
`grid[0]` as an expression like `ceil_div(n, 256) * 128`, where `ceil_div(n,
256)` is the Triton program count and `128` is the workgroup size. The
example-side launcher converts that expression into `lr_launch_config_t.grid`.

The manifest is an ABI description, not a tensor or operator IR. It should not
encode high-level graph semantics, tensor ownership, scheduling policy, or
framework-specific concepts. Those belong in a compiler or executor layer above
the runtime.

### Consistency checks

The Triton example build verifies that `manifest.json` matches HSACO metadata:

- `target` matches `amdhsa.target`
- `symbol` matches the kernel metadata name and descriptor symbol
- `kernarg_size` matches `.kernarg_segment_size`
- `block[0]` matches `.max_flat_workgroup_size`
- each manifest argument offset and size matches the HSACO argument metadata

These checks are intentionally outside the runtime core. They protect the
example bundle ABI from Triton version drift while keeping `light-rocm-runtime`
focused on loading code objects, managing resources, dispatching kernels, and
synchronizing work.

## Milestones

### M0: Hand-written bundle

- Build a hand-written HIP kernel into `kernels.hsaco`.
- Write a matching `manifest.json` by hand.
- Load the bundle from a small example and dispatch one kernel through lrrt.

### M1: Manifest-driven launcher

- Add an example-side launcher that reads the manifest.
- Resolve kernel symbols from manifest entries.
- Bind arguments explicitly from host-side values and `DeviceBuffer` handles.
- Keep manifest parsing outside the runtime core until the ABI shape settles.

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
- Prototype either an lrrt executor mapping or a HAL-style adapter.
- Decide whether IREE integration belongs in this repository or a companion
  compiler/executor project.

## Non-goals

- Do not implement a compiler inside the runtime core.
- Do not absorb PyTorch, Triton, or IREE runtime responsibilities.
- Do not make the runtime own high-level tensor graph semantics yet.
- Do not make the kernel bundle ABI depend on a single frontend.
- Do not require async launch, streams, or graph execution for the first bundle
  prototype.
