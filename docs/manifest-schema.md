# Bundle Manifest Schema

`light-rocm-runtime` uses a small JSON manifest to describe compiler-generated
kernel bundles. The manifest is the ABI boundary between a compiler or executor
layer and the `lrrt::bundle` launcher. It describes what the runtime needs to
load a code object, resolve a kernel symbol, pack kernel arguments, derive the
HSA launch shape, and pass dynamic shared-memory requirements.

The manifest is not a graph IR, tensor IR, or operator description. It should
not encode framework semantics, scheduling policy, tensor ownership, or
compiler internals that are not needed to dispatch a kernel.

## Bundle Layout

A bundle is a directory that contains one manifest and one or more AMDGPU code
objects:

```text
bundle/
  manifest.json
  kernels.hsaco
```

Multiple kernel entries may reference the same `kernels.hsaco`, or they may
reference different code objects in the same bundle directory.

## Example

```json
{
  "target": "gfx1101",
  "kernels": [
    {
      "name": "vector_add",
      "symbol": "vector_add_kernel",
      "code_object": "kernels.hsaco",
      "args": [
        {"name": "x", "type": "ptr", "offset": 0, "size": 8},
        {"name": "y", "type": "ptr", "offset": 8, "size": 8},
        {"name": "out", "type": "ptr", "offset": 16, "size": 8},
        {"name": "n", "type": "i32", "offset": 24, "size": 4}
      ],
      "kernarg_size": 32,
      "block": [128, 1, 1],
      "grid": ["ceil_div(n, 256) * 128", 1, 1],
      "shared_memory_bytes": 0,
      "workspace_bytes": 0
    }
  ]
}
```

## Top-Level Fields

### `target`

Required string. AMDGPU target architecture for the bundle, such as `gfx1101`.

The value should match the target reported by the HSACO metadata. The manifest
parser requires the field to be present and non-empty. Consistency with the
HSACO target is checked by bundle consistency tests, not by the runtime core.

### `kernels`

Required array. Each item describes one dispatchable kernel entry. Kernel names
must be unique within the manifest.

`lrrt::Bundle(device, manifest_path)` selects the first kernel entry.
`lrrt::Bundle(device, manifest_path, kernel_name)` selects an entry by `name`.

## Kernel Fields

### `name`

Required string. Stable logical kernel name used by examples, tests, or a
higher-level executor.

### `symbol`

Required string. Kernel entry symbol to resolve from the loaded code object.
This is the symbol passed to `lrrt::Module::kernel`.

### `code_object`

Required string. Relative path to the HSACO file inside the bundle directory.

The path must stay within the bundle. Absolute paths and paths containing a
`..` component are rejected.

### `args`

Required array. Ordered kernarg ABI description. The order should match the
packed kernel argument layout used by the generated code object.

Each argument entry uses these fields:

- `name`: logical argument name.
- `type`: compact type label, such as `ptr`, `i32`, `fp32`, `fp16`, or `bf16`.
- `offset`: byte offset in the packed kernarg buffer.
- `size`: byte size in the packed kernarg buffer.
- `optional`: optional boolean for compiler-internal arguments that may be
  bound to a default value by small examples.

The current parser requires each entry to contain `offset`. Offsets must be
strictly increasing and each offset must be smaller than `kernarg_size`.
Argument type and size consistency is checked by bundle consistency tests
against HSACO metadata.

`lrrt::KernargBuffer` can use this metadata to pack values by argument name or
argument index without requiring the caller to define a matching C++ struct. The
helper still uses raw ABI bytes: it does not interpret tensor types, allocate
buffers, or convert data formats.

Call `KernargBuffer::validate()` before dispatch to catch missing required
arguments. Arguments marked with `optional: true` may remain unset and keep
their zero-initialized bytes.

### `kernarg_size`

Required integer. Total byte size of the packed kernarg buffer.

This value must match the HSACO `.kernarg_segment_size`. Runtime examples call
`lrrt::require_kernarg_layout` to check that their host-side argument struct
matches the manifest before launching.

### `block`

Required array of three integers `[x, y, z]`. HSA workgroup size used as
`lr_launch_config_t.block`.

All dimensions must be non-zero. Current Triton examples use one-dimensional
workgroups with `y = 1` and `z = 1`.

### `grid`

Required array `[x_expr, y, z]`. The current schema supports a constrained
one-dimensional grid shape:

```json
["ceil_div(n, DIVISOR) * MULTIPLIER", 1, 1]
```

`lrrt::Bundle::launch_config(n)` evaluates the expression as:

```text
grid.x = ceil_div(n, DIVISOR) * MULTIPLIER
grid.y = 1
grid.z = 1
```

The grid is the HSA total grid size, not the Triton program count. For Triton
kernels, `MULTIPLIER` is usually the workgroup size, so the expression maps a
program count to a total work-item count.

Non-unit `grid[1]` or `grid[2]` values are rejected until the runtime exposes a
manifest ABI for multi-dimensional launch derivation.

### `shared_memory_bytes`

Optional integer. Dynamic shared-memory requirement passed to
`lr_launch_config_t.shared_memory_bytes`. Missing values default to `0`.

This is separate from fixed group segment usage encoded in the HSACO metadata.

### `workspace_bytes`

Optional integer. Per-dispatch temporary workspace requirement.

The current runtime does not allocate workspace from this field. It is reserved
for an executor layer or a future bundle launcher that owns temporary buffer
management.

### Producer Metadata

Kernel entries may include producer-specific metadata objects, such as
`triton`. The runtime ignores these fields. Tests and debugging tools may use
them to detect compiler version drift or to explain how the bundle was
generated.

Example:

```json
"triton": {
  "version": "3.4.0",
  "block_size": 256,
  "num_warps": 8
}
```

## Validation Responsibility

`lrrt::bundle` performs lightweight runtime-facing validation:

- required `target` and kernel fields are present
- kernel names are unique
- `code_object` stays inside the bundle directory
- `block` has exactly three non-zero dimensions
- `grid` uses the supported one-dimensional expression form
- argument offsets are present, strictly increasing, and inside `kernarg_size`

The build and test layer performs producer consistency checks that require
inspecting HSACO metadata:

- manifest `target` matches the code object target
- manifest `symbol` matches the HSACO kernel metadata
- `kernarg_size` matches `.kernarg_segment_size`
- manifest argument offsets and sizes match HSACO argument metadata
- `shared_memory_bytes` matches the dynamic shared-memory requirement expected
  by generated kernels

Keeping these checks separate lets the runtime remain small while still giving
compiler integrations a precise contract to test against.

## Non-Goals

- The manifest is not a stable high-level operator schema.
- The manifest does not describe tensor shapes, strides, ownership, or graph
  scheduling.
- The runtime core does not compile kernels or inspect full compiler metadata.
- The current schema does not support multi-dimensional grid derivation.
