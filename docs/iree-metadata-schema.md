# IREE Metadata Summary Schema

## Purpose

`tools/iree_metadata_summary.py` converts IREE `executable-targets` MLIR into a
small JSON document for the experimental lrrt-backed IREE HAL adapter. The
summary is an adapter contract: it records the dispatch metadata that lrrt needs
to load an executable export, build a launch configuration, and understand the
buffer binding order.

This schema is intentionally narrower than IREE VMFB. It does not describe IREE
VM bytecode, Flow, Stream scheduling, tensor shapes, or high-level model
semantics.

## Top-Level Object

| Field | Type | Required | Adapter mapping | Meaning |
| --- | --- | --- | --- | --- |
| `target` | string | yes | `ExecutableMetadata::target` | ROCm target architecture such as `gfx1101`. |
| `executables` | array of executable objects | yes | flattened into `ExecutableMetadata::exports` | IREE HAL executables found in the `executable-targets` MLIR. |

The current C++ parser also accepts the older single-executable shape with
top-level `executable`, `variant`, and `exports` fields for existing tests and
probes. New summaries should use `executables`.

## Executable Object

| Field | Type | Required | Adapter mapping | Meaning |
| --- | --- | --- | --- | --- |
| `executable` | string | yes | `ExecutableMetadata::executable` for the first executable; `ExportMetadata::dispatch.executable` for each export | IREE HAL executable name. |
| `variant` | string | yes | `ExecutableMetadata::variant` for the first executable; `ExportMetadata::dispatch.variant` for each export | Executable variant name, currently expected to identify the ROCm HSACO path. |
| `exports` | array of export objects | yes | `ExecutableMetadata::exports` | Exported dispatch entry points in this executable. |

## Export Object

| Field | Type | Required | Adapter mapping | Meaning |
| --- | --- | --- | --- | --- |
| `symbol` | string | yes | `ExportMetadata::symbol` | Exported kernel entry point symbol passed to `lr_kernel_get`. |
| `ordinal` | integer | yes | `ExportMetadata::ordinal` | Stable export index for adapter lookup. |
| `workgroup_size` | array of 3 integers | yes | `ExportMetadata::workgroup_size` | Launch block size converted to `lr_launch_config_t::block`. |
| `subgroup_size` | integer | yes | `ExportMetadata::subgroup_size` | Compiler-selected subgroup size for diagnostics and validation. |
| `bindings` | array of binding objects | yes | `ExportMetadata::bindings` | Buffer binding order used when packing kernargs. |
| `kernel` | kernel object | yes | `ExportMetadata::kernel` | Lowered LLVM/ROCDL kernel metadata. |
| `dispatch` | dispatch object | yes | `ExportMetadata::dispatch` | Stream dispatch reference back to the executable variant and symbol. |

## Binding Object

| Field | Type | Required | Adapter mapping | Meaning |
| --- | --- | --- | --- | --- |
| `index` | integer | yes | `BindingMetadata::index` | Binding index in the HAL pipeline layout. |
| `type` | string | yes | `BindingMetadata::type` | HAL binding type, for example `storage_buffer`. |
| `flags` | array of strings | yes | `BindingMetadata::flags` | Binding flags split from IREE's pipeline layout, for example `ReadOnly` and `Indirect`. |

The first adapter prototype should treat binding order as ABI-relevant. The
metadata tells the adapter which buffers exist and in what order they must be
packed, but it does not make lrrt own tensor shapes or graph semantics.

## Kernel Object

| Field | Type | Required | Adapter mapping | Meaning |
| --- | --- | --- | --- | --- |
| `symbol` | string | yes | `KernelMetadata::symbol` | Lowered `llvm.func` kernel symbol. |
| `attributes` | array of strings | yes | `KernelMetadata::attributes` | Selected compiler attributes such as `rocdl.kernel` and workgroup attributes. |

The current parser records a small allowlist of attributes that matter to the
adapter investigation:

- `gpu.known_block_size`
- `rocdl.flat_work_group_size`
- `rocdl.kernel`
- `rocdl.reqd_work_group_size`

## Dispatch Object

| Field | Type | Required | Adapter mapping | Meaning |
| --- | --- | --- | --- | --- |
| `executable` | string | yes | `DispatchMetadata::executable` | Executable name referenced by `stream.cmd.dispatch`. |
| `variant` | string | yes | `DispatchMetadata::variant` | Executable variant referenced by `stream.cmd.dispatch`. |
| `symbol` | string | yes | `DispatchMetadata::symbol` | Export symbol referenced by `stream.cmd.dispatch`. |

## Current Minimal Example

For `tools/iree_minimal_mul.mlir`, the generated summary currently has one
executable with one export:

```json
{
  "target": "gfx1101",
  "executables": [
    {
      "executable": "simple_mul_dispatch_0",
      "variant": "rocm_hsaco_fb",
      "exports": [
        {
          "symbol": "simple_mul_dispatch_0_elementwise_4_f32",
          "ordinal": 0,
          "workgroup_size": [32, 1, 1],
          "subgroup_size": 32,
          "bindings": [
            {"index": 0, "type": "storage_buffer", "flags": ["ReadOnly", "Indirect"]},
            {"index": 1, "type": "storage_buffer", "flags": ["ReadOnly", "Indirect"]},
            {"index": 2, "type": "storage_buffer", "flags": ["Indirect"]}
          ],
          "kernel": {
            "symbol": "simple_mul_dispatch_0_elementwise_4_f32",
            "attributes": [
              "gpu.known_block_size",
              "rocdl.flat_work_group_size",
              "rocdl.kernel",
              "rocdl.reqd_work_group_size"
            ]
          },
          "dispatch": {
            "executable": "simple_mul_dispatch_0",
            "variant": "rocm_hsaco_fb",
            "symbol": "simple_mul_dispatch_0_elementwise_4_f32"
          }
        }
      ]
    }
  ]
}
```

For `tools/iree_mixed_matmuls.mlir`, the same IREE entry point lowers into two
HAL executables. The summary preserves that shape instead of pretending there
is only one top-level executable:

```json
{
  "target": "gfx1101",
  "executables": [
    {
      "executable": "mixed_matmuls_dispatch_0",
      "variant": "rocm_hsaco_fb",
      "exports": [
        {"symbol": "mixed_matmuls_dispatch_0_matmul_2x2x2_f32", "...": "..."}
      ]
    },
    {
      "executable": "mixed_matmuls_dispatch_1",
      "variant": "rocm_hsaco_fb",
      "exports": [
        {"symbol": "mixed_matmuls_dispatch_1_matmul_2x3x2_f32", "...": "..."}
      ]
    }
  ]
}
```

The adapter uses this data as:

```text
ExecutableMetadata::target
  -> validate selected device/code object target
ExportMetadata::symbol
  -> Executable::entry_point / lr_kernel_get
ExportMetadata::workgroup_size
  -> lr_launch_config_t::block
BindingMetadata list
  -> adapter-side kernarg packing order
DispatchMetadata
  -> cross-check the Stream dispatch target
```

`ExecutableMetadata::executable` and `ExecutableMetadata::variant` are retained
as first-executable compatibility fields. For multi-executable summaries, use
`ExportMetadata::dispatch.executable` and `ExportMetadata::dispatch.variant`
when the exact executable owner matters.

## Non-Goals

- Do not parse VMFB in the lrrt C runtime core.
- Do not make this JSON format a replacement for IREE's runtime metadata.
- Do not add tensor graph semantics or shape propagation to this schema.
- Do not use this schema to bypass the planned HAL adapter boundary.
