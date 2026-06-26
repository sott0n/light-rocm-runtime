import argparse
import json
import os

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def rope_kernel(
    x,
    cos,
    sin,
    out,
    rows,
    heads,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    half = head_dim // 2
    partner_offsets = tl.where(offsets < half, offsets + half, offsets - half)
    mask = (row < rows) & (offsets < head_dim)
    base = row * head_dim
    value = tl.load(x + base + offsets, mask=mask, other=0.0)
    partner = tl.load(x + base + partner_offsets, mask=mask, other=0.0)

    token = row // heads
    frequency = tl.where(offsets < half, offsets, offsets - half)
    table_offset = token * half + frequency
    cos_value = tl.load(cos + table_offset, mask=mask, other=0.0)
    sin_value = tl.load(sin + table_offset, mask=mask, other=0.0)
    rotated = tl.where(offsets < half, -partner, partner)
    tl.store(out + base + offsets, value * cos_value + rotated * sin_value, mask=mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    kernel_args = [
        {"name": "x", "type": "ptr", "offset": 0, "size": 8},
        {"name": "cos", "type": "ptr", "offset": 8, "size": 8},
        {"name": "sin", "type": "ptr", "offset": 16, "size": 8},
        {"name": "out", "type": "ptr", "offset": 24, "size": 8},
        {"name": "rows", "type": "i32", "offset": 32, "size": 4},
        {"name": "heads", "type": "i32", "offset": 36, "size": 4},
        {"name": "head_dim", "type": "i32", "offset": 40, "size": 4},
        {
            "name": "_triton_global_scratch",
            "type": "ptr",
            "offset": 48,
            "size": 8,
            "optional": True,
        },
        {
            "name": "_triton_profile_scratch",
            "type": "ptr",
            "offset": 56,
            "size": 8,
            "optional": True,
        },
    ]
    specializations = [
        (64, 2, "kernels.hsaco"),
        (128, 4, "kernels_128.hsaco"),
    ]
    kernels = []
    for block_size, num_warps, code_object in specializations:
        workgroup_size = num_warps * 32
        source = triton.compiler.ASTSource(
            rope_kernel,
            signature={
                "x": "*fp32",
                "cos": "*fp32",
                "sin": "*fp32",
                "out": "*fp32",
                "rows": "i32",
                "heads": "i32",
                "head_dim": "i32",
                "BLOCK_SIZE": "constexpr",
            },
            constexprs={"BLOCK_SIZE": block_size},
        )
        compiled = triton.compile(
            source,
            target=GPUTarget("hip", args.arch, 64),
            options={"num_warps": num_warps, "num_stages": 2},
        )

        hsaco_path = os.path.join(args.output_dir, code_object)
        with open(hsaco_path, "wb") as hsaco_file:
            hsaco_file.write(compiled.asm["hsaco"])

        kernels.append(
            {
                "name": f"rope_{block_size}",
                "symbol": "rope_kernel",
                "code_object": code_object,
                "args": kernel_args,
                "kernarg_size": 64,
                "block": [workgroup_size, 1, 1],
                "grid": [f"ceil_div(n, 1) * {workgroup_size}", 1, 1],
                "shared_memory_bytes": compiled.metadata.shared,
                "triton": {
                    "version": triton.__version__,
                    "block_size": block_size,
                    "max_head_dim": block_size,
                    "num_warps": num_warps,
                },
                "workspace_bytes": 0,
            }
        )

    manifest = {"manifest_version": 1, "target": args.arch, "kernels": kernels}
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")


if __name__ == "__main__":
    main()
