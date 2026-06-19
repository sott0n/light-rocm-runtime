import argparse
import json
import os

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def rmsnorm_kernel(x, weight, out, eps, rows, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = row < rows
    base = row * BLOCK_SIZE + offsets
    xv = tl.load(x + base, mask=mask, other=0.0)
    wv = tl.load(weight + offsets)
    mean_square = tl.sum(xv * xv, axis=0) / BLOCK_SIZE
    scale = tl.rsqrt(mean_square + eps)
    tl.store(out + base, xv * scale * wv, mask=mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    block_size = 1024
    num_warps = 4
    workgroup_size = 128
    source = triton.compiler.ASTSource(
        rmsnorm_kernel,
        signature={
            "x": "*fp32",
            "weight": "*fp32",
            "out": "*fp32",
            "eps": "fp32",
            "rows": "i32",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"BLOCK_SIZE": block_size},
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", args.arch, 64),
        options={"num_warps": num_warps, "num_stages": 2},
    )

    hsaco_path = os.path.join(args.output_dir, "kernels.hsaco")
    with open(hsaco_path, "wb") as hsaco_file:
        hsaco_file.write(compiled.asm["hsaco"])

    manifest = {
        "target": args.arch,
        "kernels": [
            {
                "name": "rmsnorm",
                "symbol": "rmsnorm_kernel",
                "code_object": "kernels.hsaco",
                "args": [
                    {"name": "x", "type": "ptr", "offset": 0, "size": 8},
                    {"name": "weight", "type": "ptr", "offset": 8, "size": 8},
                    {"name": "out", "type": "ptr", "offset": 16, "size": 8},
                    {"name": "eps", "type": "fp32", "offset": 24, "size": 4},
                    {"name": "rows", "type": "i32", "offset": 28, "size": 4},
                    {
                        "name": "_triton_global_scratch",
                        "type": "ptr",
                        "offset": 32,
                        "size": 8,
                        "optional": True,
                    },
                    {
                        "name": "_triton_profile_scratch",
                        "type": "ptr",
                        "offset": 40,
                        "size": 8,
                        "optional": True,
                    },
                ],
                "kernarg_size": 48,
                "block": [workgroup_size, 1, 1],
                "grid": ["ceil_div(n, 1) * 128", 1, 1],
                "shared_memory_bytes": compiled.metadata.shared,
                "triton": {
                    "version": triton.__version__,
                    "block_size": block_size,
                    "num_warps": num_warps,
                },
                "workspace_bytes": 0,
            }
        ],
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")


if __name__ == "__main__":
    main()
