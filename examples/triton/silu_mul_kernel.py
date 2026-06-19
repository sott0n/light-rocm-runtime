import argparse
import json
import os

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def silu_mul_kernel(gate, up, out, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    gate_value = tl.load(gate + offsets, mask=mask, other=0.0)
    up_value = tl.load(up + offsets, mask=mask, other=0.0)
    activated = gate_value * tl.sigmoid(gate_value)
    tl.store(out + offsets, activated * up_value, mask=mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    block_size = 256
    num_warps = 4
    workgroup_size = 128
    source = triton.compiler.ASTSource(
        silu_mul_kernel,
        signature={
            "gate": "*fp32",
            "up": "*fp32",
            "out": "*fp32",
            "n": "i32",
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
                "name": "silu_mul",
                "symbol": "silu_mul_kernel",
                "code_object": "kernels.hsaco",
                "args": [
                    {"name": "gate", "type": "ptr", "offset": 0, "size": 8},
                    {"name": "up", "type": "ptr", "offset": 8, "size": 8},
                    {"name": "out", "type": "ptr", "offset": 16, "size": 8},
                    {"name": "n", "type": "i32", "offset": 24, "size": 4},
                    {
                        "name": "_triton_scratch_0",
                        "type": "ptr",
                        "offset": 32,
                        "size": 8,
                        "optional": True,
                    },
                    {
                        "name": "_triton_scratch_1",
                        "type": "ptr",
                        "offset": 40,
                        "size": 8,
                        "optional": True,
                    },
                ],
                "kernarg_size": 48,
                "block": [workgroup_size, 1, 1],
                "grid": ["ceil_div(n, 256) * 128", 1, 1],
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
