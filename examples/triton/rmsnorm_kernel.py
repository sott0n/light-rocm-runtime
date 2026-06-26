import argparse
import json
import os

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def rmsnorm_kernel(x, weight, out, eps, rows, hidden, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (row < rows) & (offsets < hidden)
    base = row * hidden + offsets
    xv = tl.load(x + base, mask=mask, other=0.0).to(tl.float32)
    wv = tl.load(weight + offsets, mask=offsets < hidden, other=0.0).to(tl.float32)
    mean_square = tl.sum(xv * xv, axis=0) / hidden
    scale = tl.rsqrt(mean_square + eps)
    tl.store(out + base, xv * scale * wv, mask=mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    kernel_args = [
        {"name": "x", "type": "ptr", "offset": 0, "size": 8},
        {"name": "weight", "type": "ptr", "offset": 8, "size": 8},
        {"name": "out", "type": "ptr", "offset": 16, "size": 8},
        {"name": "eps", "type": "fp32", "offset": 24, "size": 4},
        {"name": "rows", "type": "i32", "offset": 28, "size": 4},
        {"name": "hidden", "type": "i32", "offset": 32, "size": 4},
        {
            "name": "_triton_global_scratch",
            "type": "ptr",
            "offset": 40,
            "size": 8,
            "optional": True,
        },
        {
            "name": "_triton_profile_scratch",
            "type": "ptr",
            "offset": 48,
            "size": 8,
            "optional": True,
        },
    ]
    data_types = [
        ("fp32", "*fp32"),
        ("fp16", "*fp16"),
        ("bf16", "*bf16"),
    ]
    hidden_specializations = [
        (1024, 4),
        (2048, 8),
        (4096, 8),
    ]
    kernels = []
    for dtype, pointer_type in data_types:
        for block_size, num_warps in hidden_specializations:
            workgroup_size = num_warps * 32
            code_object = f"kernels_{dtype}_{block_size}.hsaco"
            if dtype == "fp32" and block_size == 1024:
                code_object = "kernels.hsaco"
            source = triton.compiler.ASTSource(
                rmsnorm_kernel,
                signature={
                    "x": pointer_type,
                    "weight": pointer_type,
                    "out": pointer_type,
                    "eps": "fp32",
                    "rows": "i32",
                    "hidden": "i32",
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
                    "name": f"rmsnorm_{dtype}_{block_size}",
                    "symbol": "rmsnorm_kernel",
                    "code_object": code_object,
                    "args": kernel_args,
                    "kernarg_size": 56,
                    "block": [workgroup_size, 1, 1],
                    "grid": [f"ceil_div(n, 1) * {workgroup_size}", 1, 1],
                    "shared_memory_bytes": compiled.metadata.shared,
                    "triton": {
                        "version": triton.__version__,
                        "dtype": dtype,
                        "block_size": block_size,
                        "max_hidden_size": block_size,
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
