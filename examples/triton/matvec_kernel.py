import argparse
import json
import os

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def matvec_kernel(x, weight, out, outputs, hidden, BLOCK_SIZE: tl.constexpr):
    output = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (output < outputs) & (offsets < hidden)
    x_values = tl.load(x + offsets, mask=offsets < hidden, other=0.0).to(tl.float32)
    weight_values = tl.load(
        weight + output * hidden + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    result = tl.sum(x_values * weight_values, axis=0)
    tl.store(out + output, result, mask=output < outputs)


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
        {"name": "outputs", "type": "i32", "offset": 24, "size": 4},
        {"name": "hidden", "type": "i32", "offset": 28, "size": 4},
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
    ]
    specializations = [
        (1024, 4, "kernels.hsaco"),
        (2048, 8, "kernels_2048.hsaco"),
        (4096, 8, "kernels_4096.hsaco"),
    ]
    kernels = []
    for block_size, num_warps, code_object in specializations:
        workgroup_size = num_warps * 32
        source = triton.compiler.ASTSource(
            matvec_kernel,
            signature={
                "x": "*fp32",
                "weight": "*fp32",
                "out": "*fp32",
                "outputs": "i32",
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
                "name": f"matvec_fp32_{block_size}",
                "symbol": "matvec_kernel",
                "code_object": code_object,
                "args": kernel_args,
                "kernarg_size": 48,
                "block": [workgroup_size, 1, 1],
                "grid": [f"ceil_div(n, 1) * {workgroup_size}", 1, 1],
                "shared_memory_bytes": compiled.metadata.shared,
                "triton": {
                    "version": triton.__version__,
                    "dtype": "fp32",
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
