import triton.language as tl
from manifest_utils import (
    arg,
    compile_kernel,
    kernel_entry,
    parse_generator_args,
    profile_scratch_args,
    triton,
    triton_metadata,
    write_code_object,
    write_manifest,
)


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
    args = parse_generator_args()

    kernel_args = [
        arg("x", "ptr", 0, 8),
        arg("weight", "ptr", 8, 8),
        arg("out", "ptr", 16, 8),
        arg("eps", "fp32", 24, 4),
        arg("rows", "i32", 28, 4),
        arg("hidden", "i32", 32, 4),
        *profile_scratch_args(40),
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
            compiled = compile_kernel(
                rmsnorm_kernel,
                args.arch,
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
                num_warps=num_warps,
            )
            write_code_object(args.output_dir, code_object, compiled)

            kernels.append(
                kernel_entry(
                    f"rmsnorm_{dtype}_{block_size}",
                    "rmsnorm_kernel",
                    code_object,
                    kernel_args,
                    56,
                    workgroup_size,
                    f"ceil_div(n, 1) * {workgroup_size}",
                    compiled,
                    triton_metadata(
                        dtype=dtype,
                        block_size=block_size,
                        max_hidden_size=block_size,
                        num_warps=num_warps,
                    ),
                )
            )

    write_manifest(args.output_dir, args.arch, kernels)


if __name__ == "__main__":
    main()
