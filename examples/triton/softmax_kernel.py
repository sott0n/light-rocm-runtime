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
def softmax_kernel(x, out, rows, hidden, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (row < rows) & (offsets < hidden)
    base = row * hidden + offsets
    values = tl.load(x + base, mask=mask, other=-float("inf")).to(tl.float32)
    values = values - tl.max(values, axis=0)
    numerators = tl.exp(values)
    denominator = tl.sum(numerators, axis=0)
    tl.store(out + base, numerators / denominator, mask=mask)


def main():
    args = parse_generator_args()

    kernel_args = [
        arg("x", "ptr", 0, 8),
        arg("out", "ptr", 8, 8),
        arg("rows", "i32", 16, 4),
        arg("hidden", "i32", 20, 4),
        *profile_scratch_args(24),
    ]
    specializations = [
        (1024, 4, "kernels.hsaco"),
        (2048, 8, "kernels_2048.hsaco"),
        (4096, 8, "kernels_4096.hsaco"),
    ]
    kernels = []
    for block_size, num_warps, code_object in specializations:
        workgroup_size = num_warps * 32
        compiled = compile_kernel(
            softmax_kernel,
            args.arch,
            signature={
                "x": "*fp32",
                "out": "*fp32",
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
                f"softmax_fp32_{block_size}",
                "softmax_kernel",
                code_object,
                kernel_args,
                40,
                workgroup_size,
                f"ceil_div(n, 1) * {workgroup_size}",
                compiled,
                triton_metadata(
                    dtype="fp32",
                    block_size=block_size,
                    max_hidden_size=block_size,
                    num_warps=num_warps,
                ),
            )
        )

    write_manifest(args.output_dir, args.arch, kernels)


if __name__ == "__main__":
    main()
