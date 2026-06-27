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
    args = parse_generator_args()

    kernel_args = [
        arg("x", "ptr", 0, 8),
        arg("weight", "ptr", 8, 8),
        arg("out", "ptr", 16, 8),
        arg("outputs", "i32", 24, 4),
        arg("hidden", "i32", 28, 4),
        *profile_scratch_args(32),
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
            matvec_kernel,
            args.arch,
            signature={
                "x": "*fp32",
                "weight": "*fp32",
                "out": "*fp32",
                "outputs": "i32",
                "hidden": "i32",
                "BLOCK_SIZE": "constexpr",
            },
            constexprs={"BLOCK_SIZE": block_size},
            num_warps=num_warps,
        )
        write_code_object(args.output_dir, code_object, compiled)

        kernels.append(
            kernel_entry(
                f"matvec_fp32_{block_size}",
                "matvec_kernel",
                code_object,
                kernel_args,
                48,
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
