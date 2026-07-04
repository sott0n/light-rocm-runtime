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
    data_types = [
        ("fp32", "*fp32"),
        ("fp16", "*fp16"),
        ("bf16", "*bf16"),
    ]
    hidden_specializations = [
        (1024, 4),
        (2048, 8),
        (4096, 8),
        (8192, 8),
    ]
    kernels = []
    for dtype, pointer_type in data_types:
        for block_size, num_warps in hidden_specializations:
            workgroup_size = num_warps * 32
            code_object = f"kernels_{dtype}_{block_size}.hsaco"
            if dtype == "fp32" and block_size == 1024:
                code_object = "kernels.hsaco"
            elif dtype == "fp32":
                code_object = f"kernels_{block_size}.hsaco"

            compiled = compile_kernel(
                matvec_kernel,
                args.arch,
                signature={
                    "x": pointer_type,
                    "weight": pointer_type,
                    "out": pointer_type,
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
                    f"matvec_{dtype}_{block_size}",
                    "matvec_kernel",
                    code_object,
                    kernel_args,
                    48,
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
