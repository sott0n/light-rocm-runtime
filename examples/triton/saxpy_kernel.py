import triton.language as tl
from manifest_utils import (
    arg,
    compile_kernel,
    kernel_entry,
    legacy_scratch_args,
    parse_generator_args,
    triton,
    triton_metadata,
    write_code_object,
    write_manifest,
)


@triton.jit
def saxpy_kernel(x, y, out, a, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    xv = tl.load(x + offsets, mask=mask, other=0.0)
    yv = tl.load(y + offsets, mask=mask, other=0.0)
    result = a * xv + yv
    tl.store(out + offsets, result, mask=mask)


def main():
    args = parse_generator_args()

    block_size = 256
    num_warps = 4
    workgroup_size = 128
    compiled = compile_kernel(
        saxpy_kernel,
        args.arch,
        signature={
            "x": "*fp32",
            "y": "*fp32",
            "out": "*fp32",
            "a": "fp32",
            "n": "i32",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"BLOCK_SIZE": block_size},
        num_warps=num_warps,
    )
    write_code_object(args.output_dir, "kernels.hsaco", compiled)

    kernel_args = [
        arg("x", "ptr", 0, 8),
        arg("y", "ptr", 8, 8),
        arg("out", "ptr", 16, 8),
        arg("a", "fp32", 24, 4),
        arg("n", "i32", 28, 4),
        *legacy_scratch_args(32),
    ]
    write_manifest(
        args.output_dir,
        args.arch,
        [
            kernel_entry(
                "saxpy",
                "saxpy_kernel",
                "kernels.hsaco",
                kernel_args,
                48,
                workgroup_size,
                "ceil_div(n, 256) * 128",
                compiled,
                triton_metadata(block_size=block_size, num_warps=num_warps),
            )
        ],
    )


if __name__ == "__main__":
    main()
