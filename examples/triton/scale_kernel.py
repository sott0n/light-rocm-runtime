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
def scale_kernel(x, out, factor, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    xv = tl.load(x + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, factor * xv, mask=mask)


def main():
    args = parse_generator_args()

    block_size = 256
    num_warps = 4
    workgroup_size = 128
    compiled = compile_kernel(
        scale_kernel,
        args.arch,
        signature={
            "x": "*fp32",
            "out": "*fp32",
            "factor": "fp32",
            "n": "i32",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"BLOCK_SIZE": block_size},
        num_warps=num_warps,
    )
    write_code_object(args.output_dir, "kernels.hsaco", compiled)

    kernel_args = [
        arg("x", "ptr", 0, 8),
        arg("out", "ptr", 8, 8),
        arg("factor", "fp32", 16, 4),
        arg("n", "i32", 20, 4),
        *legacy_scratch_args(24),
    ]
    write_manifest(
        args.output_dir,
        args.arch,
        [
            kernel_entry(
                "scale",
                "scale_kernel",
                "kernels.hsaco",
                kernel_args,
                40,
                workgroup_size,
                "ceil_div(n, 256) * 128",
                compiled,
                triton_metadata(block_size=block_size, num_warps=num_warps),
            )
        ],
    )


if __name__ == "__main__":
    main()
