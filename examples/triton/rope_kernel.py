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
def rope_kernel(
    x,
    cos,
    sin,
    out,
    rows,
    heads,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    half = head_dim // 2
    partner_offsets = tl.where(offsets < half, offsets + half, offsets - half)
    mask = (row < rows) & (offsets < head_dim)
    base = row * head_dim
    value = tl.load(x + base + offsets, mask=mask, other=0.0)
    partner = tl.load(x + base + partner_offsets, mask=mask, other=0.0)

    token = row // heads
    frequency = tl.where(offsets < half, offsets, offsets - half)
    table_offset = token * half + frequency
    cos_value = tl.load(cos + table_offset, mask=mask, other=0.0)
    sin_value = tl.load(sin + table_offset, mask=mask, other=0.0)
    rotated = tl.where(offsets < half, -partner, partner)
    tl.store(out + base + offsets, value * cos_value + rotated * sin_value, mask=mask)


def main():
    args = parse_generator_args()

    kernel_args = [
        arg("x", "ptr", 0, 8),
        arg("cos", "ptr", 8, 8),
        arg("sin", "ptr", 16, 8),
        arg("out", "ptr", 24, 8),
        arg("rows", "i32", 32, 4),
        arg("heads", "i32", 36, 4),
        arg("head_dim", "i32", 40, 4),
        *profile_scratch_args(48),
    ]
    specializations = [
        (64, 2, "kernels.hsaco"),
        (128, 4, "kernels_128.hsaco"),
        (256, 8, "kernels_256.hsaco"),
    ]
    kernels = []
    for block_size, num_warps, code_object in specializations:
        workgroup_size = num_warps * 32
        compiled = compile_kernel(
            rope_kernel,
            args.arch,
            signature={
                "x": "*fp32",
                "cos": "*fp32",
                "sin": "*fp32",
                "out": "*fp32",
                "rows": "i32",
                "heads": "i32",
                "head_dim": "i32",
                "BLOCK_SIZE": "constexpr",
            },
            constexprs={"BLOCK_SIZE": block_size},
            num_warps=num_warps,
        )
        write_code_object(args.output_dir, code_object, compiled)

        kernels.append(
            kernel_entry(
                f"rope_{block_size}",
                "rope_kernel",
                code_object,
                kernel_args,
                64,
                workgroup_size,
                f"ceil_div(n, 1) * {workgroup_size}",
                compiled,
                triton_metadata(
                    block_size=block_size,
                    max_head_dim=block_size,
                    num_warps=num_warps,
                ),
            )
        )

    write_manifest(args.output_dir, args.arch, kernels)


if __name__ == "__main__":
    main()
