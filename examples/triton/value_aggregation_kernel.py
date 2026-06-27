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
def value_aggregation_kernel(probs, v, out, keys, head_dim, BLOCK_KEYS: tl.constexpr):
    col = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_KEYS)
    mask = (col < head_dim) & (offsets < keys)
    probs_values = tl.load(probs + offsets, mask=offsets < keys, other=0.0).to(
        tl.float32
    )
    v_values = tl.load(v + offsets * head_dim + col, mask=mask, other=0.0).to(
        tl.float32
    )
    result = tl.sum(probs_values * v_values, axis=0)
    tl.store(out + col, result, mask=col < head_dim)


def main():
    args = parse_generator_args()

    kernel_args = [
        arg("probs", "ptr", 0, 8),
        arg("v", "ptr", 8, 8),
        arg("out", "ptr", 16, 8),
        arg("keys", "i32", 24, 4),
        arg("head_dim", "i32", 28, 4),
        *profile_scratch_args(32),
    ]
    specializations = [
        (1024, 4, "kernels.hsaco"),
        (2048, 8, "kernels_2048.hsaco"),
        (4096, 8, "kernels_4096.hsaco"),
    ]
    kernels = []
    for block_keys, num_warps, code_object in specializations:
        workgroup_size = num_warps * 32
        compiled = compile_kernel(
            value_aggregation_kernel,
            args.arch,
            signature={
                "probs": "*fp32",
                "v": "*fp32",
                "out": "*fp32",
                "keys": "i32",
                "head_dim": "i32",
                "BLOCK_KEYS": "constexpr",
            },
            constexprs={"BLOCK_KEYS": block_keys},
            num_warps=num_warps,
        )
        write_code_object(args.output_dir, code_object, compiled)

        kernels.append(
            kernel_entry(
                f"value_aggregation_fp32_{block_keys}",
                "value_aggregation_kernel",
                code_object,
                kernel_args,
                48,
                workgroup_size,
                f"ceil_div(n, 1) * {workgroup_size}",
                compiled,
                triton_metadata(
                    dtype="fp32",
                    block_keys=block_keys,
                    max_keys=block_keys,
                    num_warps=num_warps,
                    op="value_aggregation",
                ),
            )
        )

    write_manifest(args.output_dir, args.arch, kernels)


if __name__ == "__main__":
    main()
