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
def attention_score_kernel(q, k, out, keys, head_dim, scale, BLOCK_SIZE: tl.constexpr):
    key = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (key < keys) & (offsets < head_dim)
    q_values = tl.load(q + offsets, mask=offsets < head_dim, other=0.0).to(tl.float32)
    k_values = tl.load(k + key * head_dim + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    score = tl.sum(q_values * k_values, axis=0) * scale
    tl.store(out + key, score, mask=key < keys)


def main():
    args = parse_generator_args()

    kernel_args = [
        arg("q", "ptr", 0, 8),
        arg("k", "ptr", 8, 8),
        arg("out", "ptr", 16, 8),
        arg("keys", "i32", 24, 4),
        arg("head_dim", "i32", 28, 4),
        arg("scale", "fp32", 32, 4),
        *profile_scratch_args(40),
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
            attention_score_kernel,
            args.arch,
            signature={
                "q": "*fp32",
                "k": "*fp32",
                "out": "*fp32",
                "keys": "i32",
                "head_dim": "i32",
                "scale": "fp32",
                "BLOCK_SIZE": "constexpr",
            },
            constexprs={"BLOCK_SIZE": block_size},
            num_warps=num_warps,
        )
        write_code_object(args.output_dir, code_object, compiled)

        kernels.append(
            kernel_entry(
                f"attention_score_fp32_{block_size}",
                "attention_score_kernel",
                code_object,
                kernel_args,
                56,
                workgroup_size,
                f"ceil_div(n, 1) * {workgroup_size}",
                compiled,
                triton_metadata(
                    dtype="fp32",
                    block_size=block_size,
                    max_head_dim=block_size,
                    num_warps=num_warps,
                    op="attention_score",
                ),
            )
        )

    write_manifest(args.output_dir, args.arch, kernels)


if __name__ == "__main__":
    main()
