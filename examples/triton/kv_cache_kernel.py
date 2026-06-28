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
def kv_cache_update_kernel(
    k,
    v,
    k_cache,
    v_cache,
    position,
    max_tokens,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (position < max_tokens) & (offsets < head_dim)
    cache_offsets = position * head_dim + offsets
    k_values = tl.load(k + offsets, mask=offsets < head_dim, other=0.0)
    v_values = tl.load(v + offsets, mask=offsets < head_dim, other=0.0)
    tl.store(k_cache + cache_offsets, k_values, mask=mask)
    tl.store(v_cache + cache_offsets, v_values, mask=mask)


@triton.jit
def kv_cache_read_kernel(
    k_cache,
    v_cache,
    k,
    v,
    position,
    max_tokens,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (position < max_tokens) & (offsets < head_dim)
    cache_offsets = position * head_dim + offsets
    k_values = tl.load(k_cache + cache_offsets, mask=mask, other=0.0)
    v_values = tl.load(v_cache + cache_offsets, mask=mask, other=0.0)
    tl.store(k + offsets, k_values, mask=offsets < head_dim)
    tl.store(v + offsets, v_values, mask=offsets < head_dim)


def main():
    args = parse_generator_args()

    update_args = [
        arg("k", "ptr", 0, 8),
        arg("v", "ptr", 8, 8),
        arg("k_cache", "ptr", 16, 8),
        arg("v_cache", "ptr", 24, 8),
        arg("position", "i32", 32, 4),
        arg("max_tokens", "i32", 36, 4),
        arg("head_dim", "i32", 40, 4),
        *profile_scratch_args(48),
    ]
    read_args = [
        arg("k_cache", "ptr", 0, 8),
        arg("v_cache", "ptr", 8, 8),
        arg("k", "ptr", 16, 8),
        arg("v", "ptr", 24, 8),
        arg("position", "i32", 32, 4),
        arg("max_tokens", "i32", 36, 4),
        arg("head_dim", "i32", 40, 4),
        *profile_scratch_args(48),
    ]
    specializations = [
        (64, 2, "kernels.hsaco", "kernels_read.hsaco"),
        (128, 4, "kernels_128.hsaco", "kernels_read_128.hsaco"),
        (256, 8, "kernels_256.hsaco", "kernels_read_256.hsaco"),
    ]
    kernels = []
    for block_size, num_warps, update_code_object, read_code_object in specializations:
        workgroup_size = num_warps * 32
        common_signature = {
            "position": "i32",
            "max_tokens": "i32",
            "head_dim": "i32",
            "BLOCK_SIZE": "constexpr",
        }
        common_metadata = {
            "dtype": "fp32",
            "block_size": block_size,
            "max_head_dim": block_size,
            "num_warps": num_warps,
        }

        update_compiled = compile_kernel(
            kv_cache_update_kernel,
            args.arch,
            signature={
                "k": "*fp32",
                "v": "*fp32",
                "k_cache": "*fp32",
                "v_cache": "*fp32",
                **common_signature,
            },
            constexprs={"BLOCK_SIZE": block_size},
            num_warps=num_warps,
        )
        write_code_object(args.output_dir, update_code_object, update_compiled)
        kernels.append(
            kernel_entry(
                f"kv_cache_update_fp32_{block_size}",
                "kv_cache_update_kernel",
                update_code_object,
                update_args,
                64,
                workgroup_size,
                f"ceil_div(n, 1) * {workgroup_size}",
                update_compiled,
                triton_metadata(op="kv_cache_update", **common_metadata),
            )
        )

        read_compiled = compile_kernel(
            kv_cache_read_kernel,
            args.arch,
            signature={
                "k_cache": "*fp32",
                "v_cache": "*fp32",
                "k": "*fp32",
                "v": "*fp32",
                **common_signature,
            },
            constexprs={"BLOCK_SIZE": block_size},
            num_warps=num_warps,
        )
        write_code_object(args.output_dir, read_code_object, read_compiled)
        kernels.append(
            kernel_entry(
                f"kv_cache_read_fp32_{block_size}",
                "kv_cache_read_kernel",
                read_code_object,
                read_args,
                64,
                workgroup_size,
                f"ceil_div(n, 1) * {workgroup_size}",
                read_compiled,
                triton_metadata(op="kv_cache_read", **common_metadata),
            )
        )

    write_manifest(args.output_dir, args.arch, kernels)


if __name__ == "__main__":
    main()
