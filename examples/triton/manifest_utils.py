import argparse
import json
import os

import triton
from triton.backends.compiler import GPUTarget

MANIFEST_VERSION = 1


def parse_generator_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def arg(name, type_name, offset, size):
    return {"name": name, "type": type_name, "offset": offset, "size": size}


def optional_ptr_arg(name, offset):
    value = arg(name, "ptr", offset, 8)
    value["optional"] = True
    return value


def legacy_scratch_args(first_offset):
    return [
        optional_ptr_arg("_triton_scratch_0", first_offset),
        optional_ptr_arg("_triton_scratch_1", first_offset + 8),
    ]


def profile_scratch_args(first_offset):
    return [
        optional_ptr_arg("_triton_global_scratch", first_offset),
        optional_ptr_arg("_triton_profile_scratch", first_offset + 8),
    ]


def compile_kernel(kernel, arch, signature, constexprs, num_warps, num_stages=2):
    source = triton.compiler.ASTSource(
        kernel,
        signature=signature,
        constexprs=constexprs,
    )
    return triton.compile(
        source,
        target=GPUTarget("hip", arch, 64),
        options={"num_warps": num_warps, "num_stages": num_stages},
    )


def write_code_object(output_dir, code_object, compiled):
    hsaco_path = os.path.join(output_dir, code_object)
    with open(hsaco_path, "wb") as hsaco_file:
        hsaco_file.write(compiled.asm["hsaco"])


def triton_metadata(**fields):
    return {"version": triton.__version__, **fields}


def kernel_entry(
    name,
    symbol,
    code_object,
    kernel_args,
    kernarg_size,
    workgroup_size,
    grid_expr,
    compiled,
    triton_info,
    workspace_bytes=0,
):
    return {
        "name": name,
        "symbol": symbol,
        "code_object": code_object,
        "args": kernel_args,
        "kernarg_size": kernarg_size,
        "block": [workgroup_size, 1, 1],
        "grid": [grid_expr, 1, 1],
        "shared_memory_bytes": compiled.metadata.shared,
        "triton": triton_info,
        "workspace_bytes": workspace_bytes,
    }


def write_manifest(output_dir, arch, kernels):
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "target": arch,
        "kernels": kernels,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")
