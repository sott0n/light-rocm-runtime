#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys

TYPE_METADATA = {
    "ptr": {"size": 8, "value_kind": "global_buffer", "address_space": "global"},
    "i32": {"size": 4, "value_kind": "by_value"},
    "fp32": {"size": 4, "value_kind": "by_value"},
    "fp16": {"size": 2, "value_kind": "by_value"},
    "bf16": {"size": 2, "value_kind": "by_value"},
}


def run_readelf(llvm_readelf, flag, hsaco):
    result = subprocess.run(
        [llvm_readelf, flag, hsaco],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def metadata_value(metadata, field):
    match = re.search(rf"^\s*{re.escape(field)}:\s*(\S+)\s*$", metadata, re.M)
    if not match:
        raise AssertionError(f"missing HSACO metadata field: {field}")
    return match.group(1)


def metadata_int(metadata, field):
    return int(metadata_value(metadata, field))


def metadata_args(metadata):
    args_match = re.search(r"^\s*-\s*\.args:\s*$", metadata, re.M)
    if not args_match:
        raise AssertionError("missing HSACO metadata args")
    args_begin = args_match.end()
    args_end_match = re.search(
        r"^\s*\.group_segment_fixed_size:\s*", metadata[args_begin:], re.M
    )
    if not args_end_match:
        raise AssertionError("missing HSACO metadata args terminator")
    args_end = args_begin + args_end_match.start()

    args = []
    current = None
    for line in metadata[args_begin:args_end].splitlines():
        if re.search(r"^\s*-\s*", line):
            current = {}
            args.append(current)
        offset = re.search(r"\.offset:\s*(\d+)", line)
        size = re.search(r"\.size:\s*(\d+)", line)
        value_kind = re.search(r"\.value_kind:\s*(\S+)", line)
        address_space = re.search(r"\.address_space:\s*(\S+)", line)
        if offset:
            if current is None:
                current = {}
                args.append(current)
            current["offset"] = int(offset.group(1))
        elif size and current is not None:
            current["size"] = int(size.group(1))
        elif value_kind and current is not None:
            current["value_kind"] = value_kind.group(1)
        elif address_space and current is not None:
            current["address_space"] = address_space.group(1)

    if not args:
        raise AssertionError("HSACO metadata has no args")
    return args


def require_object(name, value):
    if not isinstance(value, dict):
        raise AssertionError(f"{name} must be an object")


def require_array(name, value):
    if not isinstance(value, list):
        raise AssertionError(f"{name} must be an array")


def require_string(name, value):
    if not isinstance(value, str) or not value:
        raise AssertionError(f"{name} must be a non-empty string")


def require_int(name, value, minimum=0):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise AssertionError(f"{name} must be an integer >= {minimum}")


def require_relative_bundle_path(path):
    require_string("code_object", path)
    if os.path.isabs(path) or ".." in path.split("/"):
        raise AssertionError("code_object must stay inside the bundle")


def validate_grid(kernel):
    grid = kernel.get("grid")
    require_array("grid", grid)
    if len(grid) != 3:
        raise AssertionError("grid must have exactly three dimensions")
    require_string("grid[0]", grid[0])
    if grid[1] != 1 or grid[2] != 1:
        raise AssertionError("grid[1] and grid[2] must be 1")
    if not re.fullmatch(r"ceil_div\(n,\s*\d+\)\s*\*\s*\d+", grid[0]):
        raise AssertionError(f"unsupported grid expression: {grid[0]}")


def validate_manifest(manifest):
    require_object("manifest", manifest)
    require_int("manifest_version", manifest.get("manifest_version"), 1)
    if manifest["manifest_version"] != 1:
        raise AssertionError(
            f"unsupported manifest_version: {manifest['manifest_version']}"
        )
    require_string("target", manifest.get("target"))
    kernels = manifest.get("kernels")
    require_array("kernels", kernels)
    if not kernels:
        raise AssertionError("manifest has no kernels")

    kernel_names = set()
    for index, kernel in enumerate(kernels):
        validate_kernel_manifest(index, kernel)
        if kernel["name"] in kernel_names:
            raise AssertionError(f"duplicate kernel name: {kernel['name']}")
        kernel_names.add(kernel["name"])


def validate_kernel_manifest(index, kernel):
    require_object(f"kernels[{index}]", kernel)
    for field in ("name", "symbol"):
        require_string(field, kernel.get(field))
    require_relative_bundle_path(kernel.get("code_object"))
    require_int("kernarg_size", kernel.get("kernarg_size"), 1)
    require_int("shared_memory_bytes", kernel.get("shared_memory_bytes", 0), 0)

    block = kernel.get("block")
    require_array("block", block)
    if len(block) != 3:
        raise AssertionError("block must have exactly three dimensions")
    for dim, value in enumerate(block):
        require_int(f"block[{dim}]", value, 1)
    validate_grid(kernel)

    args = kernel.get("args")
    require_array("args", args)
    if not args:
        raise AssertionError("kernel manifest has no args")
    names = set()
    previous_offset = -1
    for arg_index, arg in enumerate(args):
        validate_argument_manifest(arg_index, arg, kernel["kernarg_size"])
        if arg["name"] in names:
            raise AssertionError(f"duplicate argument name: {arg['name']}")
        if arg["offset"] <= previous_offset:
            raise AssertionError("argument offsets must be strictly increasing")
        names.add(arg["name"])
        previous_offset = arg["offset"]


def validate_argument_manifest(index, arg, kernarg_size):
    require_object(f"args[{index}]", arg)
    require_string("arg name", arg.get("name"))
    require_string("arg type", arg.get("type"))
    require_int("arg offset", arg.get("offset"), 0)
    require_int("arg size", arg.get("size"), 1)
    if arg["offset"] + arg["size"] > kernarg_size:
        raise AssertionError(f"arg {index} exceeds kernarg_size")
    if "optional" in arg and not isinstance(arg["optional"], bool):
        raise AssertionError(f"arg {index} optional must be boolean")
    if arg["type"] not in TYPE_METADATA:
        raise AssertionError(f"unsupported manifest arg type: {arg['type']}")
    assert_equal(
        f"arg {index} type size", arg["size"], TYPE_METADATA[arg["type"]]["size"]
    )


def assert_equal(name, actual, expected):
    if actual != expected:
        raise AssertionError(f"{name}: got {actual!r}, expected {expected!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--llvm-readelf", required=True)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    validate_manifest(manifest)
    kernels = manifest["kernels"]
    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
    for kernel in kernels:
        hsaco = os.path.join(manifest_dir, kernel["code_object"])
        check_kernel(args.llvm_readelf, manifest["target"], kernel, hsaco)

    print("triton_bundle_consistency: ok")
    return 0


def check_kernel(llvm_readelf, target, kernel, hsaco):
    notes = run_readelf(llvm_readelf, "--notes", hsaco)
    symbols = run_readelf(llvm_readelf, "--symbols", hsaco)

    assert_equal(
        "target",
        metadata_value(notes, "amdhsa.target"),
        f"amdgcn-amd-amdhsa--{target}",
    )
    assert_equal("kernel name", metadata_value(notes, ".name"), kernel["symbol"])
    assert_equal(
        "kernel symbol",
        metadata_value(notes, ".symbol"),
        f"{kernel['symbol']}.kd",
    )
    assert_equal(
        "kernarg size",
        metadata_int(notes, ".kernarg_segment_size"),
        kernel["kernarg_size"],
    )
    assert_equal(
        "workgroup size",
        metadata_int(notes, ".max_flat_workgroup_size"),
        kernel["block"][0],
    )

    hsaco_args = metadata_args(notes)
    manifest_args = kernel["args"]
    assert_equal("arg count", len(hsaco_args), len(manifest_args))
    for index, (actual, expected) in enumerate(zip(hsaco_args, manifest_args)):
        assert_equal(f"arg {index} offset", actual["offset"], expected["offset"])
        assert_equal(f"arg {index} size", actual["size"], expected["size"])
        expected_metadata = TYPE_METADATA[expected["type"]]
        assert_equal(
            f"arg {index} value kind",
            actual.get("value_kind"),
            expected_metadata["value_kind"],
        )
        if "address_space" in expected_metadata:
            assert_equal(
                f"arg {index} address space",
                actual.get("address_space"),
                expected_metadata["address_space"],
            )

    if not re.search(rf"\b{re.escape(kernel['symbol'])}\b", symbols):
        raise AssertionError(f"missing kernel entry symbol: {kernel['symbol']}")
    descriptor = f"{kernel['symbol']}.kd"
    if not re.search(rf"\b{re.escape(descriptor)}\b", symbols):
        raise AssertionError(f"missing kernel descriptor symbol: {descriptor}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, KeyError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
