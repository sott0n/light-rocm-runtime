#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys


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
        offset = re.search(r"\.offset:\s*(\d+)", line)
        size = re.search(r"\.size:\s*(\d+)", line)
        if offset:
            current = {"offset": int(offset.group(1))}
            args.append(current)
        elif size and current is not None:
            current["size"] = int(size.group(1))

    if not args:
        raise AssertionError("HSACO metadata has no args")
    return args


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

    kernels = manifest["kernels"]
    if not kernels:
        raise AssertionError("manifest has no kernels")

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
