#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_vector(text: str) -> list[int]:
    values = [int(value.split(":", 1)[0].strip()) for value in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"expected three dimensions, got {values!r}")
    return values


def parse_bindings(layout: str) -> list[dict[str, object]]:
    binding_re = re.compile(
        r"#hal\.pipeline\.binding<(?P<type>[^,\s>]+)(?:,\s*(?P<flags>[^>]+))?>"
    )
    bindings = []
    for index, match in enumerate(binding_re.finditer(layout)):
        raw_flags = (match.group("flags") or "").strip().strip('"')
        flags = [flag for flag in raw_flags.split("|") if flag]
        bindings.append(
            {
                "index": index,
                "type": match.group("type"),
                "flags": flags,
            }
        )
    return bindings


def find_matching_brace(text: str, open_brace: int) -> int:
    depth = 0
    for index in range(open_brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unbalanced MLIR braces")


def iter_named_blocks(text: str, pattern: re.Pattern[str]):
    for match in pattern.finditer(text):
        open_brace = text.find("{", match.end() - 1)
        if open_brace == -1:
            raise ValueError(f"missing block body for {match.group('name')}")
        close_brace = find_matching_brace(text, open_brace)
        yield match.group("name"), text[open_brace + 1 : close_brace]


def parse_iree_metadata(text: str) -> dict[str, object]:
    target_match = re.search(r'arch = "([^"]+)"', text)
    if not target_match:
        raise ValueError("missing ROCm target arch")

    export_re = re.compile(
        r"hal\.executable\.export\s+\w+\s+@(?P<symbol>[\w$.-]+)"
        r"\s+ordinal\((?P<ordinal>\d+)\)"
        r"\s+layout\((?P<layout>.*?)\)"
        r"\s+count\(",
        re.S,
    )
    export_matches = list(export_re.finditer(text))
    if not export_matches:
        raise ValueError("missing hal.executable.export")

    dispatch_re = re.compile(
        r"stream\.cmd\.dispatch\s+@(?P<executable>[\w$.-]+)::"
        r"@(?P<variant>[\w$.-]+)::@(?P<symbol>[\w$.-]+)"
    )
    dispatches = {
        match.group("symbol"): {
            "executable": match.group("executable"),
            "variant": match.group("variant"),
            "symbol": match.group("symbol"),
        }
        for match in dispatch_re.finditer(text)
    }

    executable_re = re.compile(
        r"hal\.executable\s+\w+\s+@(?P<name>[\w$.-]+)\s*\{", re.S
    )
    variant_re = re.compile(
        r"hal\.executable\.variant\s+\w+\s+@(?P<name>[\w$.-]+)\s+target", re.S
    )

    executables = []
    for executable_name, executable_body in iter_named_blocks(text, executable_re):
        variant_match = variant_re.search(executable_body)
        if not variant_match:
            raise ValueError(f"missing hal.executable.variant: {executable_name}")
        variant = variant_match.group("name")
        export_matches = list(export_re.finditer(executable_body))
        if not export_matches:
            raise ValueError(f"missing hal.executable.export: {executable_name}")

        exports = []
        for export_match in export_matches:
            attributes_start = export_match.end()
            attributes_match = re.search(
                r"attributes\s+\{(?P<attrs>[^}]*)\}",
                executable_body[attributes_start:],
                re.S,
            )
            if not attributes_match:
                raise ValueError("missing executable export attributes")
            attrs = attributes_match.group("attrs")

            workgroup_match = re.search(r"workgroup_size\s*=\s*\[([^\]]+)\]", attrs)
            if not workgroup_match:
                raise ValueError("missing workgroup_size")
            subgroup_match = re.search(r"subgroup_size\s*=\s*(\d+)\s*:\s*index", attrs)
            if not subgroup_match:
                raise ValueError("missing subgroup_size")

            symbol = export_match.group("symbol")
            kernel_re = re.compile(
                rf"llvm\.func\s+@{re.escape(symbol)}"
                r"\((?P<args>.*?)\)\s+attributes\s+\{(?P<attrs>[^}]*)\}",
                re.S,
            )
            kernel_match = kernel_re.search(executable_body)
            if not kernel_match:
                raise ValueError(f"missing lowered llvm.func kernel: {symbol}")
            dispatch = dispatches.get(symbol)
            if dispatch is None:
                raise ValueError(f"missing stream.cmd.dispatch: {symbol}")

            exports.append(
                {
                    "symbol": symbol,
                    "ordinal": int(export_match.group("ordinal")),
                    "workgroup_size": parse_vector(workgroup_match.group(1)),
                    "subgroup_size": int(subgroup_match.group(1)),
                    "bindings": parse_bindings(export_match.group("layout")),
                    "kernel": {
                        "symbol": symbol,
                        "attributes": sorted(
                            attr
                            for attr in (
                                "rocdl.kernel",
                                "rocdl.flat_work_group_size",
                                "rocdl.reqd_work_group_size",
                                "gpu.known_block_size",
                            )
                            if attr in kernel_match.group("attrs")
                        ),
                    },
                    "dispatch": dispatch,
                }
            )
        executables.append(
            {
                "executable": executable_name,
                "variant": variant,
                "exports": exports,
            }
        )

    if not executables:
        raise ValueError("missing hal.executable")

    return {
        "target": target_match.group(1),
        "executables": executables,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write JSON summary to this path instead of stdout.",
    )
    args = parser.parse_args(argv)

    summary = parse_iree_metadata(args.input.read_text(encoding="utf-8"))
    output = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
