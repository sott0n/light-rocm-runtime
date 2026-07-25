#!/usr/bin/env python3

import argparse
from pathlib import Path

BEGIN_MARKERS = {
    "// QWEN_MATMUL_BEGIN": "matmul",
    "// QWEN_REDUCED_BEGIN": "reduced",
    "// QWEN_MIXED_BEGIN": "mixed",
}
END_MARKERS = {
    "// QWEN_MATMUL_END": "matmul",
    "// QWEN_REDUCED_END": "reduced",
    "// QWEN_MIXED_END": "mixed",
}


def select_variant(source: str, variant: str) -> str:
    output: list[str] = []
    section: str | None = None
    seen_sections = 0
    for line in source.splitlines(keepends=True):
        marker = line.strip()
        if marker in BEGIN_MARKERS:
            if section is not None:
                raise ValueError("nested Qwen linear variant section")
            section = BEGIN_MARKERS[marker]
            seen_sections += 1
            continue
        if marker in END_MARKERS:
            if section != END_MARKERS[marker]:
                raise ValueError("mismatched Qwen linear variant section")
            section = None
            continue
        if section is None or section == variant:
            output.append(line)
    if section is not None:
        raise ValueError("unterminated Qwen linear variant section")
    if seen_sections == 0:
        raise ValueError("no Qwen linear variant sections found")
    return "".join(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("matmul", "reduced", "mixed"), required=True
    )
    args = parser.parse_args()

    source = args.input.read_text(encoding="utf-8")
    args.output.write_text(select_variant(source, args.variant), encoding="utf-8")


if __name__ == "__main__":
    main()
