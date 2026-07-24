#!/usr/bin/env python3
"""Create an IREE Qwen decode bundle consumed by lrrt."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

DEFAULT_LAYER_EXPORT = "qwen_decode_layer_kv_cache_max8"
DEFAULT_TAIL_EXPORT = "qwen_decode1_tail"
DEFAULT_MAX_CACHE_TOKENS = 8
DEFAULT_KV_CACHE_DIM = 128


def default_layer_export(max_cache_tokens: int) -> str:
    return f"qwen_decode_layer_kv_cache_max{max_cache_tokens}"


def require_relative_bundle_path(value: str, field: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute():
        raise ValueError(f"{field} must be a non-empty relative path")
    if any(part == ".." for part in path.parts):
        raise ValueError(f"{field} must not contain '..'")
    return path


def copy_vmfb(source: Path, out_dir: Path, relative_name: Path, force: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"VMFB input does not exist: {source}")
    destination = out_dir / relative_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing file: {destination}")
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def manifest_data(args: argparse.Namespace) -> dict[str, object]:
    return {
        "manifest_version": 1,
        "target": args.target,
        "layer_vmfb": args.layer_name.as_posix(),
        "tail_vmfb": args.tail_name.as_posix(),
        "layer_export": args.layer_export,
        "tail_export": args.tail_export,
        "sequence_capacity": args.sequence_capacity,
        "max_cache_tokens": args.max_cache_tokens,
        "kv_cache_shape": [args.max_cache_tokens, args.kv_cache_dim],
    }


def write_manifest(out_dir: Path, data: dict[str, object], force: bool) -> Path:
    manifest = out_dir / "manifest.json"
    if manifest.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing file: {manifest}")
    manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an lrrt IREE Qwen decode bundle from layer and tail VMFB artifacts."
        )
    )
    parser.add_argument("--target", required=True, help="AMD GPU target, e.g. gfx1101")
    parser.add_argument("--layer-vmfb", required=True, type=Path)
    parser.add_argument("--tail-vmfb", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--layer-export")
    parser.add_argument("--tail-export", default=DEFAULT_TAIL_EXPORT)
    parser.add_argument(
        "--max-cache-tokens", default=DEFAULT_MAX_CACHE_TOKENS, type=int
    )
    parser.add_argument(
        "--sequence-capacity",
        type=int,
        help=(
            "Maximum sequence length this bundle supports. Defaults to "
            "--max-cache-tokens for the current static-shape VMFBs."
        ),
    )
    parser.add_argument("--kv-cache-dim", default=DEFAULT_KV_CACHE_DIM, type=int)
    parser.add_argument(
        "--layer-name",
        help="Bundle-relative layer VMFB path. Defaults to the input file name.",
    )
    parser.add_argument(
        "--tail-name",
        help="Bundle-relative tail VMFB path. Defaults to the input file name.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite bundle files")
    args = parser.parse_args(argv)

    if not args.target:
        raise ValueError("--target must not be empty")
    if args.layer_export is None:
        args.layer_export = default_layer_export(args.max_cache_tokens)
    if not args.layer_export:
        raise ValueError("--layer-export must not be empty")
    if not args.tail_export:
        raise ValueError("--tail-export must not be empty")
    if args.max_cache_tokens <= 0:
        raise ValueError("--max-cache-tokens must be positive")
    if args.sequence_capacity is None:
        args.sequence_capacity = args.max_cache_tokens
    if args.sequence_capacity <= 0:
        raise ValueError("--sequence-capacity must be positive")
    if args.sequence_capacity > args.max_cache_tokens:
        raise ValueError("--sequence-capacity must not exceed --max-cache-tokens")
    if args.kv_cache_dim <= 0:
        raise ValueError("--kv-cache-dim must be positive")

    args.layer_name = require_relative_bundle_path(
        args.layer_name if args.layer_name is not None else args.layer_vmfb.name,
        "--layer-name",
    )
    args.tail_name = require_relative_bundle_path(
        args.tail_name if args.tail_name is not None else args.tail_vmfb.name,
        "--tail-name",
    )
    return args


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        copy_vmfb(args.layer_vmfb, args.out_dir, args.layer_name, args.force)
        copy_vmfb(args.tail_vmfb, args.out_dir, args.tail_name, args.force)
        manifest = write_manifest(args.out_dir, manifest_data(args), args.force)
        print(f"wrote IREE Qwen decode bundle: {args.out_dir}")
        print(f"manifest: {manifest}")
        return 0
    except Exception as error:
        print(f"write_iree_qwen_decode_bundle.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
