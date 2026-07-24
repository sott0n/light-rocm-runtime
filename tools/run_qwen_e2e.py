#!/usr/bin/env python3
"""Convert or reuse Qwen bundles, then run a full lrrt E2E check."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = ROOT / "tools" / "run_qwen_benchmark.py"
DEFAULT_CONVERTER = ROOT / "tools" / "convert_qwen_layer.py"
DEFAULT_REQUIREMENTS = ROOT / "tools" / "requirements.txt"
DEFAULT_IREE_BUNDLE_WRITER = ROOT / "tools" / "write_iree_qwen_decode_bundle.py"
DEFAULT_IREE_RUNNER = ROOT / "build-iree" / "adapter" / "lrrt_iree_qwen_decode1_e2e"
DEFAULT_BUNDLE_DIR = Path("/tmp/lrrt-qwen-e2e")
DEFAULT_IREE_DECODE_BUNDLE_DIR = Path("/tmp/lrrt-iree-qwen-decode-bundle")
DEFAULT_IREE_PROBE_DIR = ROOT / "build-iree-probe"
DEFAULT_IREE_TARGET = "gfx1101"
SUPPORTED_IREE_CACHE_CAPACITIES = (8, 16, 32)


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def parse_token_ids(text: str) -> list[int]:
    if not text:
        raise ValueError("--token-ids must not be empty")
    result: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            raise ValueError("--token-ids contains an empty entry")
        value = int(item, 10)
        if value < 0:
            raise ValueError("--token-ids must be non-negative")
        result.append(value)
    return result


def model_layer_count(checkpoint_dir: Path, config: Path | None) -> int:
    config_path = config if config is not None else checkpoint_dir / "config.json"
    data = read_json(config_path)
    value = data.get("num_hidden_layers")
    if not isinstance(value, int) or value <= 0:
        raise ValueError("config field 'num_hidden_layers' must be a positive integer")
    return value


def infer_bundle_layer_count(bundle_dir: Path) -> int:
    count = 0
    while (bundle_dir / f"layer_{count}" / "weights.json").exists():
        count += 1
    return count


def bundle_complete(bundle_dir: Path, layers: int) -> bool:
    if layers <= 0:
        return False
    for layer in range(layers):
        if not (bundle_dir / f"layer_{layer}" / "weights.json").exists():
            return False
    return (bundle_dir / "model_tail" / "weights.json").exists()


def iree_decode_bundle_complete(bundle_dir: Path) -> bool:
    return (bundle_dir / "manifest.json").is_file()


def iree_layer_module_name(max_cache_tokens: int) -> str:
    return f"qwen_decode_layer_kv_cache_max{max_cache_tokens}"


def select_iree_cache_capacity(max_seq_len: int) -> int:
    if max_seq_len <= 0:
        raise ValueError("--max-seq-len must be positive")
    for capacity in SUPPORTED_IREE_CACHE_CAPACITIES:
        if max_seq_len <= capacity:
            return capacity
    supported = ", ".join(str(value) for value in SUPPORTED_IREE_CACHE_CAPACITIES)
    raise ValueError(
        f"--max-seq-len {max_seq_len} exceeds supported IREE cache capacities "
        f"({supported})"
    )


def default_iree_layer_vmfb(
    probe_dir: Path, target: str, max_cache_tokens: int
) -> Path:
    module_name = iree_layer_module_name(max_cache_tokens)
    return probe_dir / module_name / f"{module_name}_{target}.vmfb"


def default_iree_tail_vmfb(probe_dir: Path, target: str) -> Path:
    return probe_dir / "qwen_decode1_tail" / f"qwen_decode1_tail_{target}.vmfb"


def resolve_iree_vmfb_path(
    explicit_path: Path | None, default_path: Path, flag: str
) -> Path:
    if explicit_path is not None:
        return explicit_path
    if default_path.is_file():
        return default_path
    raise ValueError(
        f"{flag} was not provided and default VMFB was not found: {default_path}"
    )


def resolve_layers(args: argparse.Namespace) -> int:
    if args.layers is not None:
        if args.layers <= 0:
            raise ValueError("--layers must be positive")
        return args.layers
    if args.checkpoint_dir is not None:
        return model_layer_count(args.checkpoint_dir, args.config)
    inferred = infer_bundle_layer_count(args.bundle_dir)
    if inferred <= 0:
        raise ValueError("cannot infer layers; pass --layers or --checkpoint-dir")
    return inferred


def converter_command(args: argparse.Namespace, layers: int) -> list[str]:
    converter_args = [
        str(args.converter),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--layer",
        "0",
        "--layer-count",
        str(layers),
        "--keys",
        str(args.keys),
        "--token-ids",
        args.token_ids,
        "--output",
        str(args.bundle_dir),
    ]
    if args.backend == "iree":
        converter_args.extend(["--bundle-directory", "--full-token-embeddings"])
    if args.config is not None:
        converter_args.extend(["--config", str(args.config)])

    if args.python is not None:
        return [str(args.python), *converter_args]
    if args.no_uv or shutil.which("uv") is None:
        return [sys.executable, *converter_args]
    return [
        "uv",
        "run",
        "--with-requirements",
        str(args.requirements),
        "python",
        *converter_args,
    ]


def runner_command(args: argparse.Namespace, layers: int) -> list[str]:
    token_ids = parse_token_ids(args.token_ids)
    if args.keys < len(token_ids):
        raise ValueError("--keys must be at least the number of --token-ids")

    command = [
        str(args.python if args.python is not None else sys.executable),
        str(args.runner),
        "--bundle-dir",
        str(args.bundle_dir),
        "--layers",
        str(layers),
        "--keys",
        str(args.keys),
        "--token-ids",
        args.token_ids,
        "--iterations",
        str(args.iterations),
        "--valid-keys",
        str(len(token_ids)),
        "--no-warmup",
        "--e2e-check",
        "--sync-stack",
    ]
    if args.checkpoint_dir is not None:
        command.extend(["--checkpoint-dir", str(args.checkpoint_dir)])
    if args.config is not None:
        command.extend(["--config", str(args.config)])
    if args.force_convert:
        command.append("--force-convert")
    if args.no_convert:
        command.append("--no-convert")
    if args.no_uv:
        command.append("--no-uv")
    if args.trace_setup:
        command.append("--trace-setup")
    if args.trace_run:
        command.append("--trace-run")
    if args.benchmark is not None:
        command.extend(["--benchmark", str(args.benchmark)])
    return command


def iree_bundle_writer_command(args: argparse.Namespace) -> list[str]:
    cache_capacity = select_iree_cache_capacity(args.max_seq_len)
    layer_vmfb = resolve_iree_vmfb_path(
        args.iree_layer_vmfb,
        default_iree_layer_vmfb(args.iree_probe_dir, args.iree_target, cache_capacity),
        "--iree-layer-vmfb",
    )
    tail_vmfb = resolve_iree_vmfb_path(
        args.iree_tail_vmfb,
        default_iree_tail_vmfb(args.iree_probe_dir, args.iree_target),
        "--iree-tail-vmfb",
    )
    command = [
        str(args.python if args.python is not None else sys.executable),
        str(args.iree_bundle_writer),
        "--target",
        args.iree_target,
        "--layer-vmfb",
        str(layer_vmfb),
        "--tail-vmfb",
        str(tail_vmfb),
        "--out-dir",
        str(args.iree_decode_bundle_dir),
        "--max-cache-tokens",
        str(cache_capacity),
        "--layer-export",
        iree_layer_module_name(cache_capacity),
    ]
    if args.force_iree_bundle:
        command.append("--force")
    return command


def iree_runner_command(args: argparse.Namespace, layers: int) -> list[str]:
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_new_tokens > args.max_seq_len:
        raise ValueError("--max-new-tokens must not exceed --max-seq-len")
    return [
        str(args.iree_runner),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--bundle",
        str(args.iree_decode_bundle_dir),
        str(args.bundle_dir),
        str(layers),
    ]


def run_command(command: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def run_triton(args: argparse.Namespace, layers: int) -> None:
    run_command(runner_command(args, layers), args.dry_run)


def run_iree(args: argparse.Namespace, layers: int) -> None:
    complete = bundle_complete(args.bundle_dir, layers)
    should_convert = args.force_convert or not complete
    if args.no_convert:
        should_convert = False
    if should_convert:
        if args.checkpoint_dir is None:
            raise ValueError(
                "weight bundle is incomplete; pass --checkpoint-dir or disable "
                "conversion with --no-convert"
            )
        run_command(converter_command(args, layers), args.dry_run)
    elif complete:
        print(f"reusing Qwen mini bundle: {args.bundle_dir}", flush=True)
    else:
        print(f"running with unchecked Qwen mini bundle: {args.bundle_dir}", flush=True)

    should_write_bundle = args.force_iree_bundle or not iree_decode_bundle_complete(
        args.iree_decode_bundle_dir
    )
    if args.no_iree_bundle_write:
        should_write_bundle = False
    if should_write_bundle:
        run_command(iree_bundle_writer_command(args), args.dry_run)
    elif iree_decode_bundle_complete(args.iree_decode_bundle_dir):
        print(
            f"reusing IREE Qwen decode bundle: {args.iree_decode_bundle_dir}",
            flush=True,
        )
    else:
        print(
            f"running with unchecked IREE decode bundle: {args.iree_decode_bundle_dir}",
            flush=True,
        )

    run_command(iree_runner_command(args, layers), args.dry_run)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen mini E2E through lrrt: checkpoint conversion, decoder "
            "layer stack, final RMSNorm, lm_head logits, and finite-logit check."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bundle-dir", default=DEFAULT_BUNDLE_DIR, type=Path)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--keys", default=4, type=int)
    parser.add_argument("--token-ids", default="0,1,2")
    parser.add_argument("--iterations", default=1, type=int)
    parser.add_argument(
        "--max-new-tokens",
        default=1,
        type=int,
        help="maximum number of output tokens to decode",
    )
    parser.add_argument("--backend", choices=["triton", "iree"], default="triton")
    parser.add_argument("--iree", action="store_const", const="iree", dest="backend")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--no-convert", action="store_true")
    parser.add_argument("--no-uv", action="store_true")
    parser.add_argument("--trace-setup", action="store_true")
    parser.add_argument("--trace-run", action="store_true")
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--runner", default=DEFAULT_RUNNER, type=Path)
    parser.add_argument("--converter", default=DEFAULT_CONVERTER, type=Path)
    parser.add_argument("--requirements", default=DEFAULT_REQUIREMENTS, type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--iree-target", default=DEFAULT_IREE_TARGET)
    parser.add_argument("--iree-probe-dir", default=DEFAULT_IREE_PROBE_DIR, type=Path)
    parser.add_argument("--iree-layer-vmfb", type=Path)
    parser.add_argument("--iree-tail-vmfb", type=Path)
    parser.add_argument(
        "--iree-decode-bundle-dir", default=DEFAULT_IREE_DECODE_BUNDLE_DIR, type=Path
    )
    parser.add_argument(
        "--iree-bundle-writer", default=DEFAULT_IREE_BUNDLE_WRITER, type=Path
    )
    parser.add_argument("--iree-runner", default=DEFAULT_IREE_RUNNER, type=Path)
    parser.add_argument("--max-seq-len", default=8, type=int)
    parser.add_argument("--force-iree-bundle", action="store_true")
    parser.add_argument("--no-iree-bundle-write", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        layers = resolve_layers(args)
        if args.backend == "iree":
            run_iree(args, layers)
        else:
            run_triton(args, layers)
        return 0
    except subprocess.CalledProcessError as error:
        return error.returncode
    except Exception as error:
        print(f"run_qwen_e2e.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
