#!/usr/bin/env python3
"""Convert or reuse Qwen bundles, then run a full lrrt E2E check."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = ROOT / "tools" / "run_qwen_benchmark.py"
DEFAULT_BUNDLE_DIR = Path("/tmp/lrrt-qwen-e2e")


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


def runner_command(args: argparse.Namespace, layers: int) -> list[str]:
    token_ids = parse_token_ids(args.token_ids)
    if args.keys < len(token_ids):
        raise ValueError("--keys must be at least the number of --token-ids")

    command = [
        str(args.python),
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
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--no-convert", action="store_true")
    parser.add_argument("--no-uv", action="store_true")
    parser.add_argument("--trace-setup", action="store_true")
    parser.add_argument("--trace-run", action="store_true")
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--runner", default=DEFAULT_RUNNER, type=Path)
    parser.add_argument("--python", default=Path(sys.executable), type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        layers = resolve_layers(args)
        command = runner_command(args, layers)
        print("+ " + " ".join(str(part) for part in command), flush=True)
        if args.dry_run:
            return 0
        subprocess.run(command, check=True)
        return 0
    except subprocess.CalledProcessError as error:
        return error.returncode
    except Exception as error:
        print(f"run_qwen_e2e.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
