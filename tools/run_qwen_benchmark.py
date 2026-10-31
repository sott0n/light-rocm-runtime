#!/usr/bin/env python3
"""Convert or reuse Qwen mini bundles, then run the lrrt benchmark."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONVERTER = ROOT / "tools" / "convert_qwen_layer.py"
DEFAULT_REQUIREMENTS = ROOT / "tools" / "requirements.txt"
DEFAULT_BENCHMARK = (
    ROOT / "build-triton-bench" / "lrrt_triton_mini_decoder_layer_benchmark"
)


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


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


def tail_has_full_embeddings(bundle_dir: Path) -> bool:
    manifest = bundle_dir / "model_tail" / "weights.json"
    if not manifest.is_file():
        return False
    data = read_json(manifest)
    vocab = data.get("vocab")
    token_ids = data.get("token_ids")
    return (
        isinstance(vocab, int)
        and vocab > 0
        and isinstance(token_ids, list)
        and len(token_ids) == vocab
        and token_ids[0] == 0
        and token_ids[-1] == vocab - 1
    )


def bundle_complete(
    bundle_dir: Path,
    layers: int,
    require_tail: bool,
    require_full_embeddings: bool = False,
) -> bool:
    if layers <= 0:
        return False
    for layer in range(layers):
        if not (bundle_dir / f"layer_{layer}" / "weights.json").exists():
            return False
    if require_tail and not (bundle_dir / "model_tail" / "weights.json").exists():
        return False
    if require_full_embeddings and not tail_has_full_embeddings(bundle_dir):
        return False
    return True


def converter_command(args: argparse.Namespace, layers: int) -> list[str]:
    converter_args = [
        str(args.converter),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--layer",
        str(args.layer),
        "--layer-count",
        str(layers),
        "--keys",
        str(args.keys),
        "--token-ids",
        args.token_ids,
        "--output",
        str(args.bundle_dir),
    ]
    if args.config is not None:
        converter_args.extend(["--config", str(args.config)])
    if args.max_new_tokens > 1:
        converter_args.append("--full-token-embeddings")

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


def benchmark_command(args: argparse.Namespace, layers: int) -> list[str]:
    command = [
        str(args.benchmark),
        str(args.iterations),
        "--weights-dir",
        str(args.bundle_dir),
        "--layers",
        str(layers),
    ]
    if args.valid_keys is not None:
        command.extend(["--valid-keys", str(args.valid_keys)])
    if args.no_warmup:
        command.append("--no-warmup")
    if args.no_model_tail:
        command.append("--no-model-tail")
    if args.e2e_check:
        command.append("--e2e-check")
    if args.max_new_tokens > 0:
        command.extend(
            [
                "--prompt-token-ids",
                args.token_ids,
                "--max-new-tokens",
                str(args.max_new_tokens),
            ]
        )
    if args.sync_stack:
        command.append("--sync-stack")
    if args.layer_sweep:
        command.append("--layer-sweep")
    if args.trace_setup:
        command.append("--trace-setup")
    if args.trace_run:
        command.append("--trace-run")
    return command


def resolve_layers(args: argparse.Namespace) -> int:
    if args.layer != 0:
        raise ValueError(
            "run_qwen_benchmark.py currently requires --layer 0 because the "
            "benchmark weights-dir loader expects layer_0..layer_N"
        )
    if args.all_layers:
        if args.checkpoint_dir is None:
            raise ValueError("--all-layers requires --checkpoint-dir")
        return model_layer_count(args.checkpoint_dir, args.config) - args.layer
    if args.layers is not None:
        return args.layers
    inferred = infer_bundle_layer_count(args.bundle_dir)
    if inferred > 0:
        return inferred
    raise ValueError("pass --layers, --all-layers, or provide an existing bundle dir")


def run_command(command: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the lrrt Qwen mini decoder benchmark from an existing bundle "
            "directory, or convert a local checkpoint first."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--layer", default=0, type=int)
    parser.add_argument("--keys", default=4, type=int)
    parser.add_argument("--token-ids", default="0,1,2")
    parser.add_argument("--iterations", default=1, type=int)
    parser.add_argument("--max-new-tokens", default=0, type=int)
    parser.add_argument("--valid-keys", type=int)
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--no-convert", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--no-model-tail", action="store_true")
    parser.add_argument("--e2e-check", action="store_true")
    parser.add_argument("--sync-stack", action="store_true")
    parser.add_argument("--layer-sweep", action="store_true")
    parser.add_argument("--trace-setup", action="store_true")
    parser.add_argument("--trace-run", action="store_true")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, type=Path)
    parser.add_argument("--converter", default=DEFAULT_CONVERTER, type=Path)
    parser.add_argument("--requirements", default=DEFAULT_REQUIREMENTS, type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--no-uv", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        layers = resolve_layers(args)
        if layers <= 0:
            raise ValueError("layer count must be positive")
        if args.max_new_tokens < 0:
            raise ValueError("--max-new-tokens must not be negative")
        if args.max_new_tokens > 0 and args.no_model_tail:
            raise ValueError("--max-new-tokens requires model tail")
        prompt_tokens = [part for part in args.token_ids.split(",") if part]
        if args.max_new_tokens > 0 and (
            len(prompt_tokens) + args.max_new_tokens - 1 > args.keys
        ):
            raise ValueError(
                "--keys must cover the prompt and generated-token feedback"
            )
        require_tail = not args.no_model_tail
        complete = bundle_complete(
            args.bundle_dir,
            layers,
            require_tail,
            require_full_embeddings=args.max_new_tokens > 1,
        )
        should_convert = args.force_convert or not complete
        if args.no_convert:
            should_convert = False
        if should_convert:
            if args.checkpoint_dir is None:
                raise ValueError(
                    "bundle is incomplete; pass --checkpoint-dir or disable "
                    "conversion with --no-convert"
                )
            run_command(converter_command(args, layers), args.dry_run)
        elif complete:
            print(f"reusing Qwen mini bundle: {args.bundle_dir}", flush=True)
        else:
            print(f"running with unchecked bundle dir: {args.bundle_dir}", flush=True)
        run_command(benchmark_command(args, layers), args.dry_run)
        return 0
    except subprocess.CalledProcessError as error:
        return error.returncode
    except Exception as error:
        print(f"run_qwen_benchmark.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
