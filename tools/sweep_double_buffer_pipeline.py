#!/usr/bin/env python3
"""Sweep pinned-host pipeline sizes and compute rounds into one CSV table."""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "build-bench" / "lrrt_double_buffer_pipeline_benchmark"

CSV_FIELDS = [
    "chunk_size_mib",
    "compute_rounds",
    "chunks",
    "sequential_ms",
    "double_buffered_ms",
    "speedup",
    "prepare_ms_per_chunk",
    "h2d_ms_per_chunk",
    "gpu_stage_ms_per_chunk",
    "slot_wait_ms_per_chunk",
    "final_drain_ms",
]


def positive_int_list(text: str) -> list[int]:
    try:
        values = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "values must be comma-separated integers"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def benchmark_command(
    benchmark: Path,
    chunk_size_mib: int,
    compute_rounds: int,
    chunks: int,
    warmup_chunks: int,
) -> list[str]:
    return [
        str(benchmark),
        "--chunks",
        str(chunks),
        "--warmup-chunks",
        str(warmup_chunks),
        "--chunk-size-mib",
        str(chunk_size_mib),
        "--compute-rounds",
        str(compute_rounds),
        "--csv",
    ]


def run_case(command: list[str]) -> dict[str, str]:
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if len(rows) != 1 or list(rows[0]) != CSV_FIELDS:
        raise ValueError("benchmark did not return one valid CSV row")
    return rows[0]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the pinned-host double-buffer pipeline across chunk sizes "
            "and GPU compute rounds."
        )
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--chunk-sizes-mib", type=positive_int_list, default=[1, 4, 16])
    parser.add_argument(
        "--compute-rounds", type=positive_int_list, default=[1, 16, 64, 256]
    )
    parser.add_argument("--chunks", type=int, default=50)
    parser.add_argument("--warmup-chunks", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        if args.chunks <= 0 or args.warmup_chunks <= 0:
            raise ValueError("--chunks and --warmup-chunks must be positive")

        rows = []
        for chunk_size_mib in args.chunk_sizes_mib:
            for compute_rounds in args.compute_rounds:
                command = benchmark_command(
                    args.benchmark,
                    chunk_size_mib,
                    compute_rounds,
                    args.chunks,
                    args.warmup_chunks,
                )
                print(
                    f"measuring {chunk_size_mib} MiB x {compute_rounds} rounds",
                    file=sys.stderr,
                    flush=True,
                )
                rows.append(run_case(command))

        output = (
            args.output.open("w", encoding="utf-8", newline="")
            if args.output
            else sys.stdout
        )
        try:
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        finally:
            if args.output:
                output.close()
        return 0
    except subprocess.CalledProcessError as error:
        return error.returncode
    except Exception as error:
        print(f"sweep_double_buffer_pipeline.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
