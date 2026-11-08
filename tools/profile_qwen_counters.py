#!/usr/bin/env python3
"""Collect rocprofv3 counters for the compiled Qwen resident kernel."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "build" / "lrrt_qwen_stateful_graph_benchmark"
KERNEL_NAME = "qwen_stateful_graph_compiled_kernel"


@dataclass(frozen=True)
class CounterPass:
    name: str
    counters: tuple[str, ...]


COUNTER_PASSES = (
    CounterPass("dram_read", ("FETCH_SIZE",)),
    CounterPass("cache_write", ("WRITE_SIZE", "L2CacheHit")),
    CounterPass(
        "execution",
        (
            "MemUnitBusy",
            "WAVE_DEP_WAIT",
            "WAVE_ISSUE_WAIT",
            "OccupancyPercent",
            "GPUBusy",
        ),
    ),
)


@dataclass
class DispatchSample:
    dispatch_id: int
    duration_ns: float
    counters: dict[str, float]


def profiler_command(
    args: argparse.Namespace, counter_pass: CounterPass, output_dir: Path
) -> list[str]:
    return [
        args.rocprofv3,
        "--pmc",
        *counter_pass.counters,
        "--kernel-include-regex",
        f"^{KERNEL_NAME}$",
        "--output-format",
        "csv",
        "--output-directory",
        str(output_dir),
        "--output-file",
        "qwen",
        "--",
        *benchmark_command(args),
    ]


def benchmark_command(args: argparse.Namespace, trials: int | None = None) -> list[str]:
    return [
        str(args.benchmark),
        str(args.steps),
        str(args.trials if trials is None else trials),
        str(args.workers),
        "--weights-dir",
        str(args.weights_dir),
        "--layers",
        str(args.layers),
    ]


def read_counter_csv(
    path: Path, expected_counters: tuple[str, ...]
) -> list[DispatchSample]:
    grouped: dict[tuple[int, int], DispatchSample] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {
            "Process_Id",
            "Dispatch_Id",
            "Kernel_Name",
            "Counter_Name",
            "Counter_Value",
            "Start_Timestamp",
            "End_Timestamp",
        }
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"{path} is not a rocprofv3 counter CSV")
        for row in reader:
            if row["Kernel_Name"] != KERNEL_NAME:
                continue
            process_id = int(row["Process_Id"])
            dispatch_id = int(row["Dispatch_Id"])
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
            if end <= start:
                raise ValueError(
                    f"dispatch {dispatch_id} has an invalid timestamp range"
                )
            key = (process_id, dispatch_id)
            sample = grouped.setdefault(
                key,
                DispatchSample(
                    dispatch_id=dispatch_id,
                    duration_ns=float(end - start),
                    counters={},
                ),
            )
            if sample.duration_ns != float(end - start):
                raise ValueError(f"dispatch {dispatch_id} has inconsistent timestamps")
            counter = row["Counter_Name"]
            value = float(row["Counter_Value"])
            if not math.isfinite(value):
                raise ValueError(f"dispatch {dispatch_id} has a non-finite {counter}")
            if counter in sample.counters:
                raise ValueError(f"dispatch {dispatch_id} repeats counter {counter}")
            sample.counters[counter] = value

    expected = set(expected_counters)
    samples = sorted(grouped.values(), key=lambda sample: sample.dispatch_id)
    if not samples:
        raise ValueError(f"{path} contains no {KERNEL_NAME} samples")
    for sample in samples:
        missing = expected.difference(sample.counters)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"dispatch {sample.dispatch_id} is missing {names}")
    return samples


def median_counter(samples: list[DispatchSample], name: str) -> float:
    return statistics.median(sample.counters[name] for sample in samples)


def select_full_workload(
    profiles: dict[str, list[DispatchSample]], threshold: float = 0.9
) -> dict[str, list[DispatchSample]]:
    read_samples = profiles["dram_read"]
    maximum_fetch = max(sample.counters["FETCH_SIZE"] for sample in read_samples)
    selected_ids = {
        sample.dispatch_id
        for sample in read_samples
        if sample.counters["FETCH_SIZE"] >= threshold * maximum_fetch
    }
    if not selected_ids:
        raise ValueError("no full-workload dispatches were identified")
    selected: dict[str, list[DispatchSample]] = {}
    for name, samples in profiles.items():
        by_id = {sample.dispatch_id: sample for sample in samples}
        missing = selected_ids.difference(by_id)
        if missing:
            ids = ", ".join(str(value) for value in sorted(missing))
            raise ValueError(f"counter pass {name} is missing dispatches {ids}")
        selected[name] = [by_id[dispatch_id] for dispatch_id in sorted(selected_ids)]
    return selected


def summarize(
    profiles: dict[str, list[DispatchSample]],
    baseline_duration_ns: float,
    memory_bandwidth_gbps: float | None,
) -> dict[str, object]:
    captured_counts = {name: len(samples) for name, samples in profiles.items()}
    profiles = select_full_workload(profiles)
    read_samples = profiles["dram_read"]
    cache_samples = profiles["cache_write"]
    execution_samples = profiles["execution"]
    read_bytes = median_counter(read_samples, "FETCH_SIZE") * 1024.0
    write_bytes = median_counter(cache_samples, "WRITE_SIZE") * 1024.0
    # bytes/ns is numerically decimal GB/s. Use the unprofiled duration because
    # collecting all four GL2 read-size counters materially slows this kernel.
    read_gbps = read_bytes / baseline_duration_ns
    write_gbps = write_bytes / baseline_duration_ns
    total_gbps = read_gbps + write_gbps
    profiled_dispatch_ns = statistics.median(
        sample.duration_ns for sample in read_samples
    )

    summary: dict[str, object] = {
        "captured_dispatches": captured_counts,
        "selected_dispatches": {
            name: len(samples) for name, samples in profiles.items()
        },
        "dispatch_duration_ms": baseline_duration_ns / 1.0e6,
        "fetch_profile_dispatch_duration_ms": profiled_dispatch_ns / 1.0e6,
        "dram_read_gb": read_bytes / 1.0e9,
        "dram_write_gb": write_bytes / 1.0e9,
        "dram_read_gbps": read_gbps,
        "dram_write_gbps": write_gbps,
        "dram_total_gbps": total_gbps,
        "l2_cache_hit_percent": median_counter(cache_samples, "L2CacheHit"),
        "memory_unit_busy_percent": median_counter(execution_samples, "MemUnitBusy"),
        "wave_dependency_wait_percent": median_counter(
            execution_samples, "WAVE_DEP_WAIT"
        ),
        "wave_issue_wait_percent": median_counter(execution_samples, "WAVE_ISSUE_WAIT"),
        "occupancy_percent": median_counter(execution_samples, "OccupancyPercent"),
        "gpu_busy_percent": median_counter(execution_samples, "GPUBusy"),
    }
    if memory_bandwidth_gbps is not None:
        traffic_bytes = read_bytes + write_bytes
        floor_ns = traffic_bytes / memory_bandwidth_gbps
        summary.update(
            {
                "memory_bandwidth_roof_gbps": memory_bandwidth_gbps,
                "dram_bandwidth_utilization_percent": (
                    100.0 * total_gbps / memory_bandwidth_gbps
                ),
                "counter_traffic_floor_ms": floor_ns / 1.0e6,
                "dispatch_over_counter_floor": baseline_duration_ns / floor_ns,
            }
        )
    return summary


def print_summary(summary: dict[str, object]) -> None:
    captured = summary["captured_dispatches"]
    selected = summary["selected_dispatches"]
    assert isinstance(captured, dict)
    assert isinstance(selected, dict)
    print("\nQwen compiled resident hardware profile")
    print("=======================================")
    print(
        "Captured dispatches:        "
        + ", ".join(f"{name}={count}" for name, count in captured.items())
    )
    print(
        "Selected full dispatches:   "
        + ", ".join(f"{name}={count}" for name, count in selected.items())
    )
    print(f"Unprofiled compiled p50:    {summary['dispatch_duration_ms']:9.3f} ms")
    print(
        f"FETCH_SIZE profiled time:   "
        f"{summary['fetch_profile_dispatch_duration_ms']:9.3f} ms"
    )
    print(
        f"DRAM read:                  {summary['dram_read_gb']:9.3f} GB, "
        f"{summary['dram_read_gbps']:9.3f} GB/s"
    )
    print(
        f"DRAM write:                 {summary['dram_write_gb']:9.6f} GB, "
        f"{summary['dram_write_gbps']:9.3f} GB/s"
    )
    print(f"Total measured DRAM rate:   {summary['dram_total_gbps']:9.3f} GB/s")
    if "memory_bandwidth_roof_gbps" in summary:
        print(
            f"DRAM bandwidth utilization: {summary['dram_bandwidth_utilization_percent']:9.3f}% "
            f"of {summary['memory_bandwidth_roof_gbps']:.3f} GB/s"
        )
        print(
            f"Counter-traffic floor:      {summary['counter_traffic_floor_ms']:9.3f} ms, "
            f"dispatch/floor {summary['dispatch_over_counter_floor']:.2f}x"
        )
    print(f"L2 cache hit:               {summary['l2_cache_hit_percent']:9.3f}%")
    print(f"Memory unit busy:           {summary['memory_unit_busy_percent']:9.3f}%")
    print(
        f"Wave dependency wait:       {summary['wave_dependency_wait_percent']:9.3f}%"
    )
    print(f"Wave issue wait:            {summary['wave_issue_wait_percent']:9.3f}%")
    print(f"Occupancy:                  {summary['occupancy_percent']:9.3f}%")
    print(f"GPU busy:                   {summary['gpu_busy_percent']:9.3f}%")


def run_profiles(
    args: argparse.Namespace, output_root: Path
) -> dict[str, list[DispatchSample]]:
    profiles: dict[str, list[DispatchSample]] = {}
    for counter_pass in COUNTER_PASSES:
        pass_dir = output_root / counter_pass.name
        if pass_dir.exists() and any(pass_dir.iterdir()):
            raise ValueError(f"refusing to overwrite non-empty {pass_dir}")
        pass_dir.mkdir(parents=True, exist_ok=True)
        command = profiler_command(args, counter_pass, pass_dir)
        print(
            f"collecting {counter_pass.name}: " + ", ".join(counter_pass.counters),
            flush=True,
        )
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            if result.stdout:
                print(result.stdout, file=sys.stderr, end="")
            if result.stderr:
                print(result.stderr, file=sys.stderr, end="")
            raise subprocess.CalledProcessError(result.returncode, command)
        csv_path = pass_dir / "qwen_counter_collection.csv"
        profiles[counter_pass.name] = read_counter_csv(csv_path, counter_pass.counters)
    return profiles


def run_baseline(args: argparse.Namespace) -> float:
    print(
        f"measuring unprofiled compiled baseline ({args.baseline_trials}-trial p50)",
        flush=True,
    )
    command = benchmark_command(args, trials=args.baseline_trials)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        raise subprocess.CalledProcessError(result.returncode, command)
    match = re.search(
        r"^Compiled resident graph\s+([0-9]+(?:\.[0-9]+)?)\s+"
        r"([0-9]+(?:\.[0-9]+)?)\s+",
        result.stdout,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError("benchmark output has no compiled resident timing")
    return float(match.group(2)) * 1.0e3


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run three compatible rocprofv3 counter passes over the compiled "
            "Qwen resident kernel and summarize its memory bottleneck."
        )
    )
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, type=Path)
    parser.add_argument("--rocprofv3", default="rocprofv3")
    parser.add_argument("--steps", default=16, type=positive_int)
    parser.add_argument("--trials", default=1, type=positive_int)
    parser.add_argument("--baseline-trials", default=3, type=positive_int)
    parser.add_argument("--workers", default=20, type=positive_int)
    parser.add_argument("--layers", default=24, type=positive_int)
    parser.add_argument("--memory-bandwidth-gbps", type=positive_float)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="preserve raw rocprofv3 CSVs instead of using a temporary directory",
    )
    parser.add_argument("--json", type=Path, help="write the aggregate summary as JSON")
    return parser.parse_args(argv)


def execute(args: argparse.Namespace, output_root: Path) -> dict[str, object]:
    if not args.benchmark.is_file():
        raise ValueError(f"benchmark does not exist: {args.benchmark}")
    if not args.weights_dir.is_dir():
        raise ValueError(f"weights directory does not exist: {args.weights_dir}")
    if shutil.which(args.rocprofv3) is None:
        raise ValueError(f"rocprofv3 executable not found: {args.rocprofv3}")
    profiles = run_profiles(args, output_root)
    # Run this after the longer counter passes so the p50 is not biased by the
    # large clock ramp seen on short runs from an idle GPU.
    baseline_duration_ns = run_baseline(args)
    return summarize(profiles, baseline_duration_ns, args.memory_bandwidth_gbps)


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            summary = execute(args, args.output_dir)
        else:
            with tempfile.TemporaryDirectory(prefix="lrrt-qwen-counters-") as tmpdir:
                summary = execute(args, Path(tmpdir))
        print_summary(summary)
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return 0
    except subprocess.CalledProcessError as error:
        print(
            f"profile_qwen_counters.py: profiler pass failed with status "
            f"{error.returncode}",
            file=sys.stderr,
        )
        return error.returncode
    except Exception as error:
        print(f"profile_qwen_counters.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
