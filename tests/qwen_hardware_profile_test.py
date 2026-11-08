#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import math
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILER = ROOT / "tools" / "profile_qwen_counters.py"
FIELDNAMES = (
    "Process_Id",
    "Dispatch_Id",
    "Kernel_Name",
    "Counter_Name",
    "Counter_Value",
    "Start_Timestamp",
    "End_Timestamp",
)


def load_profiler():
    spec = importlib.util.spec_from_file_location("profile_qwen_counters", PROFILER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def make_rows(
    metrics: dict[str, tuple[float, float]], durations: tuple[int, int]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_index, dispatch_id in enumerate((7, 9)):
        for name, values in metrics.items():
            rows.append(
                {
                    "Process_Id": 42,
                    "Dispatch_Id": dispatch_id,
                    "Kernel_Name": "qwen_stateful_graph_compiled_kernel",
                    "Counter_Name": name,
                    "Counter_Value": values[sample_index],
                    "Start_Timestamp": 1000,
                    "End_Timestamp": 1000 + durations[sample_index],
                }
            )
    return rows


def test_parse_and_summarize() -> None:
    profiler = load_profiler()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        read_path = root / "read.csv"
        cache_path = root / "cache.csv"
        execution_path = root / "execution.csv"
        write_csv(
            read_path,
            make_rows({"FETCH_SIZE": (1000.0, 3000.0)}, (1000, 3000)),
        )
        write_csv(
            cache_path,
            make_rows(
                {"WRITE_SIZE": (10.0, 30.0), "L2CacheHit": (20.0, 40.0)},
                (1000, 3000),
            ),
        )
        write_csv(
            execution_path,
            make_rows(
                {
                    "MemUnitBusy": (70.0, 90.0),
                    "WAVE_DEP_WAIT": (80.0, 100.0),
                    "WAVE_ISSUE_WAIT": (1.0, 3.0),
                    "OccupancyPercent": (5.0, 7.0),
                    "GPUBusy": (99.0, 100.0),
                },
                (1000, 3000),
            ),
        )
        profiles = {
            "dram_read": profiler.read_counter_csv(read_path, ("FETCH_SIZE",)),
            "cache_write": profiler.read_counter_csv(
                cache_path, ("WRITE_SIZE", "L2CacheHit")
            ),
            "execution": profiler.read_counter_csv(
                execution_path,
                (
                    "MemUnitBusy",
                    "WAVE_DEP_WAIT",
                    "WAVE_ISSUE_WAIT",
                    "OccupancyPercent",
                    "GPUBusy",
                ),
            ),
        }
        summary = profiler.summarize(profiles, 2000.0, 2.0)

    assert summary["captured_dispatches"] == {
        "dram_read": 2,
        "cache_write": 2,
        "execution": 2,
    }
    assert summary["selected_dispatches"] == {
        "dram_read": 1,
        "cache_write": 1,
        "execution": 1,
    }
    assert summary["dispatch_duration_ms"] == 0.002
    assert summary["l2_cache_hit_percent"] == 40.0
    assert summary["memory_unit_busy_percent"] == 90.0
    assert summary["wave_dependency_wait_percent"] == 100.0
    assert summary["wave_issue_wait_percent"] == 3.0
    assert summary["occupancy_percent"] == 7.0
    assert math.isclose(summary["dram_read_gbps"], 1536.0)
    assert math.isclose(summary["dram_write_gbps"], 15.36)
    assert math.isclose(summary["counter_traffic_floor_ms"], 1.55136)


def test_profiler_command_keeps_counter_passes_separate() -> None:
    profiler = load_profiler()
    args = Namespace(
        rocprofv3="/opt/rocm/bin/rocprofv3",
        benchmark=Path("/tmp/qwen-benchmark"),
        steps=16,
        trials=1,
        workers=20,
        weights_dir=Path("/tmp/qwen-weights"),
        layers=24,
    )
    command = profiler.profiler_command(
        args, profiler.COUNTER_PASSES[0], Path("/tmp/output")
    )
    pmc_index = command.index("--pmc")
    assert command[pmc_index : pmc_index + 2] == ["--pmc", "FETCH_SIZE"]
    assert "WRITE_SIZE" not in command
    assert "L2CacheHit" not in command
    assert command[-7:] == [
        "16",
        "1",
        "20",
        "--weights-dir",
        "/tmp/qwen-weights",
        "--layers",
        "24",
    ]


def test_baseline_command_overrides_profiler_trial_count() -> None:
    profiler = load_profiler()
    args = Namespace(
        benchmark=Path("/tmp/qwen-benchmark"),
        steps=16,
        trials=1,
        workers=20,
        weights_dir=Path("/tmp/qwen-weights"),
        layers=24,
    )
    assert profiler.benchmark_command(args, trials=3)[1:4] == ["16", "3", "20"]


def main() -> int:
    test_parse_and_summarize()
    test_profiler_command_keeps_counter_passes_separate()
    test_baseline_command_overrides_profiler_trial_count()
    print("qwen_hardware_profile_test: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
