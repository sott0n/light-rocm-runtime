#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "tools" / "sweep_double_buffer_pipeline.py"


def load_sweep():
    spec = importlib.util.spec_from_file_location("sweep_double_buffer_pipeline", SWEEP)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_positive_int_list() -> None:
    sweep = load_sweep()
    assert sweep.positive_int_list("1,4,16") == [1, 4, 16]

    for invalid in ("", "0", "1,-2", "1,x"):
        try:
            sweep.positive_int_list(invalid)
        except Exception:
            pass
        else:
            raise AssertionError(f"expected invalid list to fail: {invalid}")


def test_benchmark_command() -> None:
    sweep = load_sweep()
    command = sweep.benchmark_command(Path("/tmp/benchmark"), 4, 64, 20, 2)
    assert command == [
        "/tmp/benchmark",
        "--chunks",
        "20",
        "--warmup-chunks",
        "2",
        "--chunk-size-mib",
        "4",
        "--compute-rounds",
        "64",
        "--csv",
    ]


def test_run_case() -> None:
    sweep = load_sweep()
    header = ",".join(sweep.CSV_FIELDS)
    values = "1,16,8,10.0,8.0,1.25,0.1,0.2,0.7,0.5,0.9"
    command = [
        sys.executable,
        "-c",
        f"print({header!r}); print({values!r})",
    ]
    row = sweep.run_case(command)
    assert row["chunk_size_mib"] == "1"
    assert row["compute_rounds"] == "16"
    assert row["speedup"] == "1.25"


def main() -> int:
    test_positive_int_list()
    test_benchmark_command()
    test_run_case()
    print("double_buffer_pipeline_sweep_test: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
