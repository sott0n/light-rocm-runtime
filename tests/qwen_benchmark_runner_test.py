#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_qwen_benchmark.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_qwen_benchmark", RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bundle_complete_requires_layers_and_tail() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = Path(tmpdir)
        (bundle / "layer_0").mkdir()
        (bundle / "layer_0" / "weights.json").write_text("{}", encoding="utf-8")

        assert runner.infer_bundle_layer_count(bundle) == 1
        assert runner.bundle_complete(bundle, 1, require_tail=False)
        assert not runner.bundle_complete(bundle, 1, require_tail=True)

        (bundle / "model_tail").mkdir()
        (bundle / "model_tail" / "weights.json").write_text("{}", encoding="utf-8")
        assert runner.bundle_complete(bundle, 1, require_tail=True)
        assert not runner.bundle_complete(bundle, 2, require_tail=True)


def test_all_layers_reads_config() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir)
        (checkpoint / "config.json").write_text(
            json.dumps({"num_hidden_layers": 24}), encoding="utf-8"
        )
        assert runner.model_layer_count(checkpoint, config=None) == 24

        args = runner.parse_args(
            [
                "--checkpoint-dir",
                str(checkpoint),
                "--bundle-dir",
                str(checkpoint / "bundle"),
                "--all-layers",
            ]
        )
        assert runner.resolve_layers(args) == 24


def test_runner_rejects_nonzero_start_layer() -> None:
    runner = load_runner()
    args = runner.parse_args(
        ["--bundle-dir", "/tmp/bundle", "--layers", "1", "--layer", "1"]
    )
    try:
        runner.resolve_layers(args)
    except ValueError as error:
        assert "layer_0" in str(error)
    else:
        raise AssertionError("expected nonzero --layer to fail")


def test_dry_run_reuses_existing_bundle() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = Path(tmpdir) / "bundle"
        (bundle / "layer_0").mkdir(parents=True)
        (bundle / "layer_0" / "weights.json").write_text("{}", encoding="utf-8")
        (bundle / "model_tail").mkdir()
        (bundle / "model_tail" / "weights.json").write_text("{}", encoding="utf-8")

        status = runner.main(
            [
                "--bundle-dir",
                str(bundle),
                "--layers",
                "1",
                "--benchmark",
                "/tmp/lrrt_fake_benchmark",
                "--dry-run",
                "--no-warmup",
            ]
        )
        assert status == 0


def main() -> int:
    test_bundle_complete_requires_layers_and_tail()
    test_all_layers_reads_config()
    test_runner_rejects_nonzero_start_layer()
    test_dry_run_reuses_existing_bundle()
    print("qwen_benchmark_runner_test: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
