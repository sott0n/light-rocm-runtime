#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_qwen_e2e.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_qwen_e2e", RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_layers_reads_checkpoint_config() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir)
        (checkpoint / "config.json").write_text(
            json.dumps({"num_hidden_layers": 24}), encoding="utf-8"
        )
        args = runner.parse_args(
            [
                "--checkpoint-dir",
                str(checkpoint),
                "--bundle-dir",
                str(checkpoint / "bundle"),
            ]
        )
        assert runner.resolve_layers(args) == 24


def test_runner_command_requires_keys_for_tokens() -> None:
    runner = load_runner()
    args = runner.parse_args(
        [
            "--bundle-dir",
            "/tmp/bundle",
            "--layers",
            "1",
            "--keys",
            "2",
            "--token-ids",
            "0,1,2",
        ]
    )
    try:
        runner.runner_command(args, 1)
    except ValueError as error:
        assert "--keys" in str(error)
    else:
        raise AssertionError("expected --keys validation failure")


def test_dry_run_builds_full_e2e_command() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = Path(tmpdir) / "bundle"
        status = runner.main(
            [
                "--bundle-dir",
                str(bundle),
                "--layers",
                "2",
                "--keys",
                "4",
                "--token-ids",
                "3,4",
                "--benchmark",
                "/tmp/lrrt_fake_benchmark",
                "--no-uv",
                "--dry-run",
            ]
        )
        assert status == 0
        command = runner.runner_command(
            runner.parse_args(
                [
                    "--bundle-dir",
                    str(bundle),
                    "--layers",
                    "2",
                    "--keys",
                    "4",
                    "--token-ids",
                    "3,4",
                    "--benchmark",
                    "/tmp/lrrt_fake_benchmark",
                    "--no-uv",
                ]
            ),
            2,
        )
        assert "--e2e-check" in command
        assert "--sync-stack" in command


def main() -> int:
    test_resolve_layers_reads_checkpoint_config()
    test_runner_command_requires_keys_for_tokens()
    test_dry_run_builds_full_e2e_command()
    print("qwen_e2e_runner_test: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
