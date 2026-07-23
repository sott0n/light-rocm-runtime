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


def write_complete_weight_bundle(bundle: Path, layers: int) -> None:
    for layer in range(layers):
        layer_dir = bundle / f"layer_{layer}"
        layer_dir.mkdir(parents=True)
        (layer_dir / "weights.json").write_text("{}", encoding="utf-8")
    tail_dir = bundle / "model_tail"
    tail_dir.mkdir()
    (tail_dir / "weights.json").write_text("{}", encoding="utf-8")


def test_iree_runner_command_uses_decode_bundle() -> None:
    runner = load_runner()
    args = runner.parse_args(
        [
            "--iree",
            "--bundle-dir",
            "/tmp/qwen-weights",
            "--layers",
            "2",
            "--steps",
            "3",
            "--iree-decode-bundle-dir",
            "/tmp/iree-decode-bundle",
            "--iree-runner",
            "/tmp/lrrt_iree_qwen_decode1_e2e",
        ]
    )
    command = runner.iree_runner_command(args, 2)
    assert command == [
        "/tmp/lrrt_iree_qwen_decode1_e2e",
        "--steps",
        "3",
        "--bundle",
        "/tmp/iree-decode-bundle",
        "/tmp/qwen-weights",
        "2",
    ]


def test_iree_dry_run_writes_bundle_then_runs() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        weights = root / "weights"
        write_complete_weight_bundle(weights, 1)

        status = runner.main(
            [
                "--iree",
                "--bundle-dir",
                str(weights),
                "--layers",
                "1",
                "--steps",
                "2",
                "--iree-layer-vmfb",
                str(root / "layer.vmfb"),
                "--iree-tail-vmfb",
                str(root / "tail.vmfb"),
                "--iree-decode-bundle-dir",
                str(root / "iree-bundle"),
                "--iree-runner",
                "/tmp/lrrt_iree_qwen_decode1_e2e",
                "--dry-run",
            ]
        )
        assert status == 0


def test_iree_dry_run_discovers_default_vmfb_inputs() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        weights = root / "weights"
        write_complete_weight_bundle(weights, 1)
        probe = root / "probe"
        layer_vmfb = (
            probe
            / "qwen_decode_layer_kv_cache_max8"
            / "qwen_decode_layer_kv_cache_max8_gfx1101.vmfb"
        )
        tail_vmfb = probe / "qwen_decode1_tail" / "qwen_decode1_tail_gfx1101.vmfb"
        layer_vmfb.parent.mkdir(parents=True)
        tail_vmfb.parent.mkdir(parents=True)
        layer_vmfb.write_text("layer", encoding="utf-8")
        tail_vmfb.write_text("tail", encoding="utf-8")

        args = runner.parse_args(
            [
                "--iree",
                "--bundle-dir",
                str(weights),
                "--layers",
                "1",
                "--steps",
                "2",
                "--iree-probe-dir",
                str(probe),
                "--iree-decode-bundle-dir",
                str(root / "iree-bundle"),
                "--iree-runner",
                "/tmp/lrrt_iree_qwen_decode1_e2e",
                "--dry-run",
            ]
        )

        command = runner.iree_bundle_writer_command(args)
        assert str(layer_vmfb) in command
        assert str(tail_vmfb) in command


def test_iree_explicit_vmfb_inputs_override_discovery() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        probe = root / "probe"
        explicit_layer = root / "explicit-layer.vmfb"
        explicit_tail = root / "explicit-tail.vmfb"
        args = runner.parse_args(
            [
                "--iree",
                "--layers",
                "1",
                "--iree-layer-vmfb",
                str(explicit_layer),
                "--iree-tail-vmfb",
                str(explicit_tail),
                "--iree-probe-dir",
                str(probe),
            ]
        )

        command = runner.iree_bundle_writer_command(args)
        assert str(explicit_layer) in command
        assert str(explicit_tail) in command


def test_iree_requires_vmfb_inputs_when_bundle_and_default_vmfb_are_missing() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        weights = root / "weights"
        write_complete_weight_bundle(weights, 1)

        status = runner.main(
            [
                "--iree",
                "--bundle-dir",
                str(weights),
                "--layers",
                "1",
                "--iree-decode-bundle-dir",
                str(root / "missing-iree-bundle"),
                "--iree-probe-dir",
                str(root / "missing-probe"),
                "--dry-run",
            ]
        )
        assert status == 1


def main() -> int:
    test_resolve_layers_reads_checkpoint_config()
    test_runner_command_requires_keys_for_tokens()
    test_dry_run_builds_full_e2e_command()
    test_iree_runner_command_uses_decode_bundle()
    test_iree_dry_run_writes_bundle_then_runs()
    test_iree_dry_run_discovers_default_vmfb_inputs()
    test_iree_explicit_vmfb_inputs_override_discovery()
    test_iree_requires_vmfb_inputs_when_bundle_and_default_vmfb_are_missing()
    print("qwen_e2e_runner_test: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
