#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "write_iree_qwen_decode_bundle.py"


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "write_iree_qwen_decode_bundle", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_vmfb(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def test_writes_bundle_manifest_and_copies_vmfb() -> None:
    tool = load_tool()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        layer = root / "layer.vmfb"
        tail = root / "tail.vmfb"
        out = root / "bundle"
        write_vmfb(layer, b"layer")
        write_vmfb(tail, b"tail")

        status = tool.main(
            [
                "--target",
                "gfx1101",
                "--layer-vmfb",
                str(layer),
                "--tail-vmfb",
                str(tail),
                "--out-dir",
                str(out),
            ]
        )

        assert status == 0
        assert (out / "layer.vmfb").read_bytes() == b"layer"
        assert (out / "tail.vmfb").read_bytes() == b"tail"
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest == {
            "manifest_version": 1,
            "target": "gfx1101",
            "layer_vmfb": "layer.vmfb",
            "tail_vmfb": "tail.vmfb",
            "layer_export": "qwen_decode_layer_kv_cache_max8",
            "tail_export": "qwen_decode1_tail",
            "sequence_capacity": 8,
            "max_cache_tokens": 8,
            "kv_cache_shape": [8, 128],
        }


def test_default_layer_export_follows_cache_capacity() -> None:
    tool = load_tool()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        layer = root / "layer.vmfb"
        tail = root / "tail.vmfb"
        out = root / "bundle"
        write_vmfb(layer, b"layer")
        write_vmfb(tail, b"tail")

        status = tool.main(
            [
                "--target",
                "gfx1101",
                "--layer-vmfb",
                str(layer),
                "--tail-vmfb",
                str(tail),
                "--out-dir",
                str(out),
                "--sequence-capacity",
                "10",
                "--max-cache-tokens",
                "16",
            ]
        )

        assert status == 0
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["layer_export"] == "qwen_decode_layer_kv_cache_max16"
        assert manifest["sequence_capacity"] == 10
        assert manifest["max_cache_tokens"] == 16
        assert manifest["kv_cache_shape"] == [16, 128]


def test_rejects_sequence_capacity_over_cache_capacity() -> None:
    tool = load_tool()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        layer = root / "layer.vmfb"
        tail = root / "tail.vmfb"
        write_vmfb(layer, b"layer")
        write_vmfb(tail, b"tail")

        status = tool.main(
            [
                "--target",
                "gfx1101",
                "--layer-vmfb",
                str(layer),
                "--tail-vmfb",
                str(tail),
                "--out-dir",
                str(root / "bundle"),
                "--sequence-capacity",
                "17",
                "--max-cache-tokens",
                "16",
            ]
        )

        assert status == 1


def test_rejects_parent_bundle_path() -> None:
    tool = load_tool()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        layer = root / "layer.vmfb"
        tail = root / "tail.vmfb"
        write_vmfb(layer, b"layer")
        write_vmfb(tail, b"tail")

        status = tool.main(
            [
                "--target",
                "gfx1101",
                "--layer-vmfb",
                str(layer),
                "--tail-vmfb",
                str(tail),
                "--out-dir",
                str(root / "bundle"),
                "--layer-name",
                "../layer.vmfb",
            ]
        )

        assert status == 1


def test_refuses_overwrite_without_force() -> None:
    tool = load_tool()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        layer = root / "layer.vmfb"
        tail = root / "tail.vmfb"
        out = root / "bundle"
        write_vmfb(layer, b"layer")
        write_vmfb(tail, b"tail")
        assert (
            tool.main(
                [
                    "--target",
                    "gfx1101",
                    "--layer-vmfb",
                    str(layer),
                    "--tail-vmfb",
                    str(tail),
                    "--out-dir",
                    str(out),
                ]
            )
            == 0
        )

        assert (
            tool.main(
                [
                    "--target",
                    "gfx1101",
                    "--layer-vmfb",
                    str(layer),
                    "--tail-vmfb",
                    str(tail),
                    "--out-dir",
                    str(out),
                ]
            )
            == 1
        )


def main() -> int:
    test_writes_bundle_manifest_and_copies_vmfb()
    test_default_layer_export_follows_cache_capacity()
    test_rejects_sequence_capacity_over_cache_capacity()
    test_rejects_parent_bundle_path()
    test_refuses_overwrite_without_force()
    print("write_iree_qwen_decode_bundle_test: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
