#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "select_iree_qwen_linear_variant.py"


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "select_iree_qwen_linear_variant", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_selects_requested_sections() -> None:
    tool = load_tool()
    source = """before
// QWEN_MATMUL_BEGIN
matmul
// QWEN_MATMUL_END
middle
// QWEN_REDUCED_BEGIN
vecmat
// QWEN_REDUCED_END
// QWEN_MIXED_BEGIN
mixed
// QWEN_MIXED_END
after
"""
    assert tool.select_variant(source, "matmul") == "before\nmatmul\nmiddle\nafter\n"
    assert tool.select_variant(source, "reduced") == "before\nmiddle\nvecmat\nafter\n"
    assert tool.select_variant(source, "mixed") == "before\nmiddle\nmixed\nafter\n"


def test_rejects_malformed_sections() -> None:
    tool = load_tool()
    malformed = """// QWEN_MATMUL_BEGIN
matmul
// QWEN_REDUCED_END
"""
    try:
        tool.select_variant(malformed, "matmul")
    except ValueError as error:
        assert "mismatched" in str(error)
    else:
        raise AssertionError("malformed section was accepted")


if __name__ == "__main__":
    test_selects_requested_sections()
    test_rejects_malformed_sections()
