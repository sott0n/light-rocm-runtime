#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "tools" / "convert_qwen_layer.py"


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_qwen_layer", CONVERTER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_derive_shape_rejects_gqa() -> None:
    converter = load_converter()
    config = {
        "hidden_size": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 16,
    }
    try:
        converter.derive_shape(config, keys=4)
    except ValueError as error:
        assert "grouped-query attention is not supported" in str(error)
    else:
        raise AssertionError("expected GQA rejection")


def test_tensor_mapping_and_bundle_writer() -> None:
    converter = load_converter()
    config = {
        "hidden_size": 8,
        "num_attention_heads": 2,
        "intermediate_size": 12,
    }
    shape = converter.derive_shape(config, keys=4)
    mappings = converter.tensor_mappings(layer=3, shape=shape)
    assert mappings[0].checkpoint_name == "model.layers.3.input_layernorm.weight"
    assert mappings[2].shape == (8, 8)

    tensors = {}
    for mapping in converter.tensor_mappings(layer=0, shape=shape):
        count = 1
        for dim in mapping.shape:
            count *= dim
        tensors[mapping.bundle_name] = [float(index) for index in range(count)]

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "weights.json"
        converter.write_bundle(manifest_path, "weights.bin", shape, tensors)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data = (Path(tmpdir) / "weights.bin").read_bytes()

    assert manifest["format"] == "lrrt.mini_decoder_weights"
    assert manifest["dtype"] == "f32"
    assert manifest["hidden"] == 8
    assert manifest["heads"] == 2
    assert manifest["head_dim"] == 4
    assert manifest["tensors"][0] == {
        "name": "attention_norm_weight",
        "offset": 0,
        "count": 8,
    }
    assert len(data) == sum(tensor["count"] for tensor in manifest["tensors"]) * 4
    assert struct.unpack_from("<f", data, 0)[0] == 0.0


def main() -> int:
    test_derive_shape_rejects_gqa()
    test_tensor_mapping_and_bundle_writer()
    print("qwen_layer_converter_test: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
