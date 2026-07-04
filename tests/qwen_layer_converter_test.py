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


def test_derive_shape_accepts_gqa() -> None:
    converter = load_converter()
    config = {
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 24,
    }
    shape = converter.derive_shape(config, keys=4)
    assert converter.derive_rope_theta({"rope_theta": 1000000.0}) == 1000000.0
    assert shape.heads == 4
    assert shape.kv_heads == 2
    assert shape.head_dim == 4
    mappings = converter.tensor_mappings(layer=0, shape=shape)
    assert mappings[2].shape == (16, 16)
    assert mappings[3].shape == (8, 16)
    assert mappings[4].shape == (8, 16)


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
        converter.write_bundle(
            manifest_path, "weights.bin", shape, tensors, rope_theta=1000000.0
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data = (Path(tmpdir) / "weights.bin").read_bytes()

    assert manifest["format"] == "lrrt.mini_decoder_weights"
    assert manifest["dtype"] == "f32"
    assert manifest["hidden"] == 8
    assert manifest["heads"] == 2
    assert manifest["kv_heads"] == 2
    assert manifest["head_dim"] == 4
    assert manifest["rope_theta"] == 1000000.0
    assert manifest["tensors"][0] == {
        "name": "attention_norm_weight",
        "offset": 0,
        "count": 8,
    }
    assert len(data) == sum(tensor["count"] for tensor in manifest["tensors"]) * 4
    assert struct.unpack_from("<f", data, 0)[0] == 0.0


def test_multi_layer_output_paths() -> None:
    converter = load_converter()
    config = {
        "hidden_size": 8,
        "num_attention_heads": 2,
        "intermediate_size": 12,
    }
    shape = converter.derive_shape(config, keys=4)
    tensors = {}
    for mapping in converter.tensor_mappings(layer=0, shape=shape):
        count = 1
        for dim in mapping.shape:
            count *= dim
        tensors[mapping.bundle_name] = [float(index) for index in range(count)]

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "layers"
        for layer in range(2):
            converter.write_bundle(
                output / f"layer_{layer}" / "weights.json",
                "weights.bin",
                shape,
                tensors,
            )

        assert (output / "layer_0" / "weights.json").exists()
        assert (output / "layer_0" / "weights.bin").exists()
        assert (output / "layer_1" / "weights.json").exists()
        assert (output / "layer_1" / "weights.bin").exists()


def test_tail_bundle_writer() -> None:
    converter = load_converter()
    np = converter._import_numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "model_tail" / "weights.json"
        converter._write_tail_bundle(
            manifest_path,
            "weights.bin",
            hidden=4,
            token_id=2,
            token_embedding=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            final_norm_weight=np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
            lm_head_weight=np.arange(12, dtype=np.float32).reshape(3, 4),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data = (manifest_path.parent / "weights.bin").read_bytes()

    assert manifest["format"] == "lrrt.mini_model_tail_weights"
    assert manifest["hidden"] == 4
    assert manifest["vocab"] == 3
    assert manifest["token_id"] == 2
    assert manifest["tensors"][0] == {
        "name": "token_embedding",
        "offset": 0,
        "count": 4,
    }
    assert len(data) == (4 + 4 + 12) * 4
    assert struct.unpack_from("<f", data, 0)[0] == 1.0


def test_read_safetensors_bf16() -> None:
    converter = load_converter()
    header = {
        "tensor": {
            "dtype": "BF16",
            "shape": [2],
            "data_offsets": [0, 4],
        }
    }
    header_data = json.dumps(header).encode("utf-8")
    values = struct.pack("<HH", 0x3F80, 0x4000)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.safetensors"
        path.write_bytes(struct.pack("<Q", len(header_data)) + header_data + values)
        tensors = converter.read_safetensors(path, {"tensor"})

    assert list(tensors["tensor"]) == [1.0, 2.0]


def main() -> int:
    test_derive_shape_accepts_gqa()
    test_tensor_mapping_and_bundle_writer()
    test_multi_layer_output_paths()
    test_read_safetensors_bf16()
    print("qwen_layer_converter_test: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
