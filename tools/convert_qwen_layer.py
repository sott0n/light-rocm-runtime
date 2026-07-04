#!/usr/bin/env python3
"""Convert Qwen decoder layers into lrrt mini decoder weight bundles."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMAT = "lrrt.mini_decoder_weights"
TAIL_FORMAT = "lrrt.mini_model_tail_weights"
VERSION = 1


@dataclass(frozen=True)
class DecoderLayerShape:
    keys: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int

    @property
    def q_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim


@dataclass(frozen=True)
class TensorMapping:
    bundle_name: str
    checkpoint_name: str
    shape: tuple[int, ...]


def _positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"config field {key!r} must be a positive integer")
    return value


def derive_shape(config: dict[str, Any], keys: int) -> DecoderLayerShape:
    if keys <= 0:
        raise ValueError("--keys must be a positive integer")

    hidden = _positive_int(config, "hidden_size")
    heads = _positive_int(config, "num_attention_heads")
    intermediate = _positive_int(config, "intermediate_size")
    kv_heads = config.get("num_key_value_heads", heads)
    if not isinstance(kv_heads, int) or kv_heads <= 0:
        raise ValueError("config field 'num_key_value_heads' must be positive")
    if kv_heads > heads or heads % kv_heads != 0:
        raise ValueError(
            "the mini decoder converter expects num_attention_heads to be a "
            f"multiple of num_key_value_heads, got num_key_value_heads={kv_heads}, "
            f"num_attention_heads={heads}"
        )

    head_dim = config.get("head_dim")
    if head_dim is None:
        if hidden % heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        head_dim = hidden // heads
    if not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError("config field 'head_dim' must be a positive integer")
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for the current RoPE kernels")
    if heads * head_dim != hidden:
        raise ValueError(
            "the current mini decoder expects heads * head_dim == hidden, "
            f"got {heads} * {head_dim} != {hidden}"
        )

    return DecoderLayerShape(
        keys=keys,
        hidden=hidden,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        intermediate=intermediate,
    )


def derive_rope_theta(config: dict[str, Any]) -> float:
    value = config.get("rope_theta", 10000.0)
    if not isinstance(value, int | float) or value <= 0:
        raise ValueError("config field 'rope_theta' must be a positive number")
    return float(value)


def tensor_mappings(layer: int, shape: DecoderLayerShape) -> list[TensorMapping]:
    if layer < 0:
        raise ValueError("--layer must be non-negative")
    prefix = f"model.layers.{layer}"
    q_dim = shape.q_dim
    kv_dim = shape.kv_dim
    return [
        TensorMapping(
            "attention_norm_weight",
            f"{prefix}.input_layernorm.weight",
            (shape.hidden,),
        ),
        TensorMapping(
            "mlp_norm_weight",
            f"{prefix}.post_attention_layernorm.weight",
            (shape.hidden,),
        ),
        TensorMapping(
            "q_weight",
            f"{prefix}.self_attn.q_proj.weight",
            (q_dim, shape.hidden),
        ),
        TensorMapping(
            "k_weight",
            f"{prefix}.self_attn.k_proj.weight",
            (kv_dim, shape.hidden),
        ),
        TensorMapping(
            "v_weight",
            f"{prefix}.self_attn.v_proj.weight",
            (kv_dim, shape.hidden),
        ),
        TensorMapping(
            "out_weight",
            f"{prefix}.self_attn.o_proj.weight",
            (shape.hidden, q_dim),
        ),
        TensorMapping(
            "gate_weight",
            f"{prefix}.mlp.gate_proj.weight",
            (shape.intermediate, shape.hidden),
        ),
        TensorMapping(
            "up_weight",
            f"{prefix}.mlp.up_proj.weight",
            (shape.intermediate, shape.hidden),
        ),
        TensorMapping(
            "down_weight",
            f"{prefix}.mlp.down_proj.weight",
            (shape.hidden, shape.intermediate),
        ),
    ]


def tail_tensor_names() -> tuple[str, str, str]:
    return ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "numpy is required for Qwen checkpoint conversion. Install with "
            "`uv pip install -r tools/requirements.txt`."
        ) from error
    return np


def checkpoint_weight_map(checkpoint_dir: Path) -> dict[str, str] | None:
    index_paths = sorted(checkpoint_dir.glob("*.safetensors.index.json"))
    if not index_paths:
        return None
    index = read_json(index_paths[0])
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_paths[0]} is missing object field 'weight_map'")
    result: dict[str, str] = {}
    for name, file_name in weight_map.items():
        if not isinstance(name, str) or not isinstance(file_name, str):
            raise ValueError(f"{index_paths[0]} contains invalid weight_map entries")
        result[name] = file_name
    return result


def _safetensor_files(checkpoint_dir: Path) -> list[Path]:
    files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files found in {checkpoint_dir}")
    return files


def _convert_safetensor_payload(
    payload: memoryview, dtype: str, shape: tuple[int, ...], name: str
) -> Any:
    np = _import_numpy()
    count = 1
    for dim in shape:
        count *= dim
    if dtype == "F32":
        array = np.frombuffer(payload, dtype="<f4", count=count)
        return np.asarray(array, dtype=np.float32).reshape(shape)
    if dtype == "F16":
        array = np.frombuffer(payload, dtype="<f2", count=count)
        return np.asarray(array, dtype=np.float32).reshape(shape)
    if dtype == "BF16":
        raw = np.frombuffer(payload, dtype="<u2", count=count).astype(np.uint32)
        array = (raw << 16).view(np.float32)
        return np.asarray(array, dtype=np.float32).reshape(shape)
    raise ValueError(f"unsupported safetensors dtype for {name!r}: {dtype}")


def read_safetensors(path: Path, wanted: set[str]) -> dict[str, Any]:
    with path.open("rb") as file:
        header_size_data = file.read(8)
        if len(header_size_data) != 8:
            raise ValueError(f"{path} is not a valid safetensors file")
        header_size = struct.unpack("<Q", header_size_data)[0]
        header_data = file.read(header_size)
        if len(header_data) != header_size:
            raise ValueError(f"{path} has a truncated safetensors header")
        header = json.loads(header_data.decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError(f"{path} safetensors header must be an object")
        data = memoryview(file.read())

    tensors: dict[str, Any] = {}
    for name in wanted:
        entry = header.get(name)
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"{path} tensor {name!r} metadata is invalid")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise ValueError(f"{path} tensor {name!r} metadata is incomplete")
        tensor_shape = tuple(int(dim) for dim in shape)
        begin = int(offsets[0])
        end = int(offsets[1])
        if begin < 0 or end < begin or end > len(data):
            raise ValueError(f"{path} tensor {name!r} data offset is invalid")
        tensors[name] = _convert_safetensor_payload(
            data[begin:end], dtype, tensor_shape, name
        )
    return tensors


def load_tensors(checkpoint_dir: Path, mappings: list[TensorMapping]) -> dict[str, Any]:
    wanted = {mapping.checkpoint_name for mapping in mappings}
    weight_map = checkpoint_weight_map(checkpoint_dir)

    tensors: dict[str, Any] = {}
    if weight_map is not None:
        missing = sorted(wanted.difference(weight_map))
        if missing:
            raise KeyError("checkpoint index is missing tensors: " + ", ".join(missing))
        shard_names = sorted({weight_map[name] for name in wanted})
        for shard_name in shard_names:
            shard = read_safetensors(checkpoint_dir / shard_name, wanted)
            for name in wanted.intersection(shard):
                tensors[name] = shard[name]
    else:
        for shard_path in _safetensor_files(checkpoint_dir):
            shard = read_safetensors(shard_path, wanted)
            for name in wanted.intersection(shard):
                tensors[name] = shard[name]
            if len(tensors) == len(wanted):
                break

    missing = sorted(wanted.difference(tensors))
    if missing:
        raise KeyError("checkpoint is missing tensors: " + ", ".join(missing))
    return tensors


def validate_no_attention_biases(
    checkpoint_dir: Path, layer: int, available_names: set[str] | None = None
) -> None:
    prefix = f"model.layers.{layer}.self_attn"
    bias_names = {
        f"{prefix}.q_proj.bias",
        f"{prefix}.k_proj.bias",
        f"{prefix}.v_proj.bias",
        f"{prefix}.o_proj.bias",
    }
    if available_names is None:
        weight_map = checkpoint_weight_map(checkpoint_dir)
        available_names = set(weight_map) if weight_map is not None else set()
    present = sorted(bias_names.intersection(available_names))
    if present:
        raise ValueError(
            "attention projection biases are not supported by the mini decoder "
            "converter: " + ", ".join(present)
        )


def convert_tensor(value: Any, expected_shape: tuple[int, ...], name: str) -> Any:
    np = _import_numpy()
    array = np.asarray(value, dtype=np.float32, order="C")
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            f"tensor {name!r} has shape {tuple(array.shape)}, expected {expected_shape}"
        )
    return np.ascontiguousarray(array)


def _validate_data_file_name(data_file_name: str) -> None:
    data_path = Path(data_file_name)
    if (
        not data_file_name
        or data_path.is_absolute()
        or ".." in data_path.parts
        or len(data_path.parts) != 1
    ):
        raise ValueError("--data-file must be a single relative file name")


def _tensor_bytes(array: Any) -> bytes:
    if hasattr(array, "tobytes"):
        return array.tobytes(order="C")
    return struct.pack(f"<{len(array)}f", *array)


def write_bundle(
    output_manifest: Path,
    data_file_name: str,
    shape: DecoderLayerShape,
    tensors: dict[str, Any],
    rope_theta: float = 10000.0,
) -> None:
    _validate_data_file_name(data_file_name)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    data_path = output_manifest.parent / data_file_name

    manifest_tensors: list[dict[str, int | str]] = []
    offset = 0
    with data_path.open("wb") as data_file:
        for mapping in tensor_mappings(0, shape):
            name = mapping.bundle_name
            if name not in tensors:
                raise KeyError(f"missing converted tensor: {name}")
            payload = _tensor_bytes(tensors[name])
            if len(payload) % 4 != 0:
                raise ValueError(f"tensor {name!r} is not packed as FP32 bytes")
            count = len(payload) // 4
            expected_count = 1
            for dim in mapping.shape:
                expected_count *= dim
            if count != expected_count:
                raise ValueError(
                    f"tensor {name!r} has {count} values, expected {expected_count}"
                )
            manifest_tensors.append({"name": name, "offset": offset, "count": count})
            data_file.write(payload)
            offset += len(payload)

    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "dtype": "f32",
        "data": data_file_name,
        "keys": shape.keys,
        "hidden": shape.hidden,
        "heads": shape.heads,
        "kv_heads": shape.kv_heads,
        "head_dim": shape.head_dim,
        "intermediate": shape.intermediate,
        "rope_theta": rope_theta,
        "tensors": manifest_tensors,
    }
    with output_manifest.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")


def _write_tail_bundle(
    output_manifest: Path,
    data_file_name: str,
    hidden: int,
    token_id: int,
    token_embedding: Any,
    final_norm_weight: Any,
    lm_head_weight: Any,
) -> None:
    _validate_data_file_name(data_file_name)
    np = _import_numpy()
    token_embedding = np.ascontiguousarray(token_embedding, dtype=np.float32)
    final_norm_weight = np.ascontiguousarray(final_norm_weight, dtype=np.float32)
    lm_head_weight = np.ascontiguousarray(lm_head_weight, dtype=np.float32)
    if token_embedding.shape != (hidden,):
        raise ValueError(
            f"token embedding has shape {token_embedding.shape}, expected {(hidden,)}"
        )
    if final_norm_weight.shape != (hidden,):
        raise ValueError(
            "model.norm.weight has shape "
            f"{final_norm_weight.shape}, expected {(hidden,)}"
        )
    if len(lm_head_weight.shape) != 2 or lm_head_weight.shape[1] != hidden:
        raise ValueError(
            "lm_head.weight must have shape [vocab, hidden], got "
            f"{lm_head_weight.shape}"
        )
    vocab = int(lm_head_weight.shape[0])
    if token_id < 0 or token_id >= vocab:
        raise ValueError(f"--token-id must be in [0, {vocab}), got {token_id}")

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    data_path = output_manifest.parent / data_file_name
    manifest_tensors: list[dict[str, int | str]] = []
    offset = 0
    tensors = [
        ("token_embedding", token_embedding),
        ("final_norm_weight", final_norm_weight),
        ("lm_head_weight", lm_head_weight),
    ]
    with data_path.open("wb") as data_file:
        for name, tensor in tensors:
            payload = tensor.tobytes(order="C")
            manifest_tensors.append(
                {"name": name, "offset": offset, "count": len(payload) // 4}
            )
            data_file.write(payload)
            offset += len(payload)

    manifest = {
        "format": TAIL_FORMAT,
        "version": VERSION,
        "dtype": "f32",
        "data": data_file_name,
        "hidden": hidden,
        "vocab": vocab,
        "token_id": token_id,
        "tensors": manifest_tensors,
    }
    with output_manifest.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")


def _convert_tail(
    checkpoint_dir: Path,
    output: Path,
    data_file: str,
    hidden: int,
    token_id: int,
) -> None:
    embed_name, norm_name, lm_head_name = tail_tensor_names()
    try:
        checkpoint_tensors = load_tensors(
            checkpoint_dir,
            [
                TensorMapping("embed_tokens", embed_name, (1,)),
                TensorMapping("final_norm_weight", norm_name, (hidden,)),
                TensorMapping("lm_head_weight", lm_head_name, (1, hidden)),
            ],
        )
    except KeyError:
        checkpoint_tensors = load_tensors(
            checkpoint_dir,
            [
                TensorMapping("embed_tokens", embed_name, (1,)),
                TensorMapping("final_norm_weight", norm_name, (hidden,)),
            ],
        )
        checkpoint_tensors[lm_head_name] = checkpoint_tensors[embed_name]

    embed_tokens = checkpoint_tensors[embed_name]
    final_norm_weight = checkpoint_tensors[norm_name]
    lm_head_weight = checkpoint_tensors[lm_head_name]
    if len(embed_tokens.shape) != 2 or embed_tokens.shape[1] != hidden:
        raise ValueError(
            "model.embed_tokens.weight must have shape [vocab, hidden], got "
            f"{embed_tokens.shape}"
        )
    if token_id < 0 or token_id >= embed_tokens.shape[0]:
        raise ValueError(
            f"--token-id must be in [0, {embed_tokens.shape[0]}), got {token_id}"
        )
    _write_tail_bundle(
        output,
        data_file,
        hidden,
        token_id,
        embed_tokens[token_id],
        final_norm_weight,
        lm_head_weight,
    )


def _convert_layer(
    checkpoint_dir: Path,
    shape: DecoderLayerShape,
    rope_theta: float,
    layer: int,
    output: Path,
    data_file: str,
    available_names: set[str] | None,
) -> None:
    mappings = tensor_mappings(layer, shape)
    validate_no_attention_biases(checkpoint_dir, layer, available_names)
    checkpoint_tensors = load_tensors(checkpoint_dir, mappings)
    converted = {
        mapping.bundle_name: convert_tensor(
            checkpoint_tensors[mapping.checkpoint_name],
            mapping.shape,
            mapping.checkpoint_name,
        )
        for mapping in mappings
    }
    write_bundle(output, data_file, shape, converted, rope_theta)


def convert_checkpoint(args: argparse.Namespace) -> None:
    checkpoint_dir = args.checkpoint_dir.resolve()
    config_path = (
        args.config.resolve() if args.config else checkpoint_dir / "config.json"
    )
    config = read_json(config_path)
    shape = derive_shape(config, args.keys)
    rope_theta = derive_rope_theta(config)
    weight_map = checkpoint_weight_map(checkpoint_dir)
    available_names = set(weight_map) if weight_map is not None else None
    layer_count = args.layer_count
    if layer_count <= 0:
        raise ValueError("--layer-count must be a positive integer")

    if layer_count == 1:
        _convert_layer(
            checkpoint_dir,
            shape,
            rope_theta,
            args.layer,
            args.output,
            args.data_file,
            available_names,
        )
        print(f"wrote Qwen mini decoder weight bundle: {args.output}")
    else:
        if args.output.suffix:
            raise ValueError(
                "--output must be a directory when --layer-count is greater than 1"
            )
        for layer in range(args.layer, args.layer + layer_count):
            layer_manifest = args.output / f"layer_{layer}" / "weights.json"
            _convert_layer(
                checkpoint_dir,
                shape,
                rope_theta,
                layer,
                layer_manifest,
                args.data_file,
                available_names,
            )
            print(f"wrote Qwen mini decoder layer {layer}: {layer_manifest}")
        tail_manifest = args.output / "model_tail" / "weights.json"
        _convert_tail(
            checkpoint_dir,
            tail_manifest,
            args.data_file,
            shape.hidden,
            args.token_id,
        )
        print(f"wrote Qwen mini model tail: {tail_manifest}")

    print(
        "shape: "
        f"keys={shape.keys} hidden={shape.hidden} heads={shape.heads} "
        f"kv_heads={shape.kv_heads} head_dim={shape.head_dim} "
        f"intermediate={shape.intermediate} rope_theta={rope_theta:g}"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert local Hugging Face Qwen decoder layers into lrrt mini "
            "decoder weight bundles."
        )
    )
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="defaults to config.json")
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument(
        "--layer-count",
        default=1,
        type=int,
        help=(
            "number of consecutive decoder layers to convert. When greater "
            "than 1, --output is treated as a directory and bundles are "
            "written under layer_<index>/weights.json."
        ),
    )
    parser.add_argument("--keys", required=True, type=int)
    parser.add_argument(
        "--token-id",
        default=0,
        type=int,
        help=(
            "token id whose embedding row is stored in the model tail bundle "
            "when --layer-count is greater than 1."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-file", default="weights.bin")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    try:
        convert_checkpoint(parse_args(argv))
        return 0
    except Exception as error:
        print(f"convert_qwen_layer.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
