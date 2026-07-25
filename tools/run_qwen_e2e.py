#!/usr/bin/env python3
"""Convert or reuse Qwen bundles, then run a full lrrt E2E check."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = ROOT / "tools" / "run_qwen_benchmark.py"
DEFAULT_CONVERTER = ROOT / "tools" / "convert_qwen_layer.py"
DEFAULT_REQUIREMENTS = ROOT / "tools" / "requirements.txt"
DEFAULT_IREE_BUNDLE_WRITER = ROOT / "tools" / "write_iree_qwen_decode_bundle.py"
DEFAULT_IREE_RUNNER = ROOT / "build-iree" / "adapter" / "lrrt_iree_qwen_decode1_e2e"
DEFAULT_BUNDLE_DIR = Path("/tmp/lrrt-qwen-e2e")
DEFAULT_IREE_DECODE_BUNDLE_DIR = Path("/tmp/lrrt-iree-qwen-decode-bundle")
DEFAULT_IREE_PROBE_DIR = ROOT / "build-iree-probe"
DEFAULT_IREE_TARGET = "gfx1101"
IREE_LAYER_MODULE_RE = re.compile(
    r"^qwen_decode_layer_kv_cache_max([1-9][0-9]*)(?:_(f16|bf16))?$"
)
GENERATED_TOKEN_IDS_RE = re.compile(
    r"^generated_token_ids=\[([0-9]+(?:,[0-9]+)*)\]$", re.MULTILINE
)
TOP_LOGIT_RE = re.compile(
    r"^step ([1-9][0-9]*) top_token=([0-9]+) logit=([-+0-9.eE]+) vocab=([0-9]+)$",
    re.MULTILINE,
)
STOP_REASON_RE = re.compile(r"^stop_reason=(eos_token|max_new_tokens)$", re.MULTILINE)


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def parse_token_ids(text: str) -> list[int]:
    if not text:
        raise ValueError("--token-ids must not be empty")
    result: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            raise ValueError("--token-ids contains an empty entry")
        value = int(item, 10)
        if value < 0:
            raise ValueError("--token-ids must be non-negative")
        result.append(value)
    return result


def tokenizer_json_path(tokenizer_dir: Path) -> Path:
    path = tokenizer_dir / "tokenizer.json"
    if not path.is_file():
        raise ValueError(f"tokenizer file was not found: {path}")
    return path


def load_tokenizer(tokenizer_dir: Path) -> Any:
    path = tokenizer_json_path(tokenizer_dir)
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError(
            "tokenizers is required for text input or output; "
            "install tools/requirements.txt"
        ) from error
    return Tokenizer.from_file(str(path))


def render_chat_prompt(tokenizer_dir: Path, system: str | None, user: str) -> str:
    config_path = tokenizer_dir / "tokenizer_config.json"
    if not config_path.is_file():
        raise ValueError(f"tokenizer config was not found: {config_path}")
    chat_template = read_json(config_path).get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError(f"{config_path} field 'chat_template' must be a string")
    try:
        from jinja2 import StrictUndefined
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as error:
        raise RuntimeError(
            "jinja2 is required for --chat-user; install tools/requirements.txt"
        ) from error

    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    return environment.from_string(chat_template).render(
        messages=messages,
        tools=None,
        add_generation_prompt=True,
    )


def prepare_text_inputs(args: argparse.Namespace) -> Any | None:
    if args.reference_check and args.backend != "iree":
        raise ValueError("--reference-check requires --iree")
    if args.iree_precision != "f32" and args.backend != "iree":
        raise ValueError("--iree-precision requires --iree")
    if (
        args.expect_generated_token_ids is not None
        or args.expect_generated_text is not None
        or args.expect_stop_reason is not None
    ) and args.backend != "iree":
        raise ValueError(
            "--expect-generated-* requires --iree; "
            "--expect-stop-reason also requires --iree"
        )
    if args.backend != "iree" and (
        args.prompt is not None
        or args.chat_user is not None
        or args.tokenizer_dir is not None
    ):
        raise ValueError(
            "--prompt, --chat-user, and --tokenizer-dir currently require --iree"
        )
    if args.prompt is None and args.chat_user is None and args.tokenizer_dir is None:
        return None

    tokenizer_dir = args.tokenizer_dir
    if tokenizer_dir is None:
        if args.checkpoint_dir is None:
            raise ValueError("text input requires --tokenizer-dir or --checkpoint-dir")
        tokenizer_dir = args.checkpoint_dir
    tokenizer = load_tokenizer(tokenizer_dir)
    input_text = args.prompt
    if args.chat_user is not None:
        input_text = render_chat_prompt(tokenizer_dir, args.chat_system, args.chat_user)
        print(
            "chat_prompt="
            + json.dumps(input_text, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )
    if input_text is not None:
        prompt_token_ids = tokenizer.encode(input_text, add_special_tokens=False).ids
        if not prompt_token_ids:
            raise ValueError("text input encoded to an empty token sequence")
        args.token_ids = ",".join(str(value) for value in prompt_token_ids)
        print(f"prompt_token_ids=[{args.token_ids}]", flush=True)
    return tokenizer


def config_eos_token_id(path: Path) -> int | None:
    if not path.is_file():
        return None
    value = read_json(path).get("eos_token_id")
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} field 'eos_token_id' must be non-negative")
    return value


def tokenizer_config_eos_token_id(tokenizer_dir: Path, tokenizer: Any) -> int | None:
    path = tokenizer_dir / "tokenizer_config.json"
    if not path.is_file():
        return None
    value = read_json(path).get("eos_token")
    if isinstance(value, dict):
        value = value.get("content")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{path} field 'eos_token' must be a string")
    token_id = tokenizer.token_to_id(value)
    if token_id is None:
        raise ValueError(f"{path} eos token {value!r} is missing from tokenizer")
    return int(token_id)


def resolve_eos_token_id(args: argparse.Namespace, tokenizer: Any | None) -> int | None:
    if args.eos_token_id is not None:
        if args.eos_token_id < 0:
            raise ValueError("--eos-token-id must be non-negative")
        return args.eos_token_id

    search_dirs: list[Path] = []
    for directory in [args.checkpoint_dir, args.tokenizer_dir]:
        if directory is not None and directory not in search_dirs:
            search_dirs.append(directory)
    for directory in search_dirs:
        for name in ["generation_config.json", "config.json"]:
            token_id = config_eos_token_id(directory / name)
            if token_id is not None:
                return token_id
        if tokenizer is not None:
            token_id = tokenizer_config_eos_token_id(directory, tokenizer)
            if token_id is not None:
                return token_id
    return None


def parse_generated_token_ids(output: str) -> list[int]:
    matches = GENERATED_TOKEN_IDS_RE.findall(output)
    if len(matches) != 1:
        raise ValueError(
            "IREE runner output must contain exactly one "
            "generated_token_ids=[...] summary"
        )
    return [int(value, 10) for value in matches[0].split(",")]


def parse_iree_top_logit(output: str, step: int = 1) -> tuple[int, float]:
    matches = [
        (int(match_step), int(token_id), float(logit))
        for match_step, token_id, logit, _ in TOP_LOGIT_RE.findall(output)
        if int(match_step) == step
    ]
    if len(matches) != 1:
        raise ValueError(
            f"IREE runner output must contain exactly one step {step} logit"
        )
    return matches[0][1], matches[0][2]


def verify_reference_outputs(
    args: argparse.Namespace,
    runner_output: str,
    reference_steps: list[tuple[int, float]],
) -> None:
    generated_token_ids = parse_generated_token_ids(runner_output)
    if len(reference_steps) != len(generated_token_ids):
        raise ValueError(
            "IREE/reference generated step count mismatch: "
            f"IREE={len(generated_token_ids)}, reference={len(reference_steps)}"
        )

    max_logit_abs_diff = 0.0
    for step, (
        (reference_token_id, reference_top_logit),
        generated_token_id,
    ) in enumerate(zip(reference_steps, generated_token_ids), start=1):
        iree_token_id, iree_top_logit = parse_iree_top_logit(runner_output, step)
        if iree_token_id != generated_token_id:
            raise ValueError(
                f"IREE step {step} top token does not match generated summary: "
                f"top_token={iree_token_id}, generated={generated_token_id}"
            )
        logit_abs_diff = abs(iree_top_logit - reference_top_logit)
        max_logit_abs_diff = max(max_logit_abs_diff, logit_abs_diff)
        print(
            f"reference_step={step} top_token={reference_token_id} "
            f"logit={reference_top_logit:.9g} "
            f"logit_abs_diff={logit_abs_diff:.9g}",
            flush=True,
        )
        if iree_token_id != reference_token_id:
            raise ValueError(
                f"IREE/reference step {step} top token mismatch: "
                f"IREE={iree_token_id}, reference={reference_token_id}"
            )
        if logit_abs_diff > args.reference_logit_atol:
            raise ValueError(
                f"IREE/reference step {step} top logit mismatch: "
                f"abs_diff={logit_abs_diff:.9g}, "
                f"atol={args.reference_logit_atol:.9g}"
            )
    print(
        f"reference_check=passed steps={len(reference_steps)} "
        f"max_logit_abs_diff={max_logit_abs_diff:.9g}",
        flush=True,
    )


def bundle_tensor(manifest_path: Path, tensor_name: str, shape: tuple[int, ...]) -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("--reference-check requires numpy") from error

    manifest = read_json(manifest_path)
    data_name = manifest.get("data")
    tensors = manifest.get("tensors")
    if not isinstance(data_name, str) or not isinstance(tensors, list):
        raise ValueError(f"{manifest_path} has invalid tensor metadata")
    data_path = manifest_path.parent / data_name
    for tensor in tensors:
        if not isinstance(tensor, dict) or tensor.get("name") != tensor_name:
            continue
        offset = tensor.get("offset")
        count = tensor.get("count")
        if not isinstance(offset, int) or not isinstance(count, int):
            raise ValueError(f"{manifest_path} tensor {tensor_name!r} is invalid")
        expected_count = 1
        for dim in shape:
            expected_count *= dim
        if count != expected_count:
            raise ValueError(
                f"{manifest_path} tensor {tensor_name!r} has unexpected size"
            )
        return np.memmap(data_path, dtype="<f4", mode="r", offset=offset, shape=shape)
    raise ValueError(f"{manifest_path} is missing tensor {tensor_name!r}")


def verify_reference_checkpoint(args: argparse.Namespace, model: Any) -> None:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("--reference-check requires numpy") from error

    tail_manifest_path = args.bundle_dir / "model_tail" / "weights.json"
    tail_manifest = read_json(tail_manifest_path)
    hidden = tail_manifest.get("hidden")
    token_ids = tail_manifest.get("token_ids")
    if not isinstance(hidden, int) or not isinstance(token_ids, list):
        raise ValueError(f"{tail_manifest_path} has invalid model-tail metadata")
    token_index = {token_id: index for index, token_id in enumerate(token_ids)}
    prompt_ids = parse_token_ids(args.token_ids)
    if any(token_id not in token_index for token_id in prompt_ids):
        raise ValueError("prompt token embedding is missing from model-tail bundle")

    token_embeddings = bundle_tensor(
        tail_manifest_path, "token_embeddings", (len(token_ids), hidden)
    )
    bundle_prompt_embeddings = np.asarray(
        [token_embeddings[token_index[token_id]] for token_id in prompt_ids]
    )
    reference_prompt_embeddings = (
        model.get_input_embeddings().weight[prompt_ids].detach().cpu().float().numpy()
    )

    layer_manifest_path = args.bundle_dir / "layer_0" / "weights.json"
    q_weight = bundle_tensor(layer_manifest_path, "q_weight", (hidden, hidden))
    reference_q_weight = (
        model.model.layers[0].self_attn.q_proj.weight.detach().cpu().float().numpy()
    )
    attention = model.model.layers[0].self_attn
    for bundle_name, projection in (
        ("q_bias", attention.q_proj),
        ("k_bias", attention.k_proj),
        ("v_bias", attention.v_proj),
    ):
        output_size = int(projection.weight.shape[0])
        bundle_bias = bundle_tensor(layer_manifest_path, bundle_name, (output_size,))
        reference_bias = (
            projection.bias.detach().cpu().float().numpy()
            if projection.bias is not None
            else np.zeros(output_size, dtype=np.float32)
        )
        if not np.array_equal(bundle_bias, reference_bias):
            raise ValueError(
                f"--checkpoint-dir layer_0 {bundle_name} does not match --bundle-dir"
            )
    if not np.array_equal(bundle_prompt_embeddings, reference_prompt_embeddings):
        raise ValueError("--checkpoint-dir token embeddings do not match --bundle-dir")
    if not np.array_equal(q_weight, reference_q_weight):
        raise ValueError("--checkpoint-dir layer_0 weights do not match --bundle-dir")
    print("reference_checkpoint_match=passed", flush=True)


def run_reference_check(args: argparse.Namespace, runner_output: str) -> None:
    if args.checkpoint_dir is None:
        raise ValueError("--reference-check requires --checkpoint-dir")
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "--reference-check requires torch and transformers"
        ) from error

    prompt_ids = parse_token_ids(args.token_ids)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.checkpoint_dir),
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    verify_reference_checkpoint(args, model)
    generated_token_ids = parse_generated_token_ids(runner_output)
    reference_steps: list[tuple[int, float]] = []
    with torch.inference_mode():
        for step in range(len(generated_token_ids)):
            reference_input_ids = prompt_ids + generated_token_ids[:step]
            input_ids = torch.tensor([reference_input_ids], dtype=torch.long)
            reference_logits = model(input_ids=input_ids).logits[0, -1].float()
            reference_logit, reference_token = torch.max(reference_logits, dim=0)
            reference_steps.append(
                (int(reference_token.item()), float(reference_logit.item()))
            )
    verify_reference_outputs(args, runner_output, reference_steps)


def check_generation_regression(
    args: argparse.Namespace, tokenizer: Any | None, runner_output: str
) -> None:
    generated_token_ids = parse_generated_token_ids(runner_output)
    generated_text: str | None = None
    if tokenizer is not None:
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        print(
            "generated_text="
            + json.dumps(generated_text, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )
    if args.expect_generated_token_ids is not None:
        expected_token_ids = parse_token_ids(args.expect_generated_token_ids)
        if generated_token_ids != expected_token_ids:
            raise ValueError(
                "generated token regression mismatch: "
                f"actual={generated_token_ids}, expected={expected_token_ids}"
            )
    if args.expect_generated_text is not None:
        if generated_text != args.expect_generated_text:
            raise ValueError(
                "generated text regression mismatch: "
                f"actual={generated_text!r}, expected={args.expect_generated_text!r}"
            )
    if args.expect_stop_reason is not None:
        stop_reasons = STOP_REASON_RE.findall(runner_output)
        if len(stop_reasons) != 1:
            raise ValueError(
                "IREE runner output must contain exactly one stop_reason summary"
            )
        if stop_reasons[0] != args.expect_stop_reason:
            raise ValueError(
                "generation stop reason regression mismatch: "
                f"actual={stop_reasons[0]!r}, expected={args.expect_stop_reason!r}"
            )
    if (
        args.expect_generated_token_ids is not None
        or args.expect_generated_text is not None
        or args.expect_stop_reason is not None
    ):
        print("text_generation_regression=passed", flush=True)


def prompt_sequence_length(args: argparse.Namespace) -> int:
    return len(parse_token_ids(args.token_ids))


def validate_generation_length(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_seq_len <= 0:
        raise ValueError("--max-seq-len must be positive")
    prompt_len = prompt_sequence_length(args)
    if prompt_len + args.max_new_tokens > args.max_seq_len:
        raise ValueError(
            "--token-ids prompt length plus --max-new-tokens must not exceed "
            "--max-seq-len"
        )


def model_layer_count(checkpoint_dir: Path, config: Path | None) -> int:
    config_path = config if config is not None else checkpoint_dir / "config.json"
    data = read_json(config_path)
    value = data.get("num_hidden_layers")
    if not isinstance(value, int) or value <= 0:
        raise ValueError("config field 'num_hidden_layers' must be a positive integer")
    return value


def infer_bundle_layer_count(bundle_dir: Path) -> int:
    count = 0
    while (bundle_dir / f"layer_{count}" / "weights.json").exists():
        count += 1
    return count


def bundle_complete(bundle_dir: Path, layers: int) -> bool:
    if layers <= 0:
        return False
    for layer in range(layers):
        if not (bundle_dir / f"layer_{layer}" / "weights.json").exists():
            return False
    return (bundle_dir / "model_tail" / "weights.json").exists()


def iree_decode_bundle_complete(bundle_dir: Path) -> bool:
    return (bundle_dir / "manifest.json").is_file()


def iree_decode_bundle_manifest(bundle_dir: Path) -> dict[str, object] | None:
    manifest = bundle_dir / "manifest.json"
    if not manifest.is_file():
        return None
    return read_json(manifest)


def iree_decode_bundle_sequence_capacity(bundle_dir: Path) -> int | None:
    manifest = iree_decode_bundle_manifest(bundle_dir)
    if manifest is None:
        return None
    value = manifest.get("sequence_capacity")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{bundle_dir / 'manifest.json'} field 'sequence_capacity' must be "
            "a positive integer"
        )
    return value


def iree_precision_suffix(precision: str) -> str:
    return "" if precision == "f32" else f"_{precision}"


def iree_layer_module_name(max_cache_tokens: int, precision: str = "f32") -> str:
    return (
        f"qwen_decode_layer_kv_cache_max{max_cache_tokens}"
        f"{iree_precision_suffix(precision)}"
    )


def parse_iree_layer_capacity(path: Path) -> int | None:
    for part in [path.stem, path.parent.name]:
        match = IREE_LAYER_MODULE_RE.match(part)
        if match is not None:
            return int(match.group(1))
    return None


def discover_iree_cache_capacities(
    probe_dir: Path, target: str, precision: str = "f32"
) -> list[int]:
    capacities: list[int] = []
    if not probe_dir.is_dir():
        return capacities
    for child in probe_dir.iterdir():
        if not child.is_dir():
            continue
        capacity = parse_iree_layer_capacity(child)
        if capacity is None:
            continue
        module_name = iree_layer_module_name(capacity, precision)
        if child.name != module_name:
            continue
        vmfb = child / f"{module_name}_{target}.vmfb"
        if vmfb.is_file():
            capacities.append(capacity)
    return sorted(set(capacities))


def select_iree_cache_capacity(max_seq_len: int, capacities: list[int]) -> int:
    if max_seq_len <= 0:
        raise ValueError("--max-seq-len must be positive")
    for capacity in sorted(capacities):
        if max_seq_len <= capacity:
            return capacity
    if not capacities:
        raise ValueError(
            "no IREE Qwen decode layer VMFBs were found; build probe modules or "
            "pass --iree-layer-vmfb"
        )
    supported = ", ".join(str(value) for value in sorted(capacities))
    raise ValueError(
        f"--max-seq-len {max_seq_len} exceeds discovered IREE cache capacities "
        f"({supported})"
    )


def resolve_iree_cache_capacity(args: argparse.Namespace) -> int:
    validate_generation_length(args)
    if args.iree_layer_vmfb is not None:
        capacity = parse_iree_layer_capacity(args.iree_layer_vmfb)
        if capacity is not None:
            if args.max_seq_len > capacity:
                raise ValueError(
                    f"--max-seq-len {args.max_seq_len} exceeds explicit IREE "
                    f"layer VMFB cache capacity ({capacity})"
                )
            return capacity
        if args.max_seq_len <= 0:
            raise ValueError("--max-seq-len must be positive")
        return args.max_seq_len
    return select_iree_cache_capacity(
        args.max_seq_len,
        discover_iree_cache_capacities(
            args.iree_probe_dir, args.iree_target, args.iree_precision
        ),
    )


def default_iree_layer_vmfb(
    probe_dir: Path, target: str, max_cache_tokens: int, precision: str = "f32"
) -> Path:
    module_name = iree_layer_module_name(max_cache_tokens, precision)
    return probe_dir / module_name / f"{module_name}_{target}.vmfb"


def default_iree_tail_vmfb(
    probe_dir: Path, target: str, precision: str = "f32"
) -> Path:
    module_name = f"qwen_decode1_tail{iree_precision_suffix(precision)}"
    return probe_dir / module_name / f"{module_name}_{target}.vmfb"


def resolve_iree_vmfb_path(
    explicit_path: Path | None, default_path: Path, flag: str
) -> Path:
    if explicit_path is not None:
        return explicit_path
    if default_path.is_file():
        return default_path
    raise ValueError(
        f"{flag} was not provided and default VMFB was not found: {default_path}"
    )


def resolve_layers(args: argparse.Namespace) -> int:
    if args.layers is not None:
        if args.layers <= 0:
            raise ValueError("--layers must be positive")
        return args.layers
    if args.checkpoint_dir is not None:
        return model_layer_count(args.checkpoint_dir, args.config)
    inferred = infer_bundle_layer_count(args.bundle_dir)
    if inferred <= 0:
        raise ValueError("cannot infer layers; pass --layers or --checkpoint-dir")
    return inferred


def converter_command(args: argparse.Namespace, layers: int) -> list[str]:
    converter_args = [
        str(args.converter),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--layer",
        "0",
        "--layer-count",
        str(layers),
        "--keys",
        str(args.keys),
        "--token-ids",
        args.token_ids,
        "--output",
        str(args.bundle_dir),
    ]
    if args.backend == "iree":
        converter_args.extend(["--bundle-directory", "--full-token-embeddings"])
    if args.config is not None:
        converter_args.extend(["--config", str(args.config)])

    if args.python is not None:
        return [str(args.python), *converter_args]
    if args.no_uv or shutil.which("uv") is None:
        return [sys.executable, *converter_args]
    return [
        "uv",
        "run",
        "--with-requirements",
        str(args.requirements),
        "python",
        *converter_args,
    ]


def runner_command(args: argparse.Namespace, layers: int) -> list[str]:
    token_ids = parse_token_ids(args.token_ids)
    if args.keys < len(token_ids):
        raise ValueError("--keys must be at least the number of --token-ids")

    command = [
        str(args.python if args.python is not None else sys.executable),
        str(args.runner),
        "--bundle-dir",
        str(args.bundle_dir),
        "--layers",
        str(layers),
        "--keys",
        str(args.keys),
        "--token-ids",
        args.token_ids,
        "--iterations",
        str(args.iterations),
        "--valid-keys",
        str(len(token_ids)),
        "--no-warmup",
        "--e2e-check",
        "--sync-stack",
    ]
    if args.checkpoint_dir is not None:
        command.extend(["--checkpoint-dir", str(args.checkpoint_dir)])
    if args.config is not None:
        command.extend(["--config", str(args.config)])
    if args.force_convert:
        command.append("--force-convert")
    if args.no_convert:
        command.append("--no-convert")
    if args.no_uv:
        command.append("--no-uv")
    if args.trace_setup:
        command.append("--trace-setup")
    if args.trace_run:
        command.append("--trace-run")
    if args.benchmark is not None:
        command.extend(["--benchmark", str(args.benchmark)])
    return command


def iree_bundle_writer_command(args: argparse.Namespace) -> list[str]:
    validate_generation_length(args)
    cache_capacity = resolve_iree_cache_capacity(args)
    layer_vmfb = resolve_iree_vmfb_path(
        args.iree_layer_vmfb,
        default_iree_layer_vmfb(
            args.iree_probe_dir,
            args.iree_target,
            cache_capacity,
            args.iree_precision,
        ),
        "--iree-layer-vmfb",
    )
    tail_vmfb = resolve_iree_vmfb_path(
        args.iree_tail_vmfb,
        default_iree_tail_vmfb(
            args.iree_probe_dir, args.iree_target, args.iree_precision
        ),
        "--iree-tail-vmfb",
    )
    command = [
        str(args.python if args.python is not None else sys.executable),
        str(args.iree_bundle_writer),
        "--target",
        args.iree_target,
        "--precision",
        args.iree_precision,
        "--layer-vmfb",
        str(layer_vmfb),
        "--tail-vmfb",
        str(tail_vmfb),
        "--out-dir",
        str(args.iree_decode_bundle_dir),
        "--sequence-capacity",
        str(args.max_seq_len),
        "--max-cache-tokens",
        str(cache_capacity),
        "--layer-export",
        iree_layer_module_name(cache_capacity, args.iree_precision),
    ]
    if args.force_iree_bundle:
        command.append("--force")
    return command


def iree_runner_command(args: argparse.Namespace, layers: int) -> list[str]:
    validate_generation_length(args)
    manifest = iree_decode_bundle_manifest(args.iree_decode_bundle_dir)
    capacity = iree_decode_bundle_sequence_capacity(args.iree_decode_bundle_dir)
    if capacity is not None and args.max_seq_len > capacity:
        raise ValueError(
            f"--max-seq-len {args.max_seq_len} exceeds IREE decode bundle "
            f"sequence_capacity ({capacity})"
        )
    if manifest is not None:
        precision = manifest.get("precision", "f32")
        if precision != args.iree_precision:
            raise ValueError(
                f"{args.iree_decode_bundle_dir / 'manifest.json'} precision "
                f"({precision}) does not match --iree-precision "
                f"({args.iree_precision})"
            )
    command = [
        str(args.iree_runner),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-seq-len",
        str(args.max_seq_len),
        "--prompt-token-ids",
        args.token_ids,
    ]
    if args.resolved_eos_token_id is not None:
        command.extend(["--eos-token-id", str(args.resolved_eos_token_id)])
    if args.iree_verbose_layers:
        command.append("--verbose-layers")
    command.extend(
        [
            "--bundle",
            str(args.iree_decode_bundle_dir),
            str(args.bundle_dir),
            str(layers),
        ]
    )
    return command


def run_command(
    command: list[str], dry_run: bool, capture_stdout: bool = False
) -> str | None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    if dry_run:
        return None
    if not capture_stdout:
        subprocess.run(command, check=True)
        return None

    output: list[str] = []
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        text=True,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            output.append(line)
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return "".join(output)


def run_triton(args: argparse.Namespace, layers: int) -> None:
    run_command(runner_command(args, layers), args.dry_run)


def run_iree(args: argparse.Namespace, layers: int, tokenizer: Any | None) -> None:
    complete = bundle_complete(args.bundle_dir, layers)
    should_convert = args.force_convert or not complete
    if args.no_convert:
        should_convert = False
    if should_convert:
        if args.checkpoint_dir is None:
            raise ValueError(
                "weight bundle is incomplete; pass --checkpoint-dir or disable "
                "conversion with --no-convert"
            )
        run_command(converter_command(args, layers), args.dry_run)
    elif complete:
        print(f"reusing Qwen mini bundle: {args.bundle_dir}", flush=True)
    else:
        print(f"running with unchecked Qwen mini bundle: {args.bundle_dir}", flush=True)

    should_write_bundle = args.force_iree_bundle or not iree_decode_bundle_complete(
        args.iree_decode_bundle_dir
    )
    if args.no_iree_bundle_write:
        should_write_bundle = False
    if should_write_bundle:
        run_command(iree_bundle_writer_command(args), args.dry_run)
    elif iree_decode_bundle_complete(args.iree_decode_bundle_dir):
        print(
            f"reusing IREE Qwen decode bundle: {args.iree_decode_bundle_dir}",
            flush=True,
        )
    else:
        print(
            f"running with unchecked IREE decode bundle: {args.iree_decode_bundle_dir}",
            flush=True,
        )

    runner_output = run_command(
        iree_runner_command(args, layers),
        args.dry_run,
        capture_stdout=(
            tokenizer is not None
            or args.reference_check
            or args.expect_generated_token_ids is not None
            or args.expect_stop_reason is not None
        ),
    )
    if (
        tokenizer is not None
        or args.expect_generated_token_ids is not None
        or args.expect_stop_reason is not None
    ) and not args.dry_run:
        assert runner_output is not None
        check_generation_regression(args, tokenizer, runner_output)
    if args.reference_check and not args.dry_run:
        assert runner_output is not None
        run_reference_check(args, runner_output)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen mini E2E through lrrt: checkpoint conversion, decoder "
            "layer stack, final RMSNorm, lm_head logits, and finite-logit check."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bundle-dir", default=DEFAULT_BUNDLE_DIR, type=Path)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--keys", default=4, type=int)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--token-ids")
    prompt_group.add_argument(
        "--prompt",
        help="text prompt to tokenize for IREE execution",
    )
    prompt_group.add_argument(
        "--chat-user",
        help="Qwen chat-template user message to tokenize for IREE execution",
    )
    parser.add_argument(
        "--chat-system",
        help="Qwen chat-template system message; with --chat-user only",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        help=(
            "directory containing tokenizer.json; defaults to --checkpoint-dir "
            "with --prompt or --chat-user"
        ),
    )
    parser.add_argument("--iterations", default=1, type=int)
    parser.add_argument(
        "--max-new-tokens",
        default=1,
        type=int,
        help="maximum number of output tokens to decode",
    )
    parser.add_argument(
        "--eos-token-id",
        type=int,
        help=(
            "stop after this token; inferred from checkpoint/tokenizer config "
            "when available"
        ),
    )
    parser.add_argument(
        "--reference-check",
        action="store_true",
        help="compare every IREE top token/logit with local Hugging Face Qwen",
    )
    parser.add_argument(
        "--reference-logit-atol",
        default=0.05,
        type=float,
        help="absolute tolerance for --reference-check top-logit comparison",
    )
    parser.add_argument(
        "--expect-generated-token-ids",
        help="require the generated token IDs to match this comma-separated list",
    )
    parser.add_argument(
        "--expect-generated-text",
        help="require tokenizer-decoded generated text to match exactly",
    )
    parser.add_argument(
        "--expect-stop-reason",
        choices=["eos_token", "max_new_tokens"],
        help="require the native runner's generation stop reason to match",
    )
    parser.add_argument("--backend", choices=["triton", "iree"], default="triton")
    parser.add_argument("--iree", action="store_const", const="iree", dest="backend")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--no-convert", action="store_true")
    parser.add_argument("--no-uv", action="store_true")
    parser.add_argument("--trace-setup", action="store_true")
    parser.add_argument("--trace-run", action="store_true")
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--runner", default=DEFAULT_RUNNER, type=Path)
    parser.add_argument("--converter", default=DEFAULT_CONVERTER, type=Path)
    parser.add_argument("--requirements", default=DEFAULT_REQUIREMENTS, type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--iree-target", default=DEFAULT_IREE_TARGET)
    parser.add_argument(
        "--iree-precision",
        choices=("f32", "f16", "bf16"),
        default="f32",
        help="IREE device compute and KV-cache precision",
    )
    parser.add_argument("--iree-probe-dir", default=DEFAULT_IREE_PROBE_DIR, type=Path)
    parser.add_argument("--iree-layer-vmfb", type=Path)
    parser.add_argument("--iree-tail-vmfb", type=Path)
    parser.add_argument(
        "--iree-decode-bundle-dir", default=DEFAULT_IREE_DECODE_BUNDLE_DIR, type=Path
    )
    parser.add_argument(
        "--iree-bundle-writer", default=DEFAULT_IREE_BUNDLE_WRITER, type=Path
    )
    parser.add_argument("--iree-runner", default=DEFAULT_IREE_RUNNER, type=Path)
    parser.add_argument(
        "--iree-verbose-layers",
        action="store_true",
        help="print completion of every decoder layer in the native IREE runner",
    )
    parser.add_argument("--max-seq-len", default=8, type=int)
    parser.add_argument("--force-iree-bundle", action="store_true")
    parser.add_argument("--no-iree-bundle-write", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.reference_logit_atol < 0:
        parser.error("--reference-logit-atol must be non-negative")
    if args.expect_generated_token_ids is not None:
        try:
            parse_token_ids(args.expect_generated_token_ids)
        except (ValueError, TypeError) as error:
            parser.error(f"invalid --expect-generated-token-ids: {error}")
    if args.expect_generated_text is not None and (
        args.prompt is None and args.chat_user is None and args.tokenizer_dir is None
    ):
        parser.error(
            "--expect-generated-text requires --prompt, --chat-user, or --tokenizer-dir"
        )
    if args.chat_system is not None and args.chat_user is None:
        parser.error("--chat-system requires --chat-user")
    if args.token_ids is None and args.prompt is None and args.chat_user is None:
        args.token_ids = "0,1,2"
    args.resolved_eos_token_id = None
    return args


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        tokenizer = prepare_text_inputs(args)
        args.resolved_eos_token_id = resolve_eos_token_id(args, tokenizer)
        layers = resolve_layers(args)
        if args.backend == "iree":
            run_iree(args, layers, tokenizer)
        else:
            run_triton(args, layers)
        return 0
    except subprocess.CalledProcessError as error:
        return error.returncode
    except Exception as error:
        print(f"run_qwen_e2e.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
