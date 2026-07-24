#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import importlib.util
import io
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


class FakeEncoding:
    def __init__(self, ids: list[int]):
        self.ids = ids


class FakeTokenizer:
    def __init__(self, encoded_ids: list[int] | None = None):
        self.encoded_ids = encoded_ids or []
        self.encoded_text: str | None = None
        self.decoded_ids: list[int] | None = None
        self.skip_special_tokens: bool | None = None
        self.token_ids: dict[str, int] = {}

    def encode(self, text: str) -> FakeEncoding:
        self.encoded_text = text
        return FakeEncoding(self.encoded_ids)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        self.decoded_ids = token_ids
        self.skip_special_tokens = skip_special_tokens
        return "decoded text"

    def token_to_id(self, token: str) -> int | None:
        return self.token_ids.get(token)


def test_parse_args_preserves_default_token_ids() -> None:
    runner = load_runner()
    args = runner.parse_args(["--layers", "1"])
    assert args.token_ids == "0,1,2"


def test_prepare_text_prompt_uses_checkpoint_tokenizer() -> None:
    runner = load_runner()
    tokenizer = FakeTokenizer([10, 20, 30])
    loaded_paths: list[Path] = []

    def fake_load_tokenizer(path: Path) -> FakeTokenizer:
        loaded_paths.append(path)
        return tokenizer

    runner.load_tokenizer = fake_load_tokenizer
    args = runner.parse_args(
        [
            "--iree",
            "--checkpoint-dir",
            "/tmp/qwen-checkpoint",
            "--prompt",
            "hello",
        ]
    )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        loaded = runner.prepare_text_inputs(args)

    assert loaded is tokenizer
    assert loaded_paths == [Path("/tmp/qwen-checkpoint")]
    assert tokenizer.encoded_text == "hello"
    assert args.token_ids == "10,20,30"
    assert output.getvalue() == "prompt_token_ids=[10,20,30]\n"


def test_prepare_text_prompt_requires_tokenizer_location() -> None:
    runner = load_runner()
    args = runner.parse_args(["--iree", "--layers", "1", "--prompt", "hello"])
    try:
        runner.prepare_text_inputs(args)
    except ValueError as error:
        assert "--tokenizer-dir or --checkpoint-dir" in str(error)
    else:
        raise AssertionError("expected missing tokenizer location to fail")


def test_text_options_require_iree() -> None:
    runner = load_runner()
    args = runner.parse_args(
        ["--layers", "1", "--token-ids", "1,2", "--tokenizer-dir", "/tmp"]
    )
    try:
        runner.prepare_text_inputs(args)
    except ValueError as error:
        assert "currently require --iree" in str(error)
    else:
        raise AssertionError("expected tokenizer-backed Triton run to fail")


def test_reference_check_requires_iree() -> None:
    runner = load_runner()
    args = runner.parse_args(["--layers", "1", "--reference-check"])
    try:
        runner.prepare_text_inputs(args)
    except ValueError as error:
        assert "--reference-check requires --iree" in str(error)
    else:
        raise AssertionError("expected reference-backed Triton run to fail")


def test_generation_regression_requires_iree() -> None:
    runner = load_runner()
    args = runner.parse_args(["--layers", "1", "--expect-generated-token-ids", "7,8"])
    try:
        runner.prepare_text_inputs(args)
    except ValueError as error:
        assert "--expect-generated-* requires --iree" in str(error)
    else:
        raise AssertionError("expected regression-backed Triton run to fail")


def test_resolve_eos_token_id_prefers_explicit_value() -> None:
    runner = load_runner()
    args = runner.parse_args(["--iree", "--layers", "1", "--eos-token-id", "151645"])
    assert runner.resolve_eos_token_id(args, tokenizer=None) == 151645


def test_resolve_eos_token_id_reads_checkpoint_config() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir)
        (checkpoint / "config.json").write_text(
            json.dumps({"eos_token_id": 151643}), encoding="utf-8"
        )
        args = runner.parse_args(
            ["--iree", "--layers", "1", "--checkpoint-dir", str(checkpoint)]
        )
        assert runner.resolve_eos_token_id(args, tokenizer=None) == 151643


def test_resolve_eos_token_id_uses_first_generation_config_value() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir)
        (checkpoint / "generation_config.json").write_text(
            json.dumps({"eos_token_id": [151645, 151643]}), encoding="utf-8"
        )
        args = runner.parse_args(
            ["--iree", "--layers", "1", "--checkpoint-dir", str(checkpoint)]
        )
        assert runner.resolve_eos_token_id(args, tokenizer=None) == 151645


def test_resolve_eos_token_id_reads_tokenizer_config() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer_dir = Path(tmpdir)
        (tokenizer_dir / "tokenizer_config.json").write_text(
            json.dumps({"eos_token": "<|im_end|>"}), encoding="utf-8"
        )
        tokenizer = FakeTokenizer()
        tokenizer.token_ids["<|im_end|>"] = 151645
        args = runner.parse_args(
            ["--iree", "--layers", "1", "--tokenizer-dir", str(tokenizer_dir)]
        )
        assert runner.resolve_eos_token_id(args, tokenizer) == 151645


def test_check_generation_regression_decodes_runner_summary() -> None:
    runner = load_runner()
    tokenizer = FakeTokenizer()
    args = runner.parse_args(
        [
            "--iree",
            "--layers",
            "1",
            "--tokenizer-dir",
            "/tmp",
            "--expect-generated-token-ids",
            "120952,67330",
            "--expect-generated-text",
            "decoded text",
        ]
    )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        runner.check_generation_regression(
            args,
            tokenizer,
            "step 1 top_token=120952\n"
            "generated_token_ids=[120952,67330]\n"
            "iree_qwen_decode2_e2e: ok layers=24\n",
        )

    assert tokenizer.decoded_ids == [120952, 67330]
    assert tokenizer.skip_special_tokens is True
    assert output.getvalue() == (
        'generated_text="decoded text"\ntext_generation_regression=passed\n'
    )


def test_check_generation_regression_rejects_token_mismatch() -> None:
    runner = load_runner()
    args = runner.parse_args(
        [
            "--iree",
            "--layers",
            "1",
            "--expect-generated-token-ids",
            "7,9",
        ]
    )
    try:
        runner.check_generation_regression(
            args, tokenizer=None, runner_output="generated_token_ids=[7,8]\n"
        )
    except ValueError as error:
        assert "generated token regression mismatch" in str(error)
    else:
        raise AssertionError("expected generated token mismatch to fail")


def test_parse_generated_token_ids_requires_one_summary() -> None:
    runner = load_runner()
    for output in ["", "generated_token_ids=[1]\ngenerated_token_ids=[2]\n"]:
        try:
            runner.parse_generated_token_ids(output)
        except ValueError as error:
            assert "exactly one" in str(error)
        else:
            raise AssertionError("expected invalid generated token output to fail")


def test_parse_iree_top_logit_reads_requested_step() -> None:
    runner = load_runner()
    output = (
        "step 1 top_token=120952 logit=3.25 vocab=151936\n"
        "step 2 top_token=67330 logit=-1.5e-2 vocab=151936\n"
    )
    assert runner.parse_iree_top_logit(output) == (120952, 3.25)
    assert runner.parse_iree_top_logit(output, step=2) == (67330, -0.015)


def test_verify_reference_outputs_checks_every_generated_step() -> None:
    runner = load_runner()
    args = runner.parse_args(
        ["--iree", "--layers", "1", "--reference-logit-atol", "0.05"]
    )
    runner_output = (
        "step 1 top_token=7 logit=3.25 vocab=10\n"
        "step 2 top_token=8 logit=4.5 vocab=10\n"
        "generated_token_ids=[7,8]\n"
    )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        runner.verify_reference_outputs(args, runner_output, [(7, 3.24), (8, 4.48)])

    assert "reference_step=1 top_token=7" in output.getvalue()
    assert "reference_step=2 top_token=8" in output.getvalue()
    assert "reference_check=passed steps=2" in output.getvalue()


def test_verify_reference_outputs_rejects_later_step_mismatch() -> None:
    runner = load_runner()
    args = runner.parse_args(["--iree", "--layers", "1"])
    runner_output = (
        "step 1 top_token=7 logit=3.25 vocab=10\n"
        "step 2 top_token=8 logit=4.5 vocab=10\n"
        "generated_token_ids=[7,8]\n"
    )
    try:
        runner.verify_reference_outputs(args, runner_output, [(7, 3.25), (9, 4.5)])
    except ValueError as error:
        assert "step 2 top token mismatch" in str(error)
    else:
        raise AssertionError("expected later reference token mismatch to fail")


def test_bundle_tensor_reads_named_tensor() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "weights.bin").write_bytes(b"\x00\x00\x80\x3f\x00\x00\x00\x40")
        manifest = root / "weights.json"
        manifest.write_text(
            json.dumps(
                {
                    "data": "weights.bin",
                    "tensors": [
                        {"name": "values", "offset": 0, "count": 2},
                    ],
                }
            ),
            encoding="utf-8",
        )
        values = runner.bundle_tensor(manifest, "values", (1, 2))
        assert values.tolist() == [[1.0, 2.0]]


def test_run_command_can_tee_and_capture_stdout() -> None:
    runner = load_runner()
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        captured = runner.run_command(
            [sys.executable, "-c", "print('generated_token_ids=[7,8]')"],
            dry_run=False,
            capture_stdout=True,
        )

    assert captured == "generated_token_ids=[7,8]\n"
    assert output.getvalue().endswith("generated_token_ids=[7,8]\n")


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


def write_iree_probe_vmfb(probe: Path, capacity: int, target: str = "gfx1101") -> Path:
    module = f"qwen_decode_layer_kv_cache_max{capacity}"
    layer_vmfb = probe / module / f"{module}_{target}.vmfb"
    layer_vmfb.parent.mkdir(parents=True)
    layer_vmfb.write_text("layer", encoding="utf-8")
    return layer_vmfb


def write_iree_tail_vmfb(probe: Path, target: str = "gfx1101") -> Path:
    tail_vmfb = probe / "qwen_decode1_tail" / f"qwen_decode1_tail_{target}.vmfb"
    tail_vmfb.parent.mkdir(parents=True)
    tail_vmfb.write_text("tail", encoding="utf-8")
    return tail_vmfb


def write_iree_decode_manifest(bundle: Path, sequence_capacity: int) -> None:
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "sequence_capacity": sequence_capacity,
                "max_cache_tokens": max(sequence_capacity, 16),
                "target": "gfx1101",
            }
        ),
        encoding="utf-8",
    )


def test_iree_runner_command_uses_decode_bundle() -> None:
    runner = load_runner()
    args = runner.parse_args(
        [
            "--iree",
            "--bundle-dir",
            "/tmp/qwen-weights",
            "--layers",
            "2",
            "--max-new-tokens",
            "3",
            "--token-ids",
            "5,6",
            "--iree-decode-bundle-dir",
            "/tmp/iree-decode-bundle",
            "--iree-runner",
            "/tmp/lrrt_iree_qwen_decode1_e2e",
        ]
    )
    command = runner.iree_runner_command(args, 2)
    assert command == [
        "/tmp/lrrt_iree_qwen_decode1_e2e",
        "--max-new-tokens",
        "3",
        "--max-seq-len",
        "8",
        "--prompt-token-ids",
        "5,6",
        "--bundle",
        "/tmp/iree-decode-bundle",
        "/tmp/qwen-weights",
        "2",
    ]


def test_iree_runner_command_passes_resolved_eos_token_id() -> None:
    runner = load_runner()
    args = runner.parse_args(
        [
            "--iree",
            "--bundle-dir",
            "/tmp/qwen-weights",
            "--layers",
            "2",
            "--token-ids",
            "5,6",
            "--iree-decode-bundle-dir",
            "/tmp/iree-decode-bundle",
        ]
    )
    args.resolved_eos_token_id = 151643

    command = runner.iree_runner_command(args, 2)

    eos_index = command.index("--eos-token-id")
    assert command[eos_index + 1] == "151643"
    assert eos_index < command.index("--bundle")


def test_iree_runner_command_accepts_max_supported_generation_capacity() -> None:
    runner = load_runner()
    args = runner.parse_args(
        [
            "--iree",
            "--bundle-dir",
            "/tmp/qwen-weights",
            "--layers",
            "24",
            "--max-new-tokens",
            "32",
            "--max-seq-len",
            "33",
            "--token-ids",
            "0",
            "--iree-decode-bundle-dir",
            "/tmp/iree-decode-bundle",
            "--iree-runner",
            "/tmp/lrrt_iree_qwen_decode1_e2e",
        ]
    )

    command = runner.iree_runner_command(args, 24)

    assert "--max-new-tokens" in command
    assert "32" in command
    assert "--max-seq-len" in command
    assert command[-2:] == ["/tmp/qwen-weights", "24"]


def test_iree_converter_command_writes_e2e_directory_bundle() -> None:
    runner = load_runner()
    args = runner.parse_args(
        [
            "--iree",
            "--checkpoint-dir",
            "/tmp/qwen-checkpoint",
            "--bundle-dir",
            "/tmp/qwen-weights",
            "--layers",
            "1",
            "--keys",
            "4",
            "--token-ids",
            "0,1,2",
            "--python",
            "/tmp/python",
            "--no-uv",
        ]
    )

    command = runner.converter_command(args, 1)
    assert command == [
        "/tmp/python",
        str(runner.DEFAULT_CONVERTER),
        "--checkpoint-dir",
        "/tmp/qwen-checkpoint",
        "--layer",
        "0",
        "--layer-count",
        "1",
        "--keys",
        "4",
        "--token-ids",
        "0,1,2",
        "--output",
        "/tmp/qwen-weights",
        "--bundle-directory",
        "--full-token-embeddings",
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
                "--max-new-tokens",
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


def test_iree_dry_run_discovers_smallest_sufficient_vmfb_capacity() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        weights = root / "weights"
        write_complete_weight_bundle(weights, 1)
        probe = root / "probe"
        write_iree_probe_vmfb(probe, 32)
        layer_vmfb = write_iree_probe_vmfb(probe, 64)
        tail_vmfb = write_iree_tail_vmfb(probe)

        args = runner.parse_args(
            [
                "--iree",
                "--bundle-dir",
                str(weights),
                "--layers",
                "1",
                "--max-new-tokens",
                "40",
                "--max-seq-len",
                "41",
                "--token-ids",
                "0",
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
        assert "--max-cache-tokens" in command
        assert "64" in command
        assert "--sequence-capacity" in command
        assert "41" in command
        assert "--layer-export" in command
        assert "qwen_decode_layer_kv_cache_max64" in command


def test_iree_rejects_sequence_length_without_discovered_cache_capacity() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        weights = root / "weights"
        write_complete_weight_bundle(weights, 1)
        args = runner.parse_args(
            [
                "--iree",
                "--bundle-dir",
                str(weights),
                "--layers",
                "1",
                "--max-new-tokens",
                "33",
                "--max-seq-len",
                "34",
                "--token-ids",
                "0",
                "--iree-probe-dir",
                str(root / "probe"),
                "--iree-decode-bundle-dir",
                str(root / "iree-bundle"),
                "--iree-runner",
                "/tmp/lrrt_iree_qwen_decode1_e2e",
                "--dry-run",
            ]
        )

        try:
            runner.iree_bundle_writer_command(args)
        except ValueError as error:
            assert "no IREE Qwen decode layer VMFBs were found" in str(error)
        else:
            raise AssertionError("expected unsupported max sequence length to fail")


def test_iree_rejects_sequence_length_over_discovered_cache_capacity() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        weights = root / "weights"
        write_complete_weight_bundle(weights, 1)
        probe = root / "probe"
        write_iree_probe_vmfb(probe, 32)
        args = runner.parse_args(
            [
                "--iree",
                "--bundle-dir",
                str(weights),
                "--layers",
                "1",
                "--max-new-tokens",
                "40",
                "--max-seq-len",
                "41",
                "--token-ids",
                "0",
                "--iree-probe-dir",
                str(probe),
                "--iree-decode-bundle-dir",
                str(root / "iree-bundle"),
                "--iree-runner",
                "/tmp/lrrt_iree_qwen_decode1_e2e",
                "--dry-run",
            ]
        )

        try:
            runner.iree_bundle_writer_command(args)
        except ValueError as error:
            assert "--max-seq-len 41 exceeds discovered IREE cache capacities" in str(
                error
            )
            assert "(32)" in str(error)
        else:
            raise AssertionError("expected capacity validation failure")


def test_iree_rejects_output_length_over_sequence_capacity() -> None:
    runner = load_runner()
    args = runner.parse_args(
        [
            "--iree",
            "--bundle-dir",
            "/tmp/qwen-weights",
            "--layers",
            "1",
            "--max-new-tokens",
            "9",
            "--max-seq-len",
            "8",
            "--token-ids",
            "0",
            "--iree-decode-bundle-dir",
            "/tmp/iree-decode-bundle",
            "--iree-runner",
            "/tmp/lrrt_iree_qwen_decode1_e2e",
        ]
    )

    try:
        runner.iree_runner_command(args, 1)
    except ValueError as error:
        assert (
            "--token-ids prompt length plus --max-new-tokens must not exceed "
            "--max-seq-len"
        ) in str(error)
    else:
        raise AssertionError("expected output length over sequence capacity to fail")


def test_iree_runner_rejects_sequence_length_over_bundle_manifest_capacity() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        bundle = root / "iree-bundle"
        write_iree_decode_manifest(bundle, sequence_capacity=8)
        args = runner.parse_args(
            [
                "--iree",
                "--bundle-dir",
                "/tmp/qwen-weights",
                "--layers",
                "1",
                "--max-new-tokens",
                "9",
                "--max-seq-len",
                "10",
                "--token-ids",
                "0",
                "--iree-decode-bundle-dir",
                str(bundle),
                "--iree-runner",
                "/tmp/lrrt_iree_qwen_decode1_e2e",
            ]
        )

        try:
            runner.iree_runner_command(args, 1)
        except ValueError as error:
            assert "exceeds IREE decode bundle sequence_capacity (8)" in str(error)
        else:
            raise AssertionError("expected bundle sequence capacity validation failure")


def test_iree_runner_rejects_prompt_and_generation_over_max_seq_len() -> None:
    runner = load_runner()
    args = runner.parse_args(
        [
            "--iree",
            "--bundle-dir",
            "/tmp/qwen-weights",
            "--layers",
            "1",
            "--token-ids",
            "3,4,5",
            "--max-new-tokens",
            "2",
            "--max-seq-len",
            "4",
            "--iree-decode-bundle-dir",
            "/tmp/iree-decode-bundle",
            "--iree-runner",
            "/tmp/lrrt_iree_qwen_decode1_e2e",
        ]
    )

    try:
        runner.iree_runner_command(args, 1)
    except ValueError as error:
        assert (
            "--token-ids prompt length plus --max-new-tokens must not exceed "
            "--max-seq-len"
        ) in str(error)
    else:
        raise AssertionError("expected prompt length validation failure")


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
    test_parse_args_preserves_default_token_ids()
    test_prepare_text_prompt_uses_checkpoint_tokenizer()
    test_prepare_text_prompt_requires_tokenizer_location()
    test_text_options_require_iree()
    test_reference_check_requires_iree()
    test_generation_regression_requires_iree()
    test_resolve_eos_token_id_prefers_explicit_value()
    test_resolve_eos_token_id_reads_checkpoint_config()
    test_resolve_eos_token_id_uses_first_generation_config_value()
    test_resolve_eos_token_id_reads_tokenizer_config()
    test_check_generation_regression_decodes_runner_summary()
    test_check_generation_regression_rejects_token_mismatch()
    test_parse_generated_token_ids_requires_one_summary()
    test_parse_iree_top_logit_reads_requested_step()
    test_verify_reference_outputs_checks_every_generated_step()
    test_verify_reference_outputs_rejects_later_step_mismatch()
    test_bundle_tensor_reads_named_tensor()
    test_run_command_can_tee_and_capture_stdout()
    test_resolve_layers_reads_checkpoint_config()
    test_runner_command_requires_keys_for_tokens()
    test_dry_run_builds_full_e2e_command()
    test_iree_runner_command_uses_decode_bundle()
    test_iree_runner_command_passes_resolved_eos_token_id()
    test_iree_runner_command_accepts_max_supported_generation_capacity()
    test_iree_converter_command_writes_e2e_directory_bundle()
    test_iree_dry_run_writes_bundle_then_runs()
    test_iree_dry_run_discovers_smallest_sufficient_vmfb_capacity()
    test_iree_rejects_sequence_length_without_discovered_cache_capacity()
    test_iree_rejects_sequence_length_over_discovered_cache_capacity()
    test_iree_rejects_output_length_over_sequence_capacity()
    test_iree_runner_rejects_sequence_length_over_bundle_manifest_capacity()
    test_iree_runner_rejects_prompt_and_generation_over_max_seq_len()
    test_iree_explicit_vmfb_inputs_override_discovery()
    test_iree_requires_vmfb_inputs_when_bundle_and_default_vmfb_are_missing()
    print("qwen_e2e_runner_test: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
