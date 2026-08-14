#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import torch
from lrrt_sparsewave import Executor
from sparse_attention import SparseAttention, make_example_inputs

KERNEL_SCHEDULE = (
    "sparse_attention_scores",
    "sparse_attention_row_max",
    "sparse_attention_exp",
    "sparse_attention_row_sum",
    "sparse_attention_normalize",
    "sparse_attention_output",
)


def announce(component, message):
    print(f"[{component}] {message}", flush=True)


def run(command):
    subprocess.run(command, check=True, stderr=subprocess.STDOUT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile and run the SparseWave SparseAttention E2E demo."
    )
    parser.add_argument("--bundle-tool", type=Path, required=True)
    parser.add_argument("--sparsewave-opt", type=Path, required=True)
    parser.add_argument("--sparsewave-pytorch-opt", type=Path, required=True)
    parser.add_argument("--chip", default="gfx1101")
    parser.add_argument("--rocm-path", type=Path, default=Path("/opt/rocm"))
    return parser.parse_args()


def main():
    args = parse_args()
    example_dir = Path(__file__).resolve().parent
    warnings.filterwarnings("ignore", category=UserWarning)

    mask, query, key, value = make_example_inputs()
    announce("E2E", "PyTorch CSR-masked scaled dot-product attention")
    announce(
        "E2E",
        f"Q={list(query.shape)} K={list(key.shape)} V={list(value.shape)} "
        f"NNZ={mask.values().numel()}",
    )

    with tempfile.TemporaryDirectory(prefix="lrrt-sparsewave-attention-") as temp:
        work_dir = Path(temp)
        torch_mlir = work_dir / "sparse_attention.torch.mlir"
        bundle = work_dir / "sparse_attention.bundle"

        announce("PyTorch frontend", "Exporting the model to Torch MLIR")
        run(
            [
                sys.executable,
                "-W",
                "ignore",
                str(example_dir / "export.py"),
                "--output",
                str(torch_mlir),
            ]
        )
        announce("PyTorch frontend", "Torch MLIR export complete")

        announce("SparseWave compiler", f"Compiling for {args.chip}")
        run(
            [
                str(args.bundle_tool),
                str(torch_mlir),
                "--output",
                str(bundle),
                "--sparsewave-opt",
                str(args.sparsewave_opt),
                "--sparsewave-pytorch-opt",
                str(args.sparsewave_pytorch_opt),
                "--chip",
                args.chip,
                "--operation",
                "sparse-attention",
                "--rocm-path",
                str(args.rocm_path),
                "--block-size",
                "64",
                "--wavefront-size",
                "32",
            ]
        )
        run([str(args.bundle_tool), "--verify", str(bundle)])

        manifest = bundle / "manifest.json"
        document = json.loads(manifest.read_text())
        generated_kernels = {kernel["name"] for kernel in document["kernels"]}
        if generated_kernels != set(KERNEL_SCHEDULE):
            raise RuntimeError(
                "SparseWave generated an unexpected SparseAttention kernel set"
            )
        announce(
            "SparseWave compiler",
            f"Generated and verified {len(KERNEL_SCHEDULE)} HSACO kernels",
        )

        announce("lrrt Executor", "Loading the bundle manifest")
        executor = Executor.load(manifest)
        announce("lrrt Executor", "Application schedule")
        for index, kernel in enumerate(KERNEL_SCHEDULE, start=1):
            print(f"  {index}. {kernel}", flush=True)

        announce(
            "lrrt Runtime",
            f"Loading HSACO and launching {len(KERNEL_SCHEDULE)} kernels on "
            f"{document['target']}",
        )
        output = executor(mask, query, key, value)
        announce("lrrt Runtime", "Queue synchronized; output copied to the host")

        expected = SparseAttention(query.shape[1])(mask, query, key, value)
        torch.testing.assert_close(output, expected)
        announce("Validation", "PASSED against the PyTorch reference")
        print("output:", flush=True)
        print(output, flush=True)


if __name__ == "__main__":
    main()
