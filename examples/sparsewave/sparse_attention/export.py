#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from sparse_attention import (
    SparseAttention,
    export_sparse_attention,
    make_example_inputs,
)
from sparsewave.torch_export import import_torch_program, render_generic_torch_mlir


def main():
    parser = argparse.ArgumentParser(
        description="Export the lrrt SparseAttention example."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = make_example_inputs()
    exported = export_sparse_attention(*inputs)
    expected = SparseAttention(inputs[1].shape[1])(*inputs)
    torch.testing.assert_close(exported.module()(*inputs), expected)
    imported = import_torch_program(exported, function_name="main")
    args.output.write_text(render_generic_torch_mlir(imported) + "\n")


if __name__ == "__main__":
    main()
