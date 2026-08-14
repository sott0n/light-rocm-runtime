#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from csr_spmm import CSRSpMM, export_csr_spmm
from sparsewave.torch_export import import_torch_program, render_generic_torch_mlir


def main():
    parser = argparse.ArgumentParser(description="Export the lrrt SpMM example.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix = torch.sparse_csr_tensor(
        torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32),
        torch.tensor([0, 3, 1, 7, 2, 4, 5, 6], dtype=torch.int32),
        torch.tensor(
            [1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 4.0, 1.5],
            dtype=torch.float32,
        ),
        size=(4, 8),
        check_invariants=True,
    )
    rhs = torch.arange(1, 33, dtype=torch.float32).reshape(8, 4) * 0.25
    exported = export_csr_spmm(matrix, rhs)
    torch.testing.assert_close(exported.module()(matrix, rhs), CSRSpMM()(matrix, rhs))
    imported = import_torch_program(exported, function_name="main")
    args.output.write_text(render_generic_torch_mlir(imported) + "\n")


if __name__ == "__main__":
    main()
