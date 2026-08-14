#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from sparsewave.torch_export import import_torch_program, render_generic_torch_mlir


class CSRSpMM(torch.nn.Module):
    def forward(self, matrix, rhs):
        return torch.sparse.mm(matrix, rhs)


def make_inputs():
    row_offsets = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32)
    column_indices = torch.tensor([0, 3, 1, 7, 2, 4, 5, 6], dtype=torch.int32)
    values = torch.tensor(
        [1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 4.0, 1.5],
        dtype=torch.float32,
    )
    matrix = torch.sparse_csr_tensor(
        row_offsets,
        column_indices,
        values,
        size=(4, 8),
        check_invariants=True,
    )
    rhs = torch.arange(1, 33, dtype=torch.float32).reshape(8, 4) * 0.25
    return matrix, rhs


def main():
    parser = argparse.ArgumentParser(
        description="Export the fixed lrrt SpMM graph through PyTorch."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix, rhs = make_inputs()
    module = CSRSpMM()
    exported = torch.export.export(module, (matrix, rhs), strict=True)
    torch.testing.assert_close(exported.module()(matrix, rhs), module(matrix, rhs))
    imported = import_torch_program(exported, function_name="spmm")
    args.output.write_text(render_generic_torch_mlir(imported) + "\n")


if __name__ == "__main__":
    main()
