#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from lrrt_sparsewave import Executor


def main():
    parser = argparse.ArgumentParser(description="Run SparseWave SpMM with lrrt.")
    parser.add_argument("--manifest", type=Path, required=True)
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
    executor = Executor.load(args.manifest, shape=(4, 8, 4))
    output = executor(matrix, rhs)
    torch.testing.assert_close(output, torch.sparse.mm(matrix, rhs))
    print(output)


if __name__ == "__main__":
    main()
