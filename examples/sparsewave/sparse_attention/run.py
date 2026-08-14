#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import torch
from lrrt_sparsewave import Executor


def main():
    parser = argparse.ArgumentParser(
        description="Run SparseWave SparseAttention with lrrt."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    mask = torch.sparse_csr_tensor(
        torch.tensor([0, 2, 3], dtype=torch.int32),
        torch.tensor([0, 2, 1], dtype=torch.int32),
        torch.zeros(3, dtype=torch.float32),
        size=(2, 3),
        check_invariants=True,
    )
    query = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    key = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)

    executor = Executor.load(args.manifest)
    output = executor(mask, query, key, value)
    scores = torch.sparse.sampled_addmm(
        mask,
        query,
        key.transpose(0, 1),
        beta=0.0,
        alpha=1.0 / math.sqrt(query.shape[1]),
    )
    probabilities = torch.sparse.softmax(scores.to_sparse_coo(), dim=1)
    torch.testing.assert_close(output, torch.sparse.mm(probabilities, value))
    print(output)


if __name__ == "__main__":
    main()
