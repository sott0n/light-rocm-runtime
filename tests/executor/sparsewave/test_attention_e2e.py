import math
import os
from pathlib import Path

import pytest
import torch
from lrrt_sparsewave import Executor


def make_inputs():
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
    return mask, query, key, value


def make_dynamic_nnz_inputs():
    mask = torch.sparse_csr_tensor(
        torch.tensor([0, 1, 4], dtype=torch.int32),
        torch.tensor([1, 0, 1, 2], dtype=torch.int32),
        torch.zeros(4, dtype=torch.float32),
        size=(2, 3),
        check_invariants=True,
    )
    query = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.float32)
    key = torch.tensor([[1.0, 2.0], [-2.0, 0.5], [0.25, -1.0]], dtype=torch.float32)
    value = torch.tensor([[2.0, -1.0], [0.5, 4.0], [3.0, 1.5]], dtype=torch.float32)
    return mask, query, key, value


def make_empty_mask_inputs():
    mask = torch.sparse_csr_tensor(
        torch.tensor([0, 0, 0], dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.float32),
        size=(2, 3),
        check_invariants=True,
    )
    query = torch.tensor([[1.0, -2.0], [3.0, 0.5]], dtype=torch.float32)
    key = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    return mask, query, key, value


def reference(mask, query, key, value):
    scores = torch.sparse.sampled_addmm(
        mask,
        query,
        key.transpose(0, 1),
        beta=0.0,
        alpha=1.0 / math.sqrt(query.shape[1]),
    )
    probabilities = torch.sparse.softmax(scores.to_sparse_coo(), dim=1)
    return torch.sparse.mm(probabilities, value)


@pytest.fixture(scope="module")
def manifest():
    return Path(os.environ["LRRT_SPARSEWAVE_ATTENTION_MANIFEST"])


@pytest.fixture(scope="module")
def executor(manifest):
    return Executor.load(manifest)


@pytest.mark.parametrize(
    "make_case", [make_inputs, make_dynamic_nnz_inputs, make_empty_mask_inputs]
)
def test_matches_pytorch_with_dynamic_nnz(executor, make_case):
    inputs = make_case()
    torch.testing.assert_close(executor(*inputs), reference(*inputs))


def test_shape_is_loaded_from_manifest(manifest):
    with pytest.raises(ValueError, match="reads its shape from the manifest"):
        Executor.load(manifest, shape=(2, 3, 2))


def test_rejects_an_incompatible_shape(executor):
    mask, query, key, value = make_inputs()
    with pytest.raises(ValueError, match="do not match compiled shapes"):
        executor(mask, query[:, :1], key, value)
