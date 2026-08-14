import os
from pathlib import Path

import pytest
import torch
from lrrt_sparsewave import Executor


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


def make_dynamic_nnz_input():
    matrix = torch.sparse_csr_tensor(
        torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        torch.tensor([2.0, -1.0, 0.5, 3.0], dtype=torch.float32),
        size=(4, 8),
        check_invariants=True,
    )
    rhs = torch.arange(1, 33, dtype=torch.float32).reshape(8, 4) * 0.125
    return matrix, rhs


@pytest.fixture(scope="module")
def manifest():
    return Path(os.environ["LRRT_SPARSEWAVE_SPMM_MANIFEST"])


@pytest.fixture(scope="module")
def executor(manifest):
    return Executor.load(manifest, shape=(4, 8, 4))


@pytest.mark.parametrize("make_case", [make_inputs, make_dynamic_nnz_input])
def test_matches_pytorch_with_dynamic_nnz(executor, make_case):
    matrix, rhs = make_case()
    torch.testing.assert_close(executor(matrix, rhs), torch.sparse.mm(matrix, rhs))


def test_multiple_executors_share_the_runtime(executor, manifest):
    second_executor = Executor.load(manifest, shape=(4, 8, 4))
    matrix, rhs = make_inputs()
    torch.testing.assert_close(
        second_executor(matrix, rhs), torch.sparse.mm(matrix, rhs)
    )


def test_load_requires_application_configuration(manifest):
    with pytest.raises(ValueError, match="requires its compiled shape"):
        Executor.load(manifest)


def test_rejects_an_incompatible_shape(executor):
    matrix, rhs = make_inputs()
    with pytest.raises(ValueError, match="does not match compiled shape"):
        executor(matrix, rhs[:, :2])
