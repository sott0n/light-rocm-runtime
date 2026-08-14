import _lrrt_sparsewave
import torch
from lrrt_sparsewave import register_application


class SpmmApplication:
    def __init__(self, manifest, shape, handle):
        self.manifest = manifest
        self.shape = shape
        self._handle = handle

    @classmethod
    def load(cls, manifest, _document, configuration):
        configuration = dict(configuration)
        shape = configuration.pop("shape", None)
        if configuration:
            raise ValueError(f"unsupported SpMM configuration: {sorted(configuration)}")
        if shape is None:
            raise ValueError("the SpMM executor requires its compiled shape")
        if len(shape) != 3 or any(dimension <= 0 for dimension in shape):
            raise ValueError(
                "SpMM shape must contain positive rows, columns, and RHS columns"
            )
        handle = _lrrt_sparsewave.load_spmm(str(manifest), *shape)
        return cls(manifest, shape, handle)

    def __call__(self, matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        if matrix.layout != torch.sparse_csr:
            raise TypeError("matrix must use torch.sparse_csr layout")
        if matrix.device.type != "cpu" or rhs.device.type != "cpu":
            raise ValueError("SparseWave lrrt inputs must be CPU tensors")
        if matrix.dtype != torch.float32 or rhs.dtype != torch.float32:
            raise TypeError("SparseWave lrrt supports float32 values and RHS")
        if matrix.crow_indices().dtype != torch.int32:
            raise TypeError("CSR row offsets must use int32")
        if matrix.col_indices().dtype != torch.int32:
            raise TypeError("CSR column indices must use int32")
        if matrix.dim() != 2 or rhs.dim() != 2:
            raise ValueError("matrix and RHS must be rank two")
        if matrix.shape[1] != rhs.shape[0]:
            raise ValueError("SpMM inner dimensions do not match")
        shape = (matrix.shape[0], matrix.shape[1], rhs.shape[1])
        if shape != self.shape:
            raise ValueError(
                f"input shape {shape} does not match compiled shape {self.shape}"
            )

        row_offsets = matrix.crow_indices().contiguous()
        column_indices = matrix.col_indices().contiguous()
        values = matrix.values().contiguous()
        rhs = rhs.contiguous()
        output = torch.empty(
            (matrix.shape[0], rhs.shape[1]), dtype=torch.float32, device="cpu"
        )
        _lrrt_sparsewave.execute_spmm(
            self._handle,
            values.numel(),
            row_offsets.data_ptr(),
            column_indices.data_ptr(),
            values.data_ptr(),
            rhs.data_ptr(),
            output.data_ptr(),
        )
        return output


register_application("spmm", SpmmApplication.load)
