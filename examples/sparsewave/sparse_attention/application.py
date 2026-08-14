import _lrrt_sparsewave
import torch
from lrrt_sparsewave import register_application


class SparseAttentionApplication:
    def __init__(self, manifest, shape, handle):
        self.manifest = manifest
        self.shape = shape
        self._handle = handle

    @classmethod
    def load(cls, manifest, document, configuration):
        if configuration:
            raise ValueError(
                "the SparseAttention executor reads its shape from the manifest"
            )
        specialization = document.get("sparsewave", {}).get("specialization")
        fields = (
            "output_rows",
            "key_value_rows",
            "head_dimension",
            "value_columns",
        )
        if not isinstance(specialization, dict):
            raise ValueError("SparseAttention manifest has no specialization")
        shape = tuple(specialization.get(field) for field in fields)
        if any(type(dimension) is not int or dimension <= 0 for dimension in shape):
            raise ValueError("SparseAttention manifest has an invalid specialization")
        handle = _lrrt_sparsewave.load_sparse_attention(str(manifest), *shape)
        return cls(manifest, shape, handle)

    def __call__(self, mask, query, key, value):
        if mask.layout != torch.sparse_csr:
            raise TypeError("mask must use torch.sparse_csr layout")
        dense_inputs = (query, key, value)
        if mask.device.type != "cpu" or any(
            tensor.device.type != "cpu" for tensor in dense_inputs
        ):
            raise ValueError("SparseWave lrrt inputs must be CPU tensors")
        if mask.dtype != torch.float32 or any(
            tensor.dtype != torch.float32 for tensor in dense_inputs
        ):
            raise TypeError("SparseWave lrrt SparseAttention supports float32")
        if mask.crow_indices().dtype != torch.int32:
            raise TypeError("CSR row offsets must use int32")
        if mask.col_indices().dtype != torch.int32:
            raise TypeError("CSR column indices must use int32")
        if mask.dim() != 2 or any(tensor.dim() != 2 for tensor in dense_inputs):
            raise ValueError("SparseAttention inputs must be rank two")

        output_rows, key_value_rows, head_dimension, value_columns = self.shape
        expected_shapes = (
            (output_rows, key_value_rows),
            (output_rows, head_dimension),
            (key_value_rows, head_dimension),
            (key_value_rows, value_columns),
        )
        actual_shapes = tuple(
            tuple(tensor.shape) for tensor in (mask, query, key, value)
        )
        if actual_shapes != expected_shapes:
            raise ValueError(
                f"input shapes {actual_shapes} do not match compiled shapes "
                f"{expected_shapes}"
            )

        row_offsets = mask.crow_indices().contiguous()
        column_indices = mask.col_indices().contiguous()
        mask_values = mask.values().contiguous()
        query = query.contiguous()
        transposed_key = key.transpose(0, 1).contiguous()
        value = value.contiguous()
        output = torch.empty(
            (output_rows, value_columns), dtype=torch.float32, device="cpu"
        )
        _lrrt_sparsewave.execute_sparse_attention(
            self._handle,
            mask_values.numel(),
            row_offsets.data_ptr(),
            column_indices.data_ptr(),
            mask_values.data_ptr(),
            query.data_ptr(),
            transposed_key.data_ptr(),
            value.data_ptr(),
            output.data_ptr(),
        )
        return output


register_application("sparse-attention", SparseAttentionApplication.load)
