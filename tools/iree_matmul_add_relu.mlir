func.func @matmul_add_relu(
    %lhs: tensor<2x2xf32>,
    %rhs: tensor<2x2xf32>,
    %bias: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant 0.0 : f32
  %matmul_empty = tensor.empty() : tensor<2x2xf32>
  %matmul_init = linalg.fill
      ins(%zero : f32)
      outs(%matmul_empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  %product = linalg.matmul
      ins(%lhs, %rhs : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%matmul_init : tensor<2x2xf32>) -> tensor<2x2xf32>

  %result_empty = tensor.empty() : tensor<2x2xf32>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%product, %bias : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%result_empty : tensor<2x2xf32>) {
    ^bb0(%value: f32, %bias_value: f32, %out: f32):
      %sum = arith.addf %value, %bias_value : f32
      %relu = arith.maximumf %sum, %zero : f32
      linalg.yield %relu : f32
    } -> tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}
