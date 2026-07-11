func.func @attention_score(
    %query: tensor<2x2xf32>,
    %key_transposed: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x2xf32>
  %init = linalg.fill
      ins(%zero : f32)
      outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  %scores = linalg.matmul
      ins(%query, %key_transposed : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %scores : tensor<2x2xf32>
}
