func.func @mixed_matmuls(
    %lhs: tensor<2x2xf32>,
    %rhs0: tensor<2x2xf32>,
    %rhs1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %zero = arith.constant 0.0 : f32
  %empty0 = tensor.empty() : tensor<2x2xf32>
  %init0 = linalg.fill
      ins(%zero : f32)
      outs(%empty0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %product0 = linalg.matmul
      ins(%lhs, %rhs0 : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%init0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %empty1 = tensor.empty() : tensor<2x3xf32>
  %init1 = linalg.fill
      ins(%zero : f32)
      outs(%empty1 : tensor<2x3xf32>) -> tensor<2x3xf32>
  %result = linalg.matmul
      ins(%product0, %rhs1 : tensor<2x2xf32>, tensor<2x3xf32>)
      outs(%init1 : tensor<2x3xf32>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}
