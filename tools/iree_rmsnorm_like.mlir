func.func @rmsnorm_like(
    %input: tensor<2x2xf32>,
    %weight: tensor<2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant 0.0 : f32
  %half = arith.constant 0.5 : f32
  %sum_empty = tensor.empty() : tensor<2xf32>
  %sum_init = linalg.fill
      ins(%zero : f32)
      outs(%sum_empty : tensor<2xf32>) -> tensor<2xf32>
  %sum_squares = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    }
      ins(%input : tensor<2x2xf32>)
      outs(%sum_init : tensor<2xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %square = arith.mulf %value, %value : f32
      %sum = arith.addf %acc, %square : f32
      linalg.yield %sum : f32
    } -> tensor<2xf32>

  %out_empty = tensor.empty() : tensor<2x2xf32>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%input, %sum_squares, %weight :
          tensor<2x2xf32>, tensor<2xf32>, tensor<2xf32>)
      outs(%out_empty : tensor<2x2xf32>) {
    ^bb0(%value: f32, %row_sum: f32, %scale: f32, %out: f32):
      %mean_square = arith.mulf %row_sum, %half : f32
      %rms = math.rsqrt %mean_square : f32
      %normalized = arith.mulf %value, %rms : f32
      %scaled = arith.mulf %normalized, %scale : f32
      linalg.yield %scaled : f32
    } -> tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}
