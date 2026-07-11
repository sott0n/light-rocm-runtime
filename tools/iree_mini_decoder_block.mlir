func.func @mini_decoder_block(
    %input: tensor<2x2xf32>,
    %norm_scale: tensor<2x2xf32>,
    %norm_bias: tensor<2x2xf32>,
    %w_up: tensor<2x2xf32>,
    %w_down: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant 0.0 : f32

  %norm_empty = tensor.empty() : tensor<2x2xf32>
  %norm = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%input, %norm_scale, %norm_bias :
          tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%norm_empty : tensor<2x2xf32>) {
    ^bb0(%value: f32, %scale: f32, %bias: f32, %out: f32):
      %scaled = arith.mulf %value, %scale : f32
      %biased = arith.addf %scaled, %bias : f32
      linalg.yield %biased : f32
    } -> tensor<2x2xf32>

  %up_empty = tensor.empty() : tensor<2x2xf32>
  %up_init = linalg.fill
      ins(%zero : f32)
      outs(%up_empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  %up = linalg.matmul
      ins(%norm, %w_up : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%up_init : tensor<2x2xf32>) -> tensor<2x2xf32>

  %act_empty = tensor.empty() : tensor<2x2xf32>
  %act = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%up : tensor<2x2xf32>)
      outs(%act_empty : tensor<2x2xf32>) {
    ^bb0(%value: f32, %out: f32):
      %relu = arith.maximumf %value, %zero : f32
      linalg.yield %relu : f32
    } -> tensor<2x2xf32>

  %down_empty = tensor.empty() : tensor<2x2xf32>
  %down_init = linalg.fill
      ins(%zero : f32)
      outs(%down_empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  %result = linalg.matmul
      ins(%act, %w_down : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%down_init : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}
