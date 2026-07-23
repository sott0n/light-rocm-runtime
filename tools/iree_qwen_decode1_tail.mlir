func.func @qwen_decode1_tail(
    %hidden: tensor<1x896xf32>,
    %final_norm_weight: tensor<896xf32>,
    %lm_head_weight: tensor<896x151936xf32>) -> tensor<1x151936xf32> {
  %zero = arith.constant 0.0 : f32
  %inv_hidden = arith.constant 0.0011160714285714285 : f32
  %eps = arith.constant 0.000001 : f32

  %sum_empty = tensor.empty() : tensor<1xf32>
  %sum_init = linalg.fill
      ins(%zero : f32)
      outs(%sum_empty : tensor<1xf32>) -> tensor<1xf32>
  %sum_squares = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    }
      ins(%hidden : tensor<1x896xf32>)
      outs(%sum_init : tensor<1xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %square = arith.mulf %value, %value : f32
      %sum = arith.addf %acc, %square : f32
      linalg.yield %sum : f32
    } -> tensor<1xf32>

  %norm_empty = tensor.empty() : tensor<1x896xf32>
  %norm = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%hidden, %sum_squares, %final_norm_weight :
          tensor<1x896xf32>, tensor<1xf32>, tensor<896xf32>)
      outs(%norm_empty : tensor<1x896xf32>) {
    ^bb0(%value: f32, %row_sum: f32, %weight: f32, %out: f32):
      %mean_square = arith.mulf %row_sum, %inv_hidden : f32
      %variance = arith.addf %mean_square, %eps : f32
      %rms = math.rsqrt %variance : f32
      %normalized = arith.mulf %value, %rms : f32
      %scaled = arith.mulf %normalized, %weight : f32
      linalg.yield %scaled : f32
    } -> tensor<1x896xf32>

  %logits_empty = tensor.empty() : tensor<1x151936xf32>
  %logits_init = linalg.fill
      ins(%zero : f32)
      outs(%logits_empty : tensor<1x151936xf32>) -> tensor<1x151936xf32>
  %logits = linalg.matmul
      ins(%norm, %lm_head_weight : tensor<1x896xf32>, tensor<896x151936xf32>)
      outs(%logits_init : tensor<1x151936xf32>) -> tensor<1x151936xf32>

  return %logits : tensor<1x151936xf32>
}
