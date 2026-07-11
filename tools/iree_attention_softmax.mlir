func.func @attention_softmax(
    %query: tensor<2x2xf32>,
    %key_transposed: tensor<2x2xf32>,
    %value: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant 0.0 : f32
  %score_empty = tensor.empty() : tensor<2x2xf32>
  %score_init = linalg.fill
      ins(%zero : f32)
      outs(%score_empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  %scores = linalg.matmul
      ins(%query, %key_transposed : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%score_init : tensor<2x2xf32>) -> tensor<2x2xf32>

  %exp_empty = tensor.empty() : tensor<2x2xf32>
  %exp_scores = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%scores : tensor<2x2xf32>)
      outs(%exp_empty : tensor<2x2xf32>) {
    ^bb0(%score: f32, %out: f32):
      %exp = math.exp %score : f32
      linalg.yield %exp : f32
    } -> tensor<2x2xf32>

  %sum_empty = tensor.empty() : tensor<2xf32>
  %sum_init = linalg.fill
      ins(%zero : f32)
      outs(%sum_empty : tensor<2xf32>) -> tensor<2xf32>
  %row_sums = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    }
      ins(%exp_scores : tensor<2x2xf32>)
      outs(%sum_init : tensor<2xf32>) {
    ^bb0(%exp_value: f32, %acc: f32):
      %sum = arith.addf %acc, %exp_value : f32
      linalg.yield %sum : f32
    } -> tensor<2xf32>

  %prob_empty = tensor.empty() : tensor<2x2xf32>
  %probabilities = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%exp_scores, %row_sums : tensor<2x2xf32>, tensor<2xf32>)
      outs(%prob_empty : tensor<2x2xf32>) {
    ^bb0(%exp_score: f32, %row_sum: f32, %out: f32):
      %probability = arith.divf %exp_score, %row_sum : f32
      linalg.yield %probability : f32
    } -> tensor<2x2xf32>

  %context_empty = tensor.empty() : tensor<2x2xf32>
  %context_init = linalg.fill
      ins(%zero : f32)
      outs(%context_empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  %context = linalg.matmul
      ins(%probabilities, %value : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%context_init : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %context : tensor<2x2xf32>
}
