func.func @mini_decoder_layer_kv_cache(
    %input: tensor<2x2xf32>,
    %query: tensor<2x2xf32>,
    %key_cache_transposed: tensor<2x3xf32>,
    %value_cache: tensor<3x2xf32>,
    %w_gate: tensor<2x2xf32>,
    %w_up: tensor<2x2xf32>,
    %w_down: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant 0.0 : f32
  %score_empty = tensor.empty() : tensor<2x3xf32>
  %score_init = linalg.fill
      ins(%zero : f32)
      outs(%score_empty : tensor<2x3xf32>) -> tensor<2x3xf32>
  %scores = linalg.matmul
      ins(%query, %key_cache_transposed : tensor<2x2xf32>, tensor<2x3xf32>)
      outs(%score_init : tensor<2x3xf32>) -> tensor<2x3xf32>

  %exp_empty = tensor.empty() : tensor<2x3xf32>
  %exp_scores = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%scores : tensor<2x3xf32>)
      outs(%exp_empty : tensor<2x3xf32>) {
    ^bb0(%score: f32, %out: f32):
      %exp = math.exp %score : f32
      linalg.yield %exp : f32
    } -> tensor<2x3xf32>

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
      ins(%exp_scores : tensor<2x3xf32>)
      outs(%sum_init : tensor<2xf32>) {
    ^bb0(%exp_value: f32, %acc: f32):
      %sum = arith.addf %acc, %exp_value : f32
      linalg.yield %sum : f32
    } -> tensor<2xf32>

  %prob_empty = tensor.empty() : tensor<2x3xf32>
  %probabilities = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%exp_scores, %row_sums : tensor<2x3xf32>, tensor<2xf32>)
      outs(%prob_empty : tensor<2x3xf32>) {
    ^bb0(%exp_score: f32, %row_sum: f32, %out: f32):
      %probability = arith.divf %exp_score, %row_sum : f32
      linalg.yield %probability : f32
    } -> tensor<2x3xf32>

  %context_empty = tensor.empty() : tensor<2x2xf32>
  %context_init = linalg.fill
      ins(%zero : f32)
      outs(%context_empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  %context = linalg.matmul
      ins(%probabilities, %value_cache : tensor<2x3xf32>, tensor<3x2xf32>)
      outs(%context_init : tensor<2x2xf32>) -> tensor<2x2xf32>

  %attention_out_empty = tensor.empty() : tensor<2x2xf32>
  %attention_out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%input, %context : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%attention_out_empty : tensor<2x2xf32>) {
    ^bb0(%residual: f32, %attention_value: f32, %out: f32):
      %sum = arith.addf %residual, %attention_value : f32
      linalg.yield %sum : f32
    } -> tensor<2x2xf32>

  %gate_empty = tensor.empty() : tensor<2x2xf32>
  %gate_init = linalg.fill
      ins(%zero : f32)
      outs(%gate_empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  %gate = linalg.matmul
      ins(%attention_out, %w_gate : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%gate_init : tensor<2x2xf32>) -> tensor<2x2xf32>

  %up_empty = tensor.empty() : tensor<2x2xf32>
  %up_init = linalg.fill
      ins(%zero : f32)
      outs(%up_empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  %up = linalg.matmul
      ins(%attention_out, %w_up : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%up_init : tensor<2x2xf32>) -> tensor<2x2xf32>

  %hidden_empty = tensor.empty() : tensor<2x2xf32>
  %hidden = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%gate, %up : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%hidden_empty : tensor<2x2xf32>) {
    ^bb0(%gate_value: f32, %up_value: f32, %out: f32):
      %activated = arith.maximumf %gate_value, %zero : f32
      %gated = arith.mulf %activated, %up_value : f32
      linalg.yield %gated : f32
    } -> tensor<2x2xf32>

  %ffn_empty = tensor.empty() : tensor<2x2xf32>
  %ffn_init = linalg.fill
      ins(%zero : f32)
      outs(%ffn_empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  %ffn_out = linalg.matmul
      ins(%hidden, %w_down : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%ffn_init : tensor<2x2xf32>) -> tensor<2x2xf32>

  %out_empty = tensor.empty() : tensor<2x2xf32>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%attention_out, %ffn_out : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%out_empty : tensor<2x2xf32>) {
    ^bb0(%residual: f32, %ffn_value: f32, %out: f32):
      %sum = arith.addf %residual, %ffn_value : f32
      linalg.yield %sum : f32
    } -> tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}
