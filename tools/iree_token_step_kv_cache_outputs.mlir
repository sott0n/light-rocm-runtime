func.func @token_step_kv_cache_outputs(
    %query: tensor<2x2xf32>,
    %old_key_cache_transposed: tensor<2x3xf32>,
    %new_key: tensor<2xf32>,
    %old_value_cache: tensor<3x2xf32>,
    %new_value: tensor<2xf32>,
    %cos: tensor<2xf32>,
    %sin: tensor<2xf32>) -> (tensor<2x3xf32>, tensor<3x2xf32>, tensor<2x2xf32>) {
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index

  %query_rope_empty = tensor.empty() : tensor<2x2xf32>
  %query_rope = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, 0)>,
        affine_map<(d0, d1) -> (d0, 1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%query, %query, %cos, %sin :
          tensor<2x2xf32>, tensor<2x2xf32>, tensor<2xf32>, tensor<2xf32>)
      outs(%query_rope_empty : tensor<2x2xf32>) {
    ^bb0(%x0: f32, %x1: f32, %cos_value: f32, %sin_value: f32, %out: f32):
      %column = linalg.index 1 : index
      %is_first = arith.cmpi eq, %column, %c0 : index
      %x0_cos = arith.mulf %x0, %cos_value : f32
      %x1_sin = arith.mulf %x1, %sin_value : f32
      %rot0 = arith.subf %x0_cos, %x1_sin : f32
      %x0_sin = arith.mulf %x0, %sin_value : f32
      %x1_cos = arith.mulf %x1, %cos_value : f32
      %rot1 = arith.addf %x0_sin, %x1_cos : f32
      %rotated = arith.select %is_first, %rot0, %rot1 : f32
      linalg.yield %rotated : f32
    } -> tensor<2x2xf32>

  %new_key_rope_empty = tensor.empty() : tensor<2xf32>
  %new_key_rope = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (0)>,
        affine_map<(d0) -> (1)>,
        affine_map<(d0) -> (0)>,
        affine_map<(d0) -> (0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
      ins(%new_key, %new_key, %cos, %sin :
          tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
      outs(%new_key_rope_empty : tensor<2xf32>) {
    ^bb0(%x0: f32, %x1: f32, %cos_value: f32, %sin_value: f32, %out: f32):
      %column = linalg.index 0 : index
      %is_first = arith.cmpi eq, %column, %c0 : index
      %x0_cos = arith.mulf %x0, %cos_value : f32
      %x1_sin = arith.mulf %x1, %sin_value : f32
      %rot0 = arith.subf %x0_cos, %x1_sin : f32
      %x0_sin = arith.mulf %x0, %sin_value : f32
      %x1_cos = arith.mulf %x1, %cos_value : f32
      %rot1 = arith.addf %x0_sin, %x1_cos : f32
      %rotated = arith.select %is_first, %rot0, %rot1 : f32
      linalg.yield %rotated : f32
    } -> tensor<2xf32>

  %key_cache_empty = tensor.empty() : tensor<2x3xf32>
  %key_cache_transposed = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%old_key_cache_transposed, %new_key_rope :
          tensor<2x3xf32>, tensor<2xf32>)
      outs(%key_cache_empty : tensor<2x3xf32>) {
    ^bb0(%old_value: f32, %new_key_value: f32, %out: f32):
      %column = linalg.index 1 : index
      %is_update_column = arith.cmpi eq, %column, %c2 : index
      %value = arith.select %is_update_column, %new_key_value, %old_value : f32
      linalg.yield %value : f32
    } -> tensor<2x3xf32>

  %value_cache_empty = tensor.empty() : tensor<3x2xf32>
  %value_cache = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%old_value_cache, %new_value : tensor<3x2xf32>, tensor<2xf32>)
      outs(%value_cache_empty : tensor<3x2xf32>) {
    ^bb0(%old_value: f32, %new_value_element: f32, %out: f32):
      %row = linalg.index 0 : index
      %is_update_row = arith.cmpi eq, %row, %c2 : index
      %value = arith.select %is_update_row, %new_value_element, %old_value : f32
      linalg.yield %value : f32
    } -> tensor<3x2xf32>

  %score_empty = tensor.empty() : tensor<2x3xf32>
  %score_init = linalg.fill
      ins(%zero : f32)
      outs(%score_empty : tensor<2x3xf32>) -> tensor<2x3xf32>
  %scores = linalg.matmul
      ins(%query_rope, %key_cache_transposed :
          tensor<2x2xf32>, tensor<2x3xf32>)
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
  return %key_cache_transposed, %value_cache, %context
      : tensor<2x3xf32>, tensor<3x2xf32>, tensor<2x2xf32>
}
