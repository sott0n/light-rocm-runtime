func.func @qwen_decode2_layer_kv_cache(
    %input: tensor<1x896xf32>,
    %old_key_cache: tensor<1x128xf32>,
    %old_value_cache: tensor<1x128xf32>,
    %attention_norm_weight: tensor<896xf32>,
    %q_weight: tensor<896x896xf32>,
    %k_weight: tensor<896x128xf32>,
    %v_weight: tensor<896x128xf32>,
    %out_weight: tensor<896x896xf32>,
    %mlp_norm_weight: tensor<896xf32>,
    %gate_weight: tensor<896x4864xf32>,
    %up_weight: tensor<896x4864xf32>,
    %down_weight: tensor<4864x896xf32>) -> (tensor<2x128xf32>, tensor<2x128xf32>, tensor<1x896xf32>) {
  %zero = arith.constant 0.0 : f32
  %half = arith.constant 0.5 : f32
  %inv_hidden = arith.constant 0.0011160714285714285 : f32
  %eps = arith.constant 0.000001 : f32
  %c0 = arith.constant 0 : index

  %attn_norm_sum_empty = tensor.empty() : tensor<1xf32>
  %attn_norm_sum_init = linalg.fill
      ins(%zero : f32)
      outs(%attn_norm_sum_empty : tensor<1xf32>) -> tensor<1xf32>
  %attn_norm_sum = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    }
      ins(%input : tensor<1x896xf32>)
      outs(%attn_norm_sum_init : tensor<1xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %square = arith.mulf %value, %value : f32
      %sum = arith.addf %acc, %square : f32
      linalg.yield %sum : f32
    } -> tensor<1xf32>

  %attn_norm_empty = tensor.empty() : tensor<1x896xf32>
  %attn_norm = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%input, %attn_norm_sum, %attention_norm_weight :
          tensor<1x896xf32>, tensor<1xf32>, tensor<896xf32>)
      outs(%attn_norm_empty : tensor<1x896xf32>) {
    ^bb0(%value: f32, %row_sum: f32, %weight: f32, %out: f32):
      %mean_square = arith.mulf %row_sum, %inv_hidden : f32
      %variance = arith.addf %mean_square, %eps : f32
      %rms = math.rsqrt %variance : f32
      %normalized = arith.mulf %value, %rms : f32
      %scaled = arith.mulf %normalized, %weight : f32
      linalg.yield %scaled : f32
    } -> tensor<1x896xf32>

  %q_empty = tensor.empty() : tensor<1x896xf32>
  %q_init = linalg.fill ins(%zero : f32)
      outs(%q_empty : tensor<1x896xf32>) -> tensor<1x896xf32>
  %q = linalg.matmul
      ins(%attn_norm, %q_weight : tensor<1x896xf32>, tensor<896x896xf32>)
      outs(%q_init : tensor<1x896xf32>) -> tensor<1x896xf32>

  %k_empty = tensor.empty() : tensor<1x128xf32>
  %k_init = linalg.fill ins(%zero : f32)
      outs(%k_empty : tensor<1x128xf32>) -> tensor<1x128xf32>
  %k = linalg.matmul
      ins(%attn_norm, %k_weight : tensor<1x896xf32>, tensor<896x128xf32>)
      outs(%k_init : tensor<1x128xf32>) -> tensor<1x128xf32>

  %v_empty = tensor.empty() : tensor<1x128xf32>
  %v_init = linalg.fill ins(%zero : f32)
      outs(%v_empty : tensor<1x128xf32>) -> tensor<1x128xf32>
  %v = linalg.matmul
      ins(%attn_norm, %v_weight : tensor<1x896xf32>, tensor<896x128xf32>)
      outs(%v_init : tensor<1x128xf32>) -> tensor<1x128xf32>

  %key_cache_empty = tensor.empty() : tensor<2x128xf32>
  %key_cache = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (0, d1)>,
        affine_map<(d0, d1) -> (0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%old_key_cache, %k : tensor<1x128xf32>, tensor<1x128xf32>)
      outs(%key_cache_empty : tensor<2x128xf32>) {
    ^bb0(%old_value: f32, %new_value: f32, %out: f32):
      %row = linalg.index 0 : index
      %is_old = arith.cmpi eq, %row, %c0 : index
      %value = arith.select %is_old, %old_value, %new_value : f32
      linalg.yield %value : f32
    } -> tensor<2x128xf32>

  %value_cache_empty = tensor.empty() : tensor<2x128xf32>
  %value_cache = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (0, d1)>,
        affine_map<(d0, d1) -> (0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%old_value_cache, %v : tensor<1x128xf32>, tensor<1x128xf32>)
      outs(%value_cache_empty : tensor<2x128xf32>) {
    ^bb0(%old_value: f32, %new_value: f32, %out: f32):
      %row = linalg.index 0 : index
      %is_old = arith.cmpi eq, %row, %c0 : index
      %value = arith.select %is_old, %old_value, %new_value : f32
      linalg.yield %value : f32
    } -> tensor<2x128xf32>

  %context_empty = tensor.empty() : tensor<1x896xf32>
  %context = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (0, (d1 floordiv 448) * 64 + (d1 mod 64))>,
        affine_map<(d0, d1) -> (1, (d1 floordiv 448) * 64 + (d1 mod 64))>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%value_cache, %value_cache, %q :
          tensor<2x128xf32>, tensor<2x128xf32>, tensor<1x896xf32>)
      outs(%context_empty : tensor<1x896xf32>) {
    ^bb0(%old_value: f32, %new_value: f32, %query_value: f32, %out: f32):
      %sum = arith.addf %old_value, %new_value : f32
      %avg = arith.mulf %sum, %half : f32
      linalg.yield %avg : f32
    } -> tensor<1x896xf32>

  %attn_out_empty = tensor.empty() : tensor<1x896xf32>
  %attn_out_init = linalg.fill ins(%zero : f32)
      outs(%attn_out_empty : tensor<1x896xf32>) -> tensor<1x896xf32>
  %attn_projected = linalg.matmul
      ins(%context, %out_weight : tensor<1x896xf32>, tensor<896x896xf32>)
      outs(%attn_out_init : tensor<1x896xf32>) -> tensor<1x896xf32>

  %attn_residual_empty = tensor.empty() : tensor<1x896xf32>
  %attn_residual = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%input, %attn_projected : tensor<1x896xf32>, tensor<1x896xf32>)
      outs(%attn_residual_empty : tensor<1x896xf32>) {
    ^bb0(%residual: f32, %value: f32, %out: f32):
      %sum = arith.addf %residual, %value : f32
      linalg.yield %sum : f32
    } -> tensor<1x896xf32>

  %mlp_norm_sum_empty = tensor.empty() : tensor<1xf32>
  %mlp_norm_sum_init = linalg.fill
      ins(%zero : f32)
      outs(%mlp_norm_sum_empty : tensor<1xf32>) -> tensor<1xf32>
  %mlp_norm_sum = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    }
      ins(%attn_residual : tensor<1x896xf32>)
      outs(%mlp_norm_sum_init : tensor<1xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %square = arith.mulf %value, %value : f32
      %sum = arith.addf %acc, %square : f32
      linalg.yield %sum : f32
    } -> tensor<1xf32>

  %mlp_norm_empty = tensor.empty() : tensor<1x896xf32>
  %mlp_norm = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%attn_residual, %mlp_norm_sum, %mlp_norm_weight :
          tensor<1x896xf32>, tensor<1xf32>, tensor<896xf32>)
      outs(%mlp_norm_empty : tensor<1x896xf32>) {
    ^bb0(%value: f32, %row_sum: f32, %weight: f32, %out: f32):
      %mean_square = arith.mulf %row_sum, %inv_hidden : f32
      %variance = arith.addf %mean_square, %eps : f32
      %rms = math.rsqrt %variance : f32
      %normalized = arith.mulf %value, %rms : f32
      %scaled = arith.mulf %normalized, %weight : f32
      linalg.yield %scaled : f32
    } -> tensor<1x896xf32>

  %gate_empty = tensor.empty() : tensor<1x4864xf32>
  %gate_init = linalg.fill ins(%zero : f32)
      outs(%gate_empty : tensor<1x4864xf32>) -> tensor<1x4864xf32>
  %gate = linalg.matmul
      ins(%mlp_norm, %gate_weight : tensor<1x896xf32>, tensor<896x4864xf32>)
      outs(%gate_init : tensor<1x4864xf32>) -> tensor<1x4864xf32>

  %up_empty = tensor.empty() : tensor<1x4864xf32>
  %up_init = linalg.fill ins(%zero : f32)
      outs(%up_empty : tensor<1x4864xf32>) -> tensor<1x4864xf32>
  %up = linalg.matmul
      ins(%mlp_norm, %up_weight : tensor<1x896xf32>, tensor<896x4864xf32>)
      outs(%up_init : tensor<1x4864xf32>) -> tensor<1x4864xf32>

  %ffn_hidden_empty = tensor.empty() : tensor<1x4864xf32>
  %ffn_hidden = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%gate, %up : tensor<1x4864xf32>, tensor<1x4864xf32>)
      outs(%ffn_hidden_empty : tensor<1x4864xf32>) {
    ^bb0(%gate_value: f32, %up_value: f32, %out: f32):
      %neg = arith.negf %gate_value : f32
      %exp = math.exp %neg : f32
      %one = arith.constant 1.0 : f32
      %denom = arith.addf %one, %exp : f32
      %sigmoid = arith.divf %one, %denom : f32
      %silu = arith.mulf %gate_value, %sigmoid : f32
      %gated = arith.mulf %silu, %up_value : f32
      linalg.yield %gated : f32
    } -> tensor<1x4864xf32>

  %ffn_out_empty = tensor.empty() : tensor<1x896xf32>
  %ffn_out_init = linalg.fill ins(%zero : f32)
      outs(%ffn_out_empty : tensor<1x896xf32>) -> tensor<1x896xf32>
  %ffn_out = linalg.matmul
      ins(%ffn_hidden, %down_weight : tensor<1x4864xf32>, tensor<4864x896xf32>)
      outs(%ffn_out_init : tensor<1x896xf32>) -> tensor<1x896xf32>

  %result_empty = tensor.empty() : tensor<1x896xf32>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%attn_residual, %ffn_out : tensor<1x896xf32>, tensor<1x896xf32>)
      outs(%result_empty : tensor<1x896xf32>) {
    ^bb0(%residual: f32, %value: f32, %out: f32):
      %sum = arith.addf %residual, %value : f32
      linalg.yield %sum : f32
    } -> tensor<1x896xf32>

  return %key_cache, %value_cache, %result
      : tensor<2x128xf32>, tensor<2x128xf32>, tensor<1x896xf32>
}
