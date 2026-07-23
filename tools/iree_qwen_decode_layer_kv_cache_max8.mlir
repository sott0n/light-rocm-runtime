func.func @qwen_decode_layer_kv_cache_max8(
    %input: tensor<1x896xf32>,
    %old_key_cache: tensor<8x128xf32>,
    %old_value_cache: tensor<8x128xf32>,
    %position: tensor<1xi32>,
    %attention_norm_weight: tensor<896xf32>,
    %q_weight: tensor<896x896xf32>,
    %k_weight: tensor<896x128xf32>,
    %v_weight: tensor<896x128xf32>,
    %out_weight: tensor<896x896xf32>,
    %mlp_norm_weight: tensor<896xf32>,
    %gate_weight: tensor<896x4864xf32>,
    %up_weight: tensor<896x4864xf32>,
    %down_weight: tensor<4864x896xf32>) -> (tensor<8x128xf32>, tensor<8x128xf32>, tensor<1x896xf32>) {
  %zero = arith.constant 0.0 : f32
  %neg_inf = arith.constant -3.40282347E+38 : f32
  %head_scale = arith.constant 0.125 : f32
  %inv_hidden = arith.constant 0.0011160714285714285 : f32
  %eps = arith.constant 0.000001 : f32
  %c0 = arith.constant 0 : index
  %position_i32 = tensor.extract %position[%c0] : tensor<1xi32>
  %position_index = arith.index_cast %position_i32 : i32 to index
  %c448 = arith.constant 448 : index

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

  %key_cache = tensor.insert_slice %k into %old_key_cache[%position_index, 0] [1, 128] [1, 1]
      : tensor<1x128xf32> into tensor<8x128xf32>

  %value_cache = tensor.insert_slice %v into %old_value_cache[%position_index, 0] [1, 128] [1, 1]
      : tensor<1x128xf32> into tensor<8x128xf32>

  %q0 = tensor.extract_slice %q[0, 0] [1, 448] [1, 1]
      : tensor<1x896xf32> to tensor<1x448xf32>
  %q1 = tensor.extract_slice %q[0, 448] [1, 448] [1, 1]
      : tensor<1x896xf32> to tensor<1x448xf32>
  %key_cache0 = tensor.extract_slice %key_cache[0, 0] [8, 64] [1, 1]
      : tensor<8x128xf32> to tensor<8x64xf32>
  %key_cache1 = tensor.extract_slice %key_cache[0, 64] [8, 64] [1, 1]
      : tensor<8x128xf32> to tensor<8x64xf32>
  %value_cache0 = tensor.extract_slice %value_cache[0, 0] [8, 64] [1, 1]
      : tensor<8x128xf32> to tensor<8x64xf32>
  %value_cache1 = tensor.extract_slice %value_cache[0, 64] [8, 64] [1, 1]
      : tensor<8x128xf32> to tensor<8x64xf32>

  %q_heads0_empty = tensor.empty() : tensor<7x64xf32>
  %q_heads0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (0, d0 * 64 + d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%q0 : tensor<1x448xf32>)
      outs(%q_heads0_empty : tensor<7x64xf32>) {
    ^bb0(%value: f32, %out: f32):
      linalg.yield %value : f32
    } -> tensor<7x64xf32>

  %q_heads1_empty = tensor.empty() : tensor<7x64xf32>
  %q_heads1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (0, d0 * 64 + d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%q1 : tensor<1x448xf32>)
      outs(%q_heads1_empty : tensor<7x64xf32>) {
    ^bb0(%value: f32, %out: f32):
      linalg.yield %value : f32
    } -> tensor<7x64xf32>

  %key_head0_empty = tensor.empty() : tensor<64x8xf32>
  %key_head0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%key_cache0 : tensor<8x64xf32>)
      outs(%key_head0_empty : tensor<64x8xf32>) {
    ^bb0(%value: f32, %out: f32):
      linalg.yield %value : f32
    } -> tensor<64x8xf32>

  %key_head1_empty = tensor.empty() : tensor<64x8xf32>
  %key_head1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%key_cache1 : tensor<8x64xf32>)
      outs(%key_head1_empty : tensor<64x8xf32>) {
    ^bb0(%value: f32, %out: f32):
      linalg.yield %value : f32
    } -> tensor<64x8xf32>

  %score0_empty = tensor.empty() : tensor<7x8xf32>
  %score0_init = linalg.fill ins(%zero : f32)
      outs(%score0_empty : tensor<7x8xf32>) -> tensor<7x8xf32>
  %scores0_unscaled = linalg.matmul
      ins(%q_heads0, %key_head0 : tensor<7x64xf32>, tensor<64x8xf32>)
      outs(%score0_init : tensor<7x8xf32>) -> tensor<7x8xf32>

  %score1_empty = tensor.empty() : tensor<7x8xf32>
  %score1_init = linalg.fill ins(%zero : f32)
      outs(%score1_empty : tensor<7x8xf32>) -> tensor<7x8xf32>
  %scores1_unscaled = linalg.matmul
      ins(%q_heads1, %key_head1 : tensor<7x64xf32>, tensor<64x8xf32>)
      outs(%score1_init : tensor<7x8xf32>) -> tensor<7x8xf32>

  %scaled_scores0_empty = tensor.empty() : tensor<7x8xf32>
  %scaled_scores0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%scores0_unscaled, %position : tensor<7x8xf32>, tensor<1xi32>)
      outs(%scaled_scores0_empty : tensor<7x8xf32>) {
    ^bb0(%score: f32, %position_value: i32, %out: f32):
      %scaled = arith.mulf %score, %head_scale : f32
      %column = linalg.index 1 : index
      %visible_position = arith.index_cast %position_value : i32 to index
      %is_visible = arith.cmpi ule, %column, %visible_position : index
      %masked = arith.select %is_visible, %scaled, %neg_inf : f32
      linalg.yield %masked : f32
    } -> tensor<7x8xf32>

  %scaled_scores1_empty = tensor.empty() : tensor<7x8xf32>
  %scaled_scores1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%scores1_unscaled, %position : tensor<7x8xf32>, tensor<1xi32>)
      outs(%scaled_scores1_empty : tensor<7x8xf32>) {
    ^bb0(%score: f32, %position_value: i32, %out: f32):
      %scaled = arith.mulf %score, %head_scale : f32
      %column = linalg.index 1 : index
      %visible_position = arith.index_cast %position_value : i32 to index
      %is_visible = arith.cmpi ule, %column, %visible_position : index
      %masked = arith.select %is_visible, %scaled, %neg_inf : f32
      linalg.yield %masked : f32
    } -> tensor<7x8xf32>

  %score_max0_empty = tensor.empty() : tensor<7xf32>
  %score_max0_init = linalg.fill ins(%neg_inf : f32)
      outs(%score_max0_empty : tensor<7xf32>) -> tensor<7xf32>
  %score_max0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    }
      ins(%scaled_scores0 : tensor<7x8xf32>)
      outs(%score_max0_init : tensor<7xf32>) {
    ^bb0(%score: f32, %acc: f32):
      %max = arith.maximumf %score, %acc : f32
      linalg.yield %max : f32
    } -> tensor<7xf32>

  %score_max1_empty = tensor.empty() : tensor<7xf32>
  %score_max1_init = linalg.fill ins(%neg_inf : f32)
      outs(%score_max1_empty : tensor<7xf32>) -> tensor<7xf32>
  %score_max1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    }
      ins(%scaled_scores1 : tensor<7x8xf32>)
      outs(%score_max1_init : tensor<7xf32>) {
    ^bb0(%score: f32, %acc: f32):
      %max = arith.maximumf %score, %acc : f32
      linalg.yield %max : f32
    } -> tensor<7xf32>

  %exp0_empty = tensor.empty() : tensor<7x8xf32>
  %exp_scores0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%scaled_scores0, %score_max0 : tensor<7x8xf32>, tensor<7xf32>)
      outs(%exp0_empty : tensor<7x8xf32>) {
    ^bb0(%score: f32, %max: f32, %out: f32):
      %shifted = arith.subf %score, %max : f32
      %exp = math.exp %shifted : f32
      linalg.yield %exp : f32
    } -> tensor<7x8xf32>

  %exp1_empty = tensor.empty() : tensor<7x8xf32>
  %exp_scores1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%scaled_scores1, %score_max1 : tensor<7x8xf32>, tensor<7xf32>)
      outs(%exp1_empty : tensor<7x8xf32>) {
    ^bb0(%score: f32, %max: f32, %out: f32):
      %shifted = arith.subf %score, %max : f32
      %exp = math.exp %shifted : f32
      linalg.yield %exp : f32
    } -> tensor<7x8xf32>

  %sum0_empty = tensor.empty() : tensor<7xf32>
  %sum0_init = linalg.fill ins(%zero : f32)
      outs(%sum0_empty : tensor<7xf32>) -> tensor<7xf32>
  %row_sums0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    }
      ins(%exp_scores0 : tensor<7x8xf32>)
      outs(%sum0_init : tensor<7xf32>) {
    ^bb0(%exp_value: f32, %acc: f32):
      %sum = arith.addf %acc, %exp_value : f32
      linalg.yield %sum : f32
    } -> tensor<7xf32>

  %sum1_empty = tensor.empty() : tensor<7xf32>
  %sum1_init = linalg.fill ins(%zero : f32)
      outs(%sum1_empty : tensor<7xf32>) -> tensor<7xf32>
  %row_sums1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    }
      ins(%exp_scores1 : tensor<7x8xf32>)
      outs(%sum1_init : tensor<7xf32>) {
    ^bb0(%exp_value: f32, %acc: f32):
      %sum = arith.addf %acc, %exp_value : f32
      linalg.yield %sum : f32
    } -> tensor<7xf32>

  %prob0_empty = tensor.empty() : tensor<7x8xf32>
  %probabilities0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%exp_scores0, %row_sums0 : tensor<7x8xf32>, tensor<7xf32>)
      outs(%prob0_empty : tensor<7x8xf32>) {
    ^bb0(%exp_score: f32, %row_sum: f32, %out: f32):
      %probability = arith.divf %exp_score, %row_sum : f32
      linalg.yield %probability : f32
    } -> tensor<7x8xf32>

  %prob1_empty = tensor.empty() : tensor<7x8xf32>
  %probabilities1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%exp_scores1, %row_sums1 : tensor<7x8xf32>, tensor<7xf32>)
      outs(%prob1_empty : tensor<7x8xf32>) {
    ^bb0(%exp_score: f32, %row_sum: f32, %out: f32):
      %probability = arith.divf %exp_score, %row_sum : f32
      linalg.yield %probability : f32
    } -> tensor<7x8xf32>

  %value_head0_empty = tensor.empty() : tensor<8x64xf32>
  %value_head0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%value_cache0 : tensor<8x64xf32>)
      outs(%value_head0_empty : tensor<8x64xf32>) {
    ^bb0(%value: f32, %out: f32):
      linalg.yield %value : f32
    } -> tensor<8x64xf32>

  %value_head1_empty = tensor.empty() : tensor<8x64xf32>
  %value_head1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%value_cache1 : tensor<8x64xf32>)
      outs(%value_head1_empty : tensor<8x64xf32>) {
    ^bb0(%value: f32, %out: f32):
      linalg.yield %value : f32
    } -> tensor<8x64xf32>

  %context_heads0_empty = tensor.empty() : tensor<7x64xf32>
  %context_heads0_init = linalg.fill ins(%zero : f32)
      outs(%context_heads0_empty : tensor<7x64xf32>) -> tensor<7x64xf32>
  %context_heads0 = linalg.matmul
      ins(%probabilities0, %value_head0 : tensor<7x8xf32>, tensor<8x64xf32>)
      outs(%context_heads0_init : tensor<7x64xf32>) -> tensor<7x64xf32>

  %context_heads1_empty = tensor.empty() : tensor<7x64xf32>
  %context_heads1_init = linalg.fill ins(%zero : f32)
      outs(%context_heads1_empty : tensor<7x64xf32>) -> tensor<7x64xf32>
  %context_heads1 = linalg.matmul
      ins(%probabilities1, %value_head1 : tensor<7x8xf32>, tensor<8x64xf32>)
      outs(%context_heads1_init : tensor<7x64xf32>) -> tensor<7x64xf32>

  %context_empty = tensor.empty() : tensor<1x896xf32>
  %context_init = linalg.fill ins(%zero : f32)
      outs(%context_empty : tensor<1x896xf32>) -> tensor<1x896xf32>
  %context = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> ((d1 mod 448) floordiv 64, d1 mod 64)>,
        affine_map<(d0, d1) -> ((d1 mod 448) floordiv 64, d1 mod 64)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%context_heads0, %context_heads1 : tensor<7x64xf32>, tensor<7x64xf32>)
      outs(%context_init : tensor<1x896xf32>) {
    ^bb0(%value0: f32, %value1: f32, %out: f32):
      %column = linalg.index 1 : index
      %is_kv0 = arith.cmpi ult, %column, %c448 : index
      %value = arith.select %is_kv0, %value0, %value1 : f32
      linalg.yield %value : f32
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
      %is_nonnegative = arith.cmpf oge, %gate_value, %zero : f32
      %one = arith.constant 1.0 : f32
      %abs = math.absf %gate_value : f32
      %neg_abs = arith.negf %abs : f32
      %exp = math.exp %neg_abs : f32
      %denom = arith.addf %one, %exp : f32
      %sigmoid_pos = arith.divf %one, %denom : f32
      %sigmoid_neg = arith.divf %exp, %denom : f32
      %sigmoid = arith.select %is_nonnegative, %sigmoid_pos, %sigmoid_neg : f32
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
      : tensor<8x128xf32>, tensor<8x128xf32>, tensor<1x896xf32>
}
