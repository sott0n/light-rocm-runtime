func.func @qwen_ffn_mini(
    %input: tensor<2x2xf32>,
    %w_gate: tensor<2x2xf32>,
    %w_up: tensor<2x2xf32>,
    %w_down: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant 0.0 : f32

  %gate_empty = tensor.empty() : tensor<2x2xf32>
  %gate_init = linalg.fill
      ins(%zero : f32)
      outs(%gate_empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  %gate = linalg.matmul
      ins(%input, %w_gate : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%gate_init : tensor<2x2xf32>) -> tensor<2x2xf32>

  %up_empty = tensor.empty() : tensor<2x2xf32>
  %up_init = linalg.fill
      ins(%zero : f32)
      outs(%up_empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  %up = linalg.matmul
      ins(%input, %w_up : tensor<2x2xf32>, tensor<2x2xf32>)
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

  %out_empty = tensor.empty() : tensor<2x2xf32>
  %out_init = linalg.fill
      ins(%zero : f32)
      outs(%out_empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  %result = linalg.matmul
      ins(%hidden, %w_down : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%out_init : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}
