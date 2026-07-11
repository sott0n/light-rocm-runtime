func.func @rope_apply(
    %input: tensor<2x2xf32>,
    %cos: tensor<2xf32>,
    %sin: tensor<2xf32>) -> tensor<2x2xf32> {
  %c0 = arith.constant 0 : index
  %out_empty = tensor.empty() : tensor<2x2xf32>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, 0)>,
        affine_map<(d0, d1) -> (d0, 1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%input, %input, %cos, %sin :
          tensor<2x2xf32>, tensor<2x2xf32>, tensor<2xf32>, tensor<2xf32>)
      outs(%out_empty : tensor<2x2xf32>) {
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
  return %result : tensor<2x2xf32>
}
