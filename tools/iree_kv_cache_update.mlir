func.func @kv_cache_update(
    %old_cache: tensor<3x2xf32>,
    %new_token: tensor<2xf32>) -> tensor<3x2xf32> {
  %c2 = arith.constant 2 : index
  %empty = tensor.empty() : tensor<3x2xf32>
  %updated = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%old_cache, %new_token : tensor<3x2xf32>, tensor<2xf32>)
      outs(%empty : tensor<3x2xf32>) {
    ^bb0(%old_value: f32, %new_value: f32, %out: f32):
      %row = linalg.index 0 : index
      %is_update_row = arith.cmpi eq, %row, %c2 : index
      %value = arith.select %is_update_row, %new_value, %old_value : f32
      linalg.yield %value : f32
    } -> tensor<3x2xf32>
  return %updated : tensor<3x2xf32>
}
