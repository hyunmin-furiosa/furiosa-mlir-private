// RUN: furiosa-mlir-opt -transform-interpreter

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @kernel(%arg0: tensor<64x64x64xf8E5M2>, %arg1: tensor<64x64x64xf8E5M2>) -> tensor<64x64x64xf8E5M2> attributes {target = #furiosa.target<npu 0 pe 0 : 0>} {
    %0 = tensor.empty() : tensor<64x64x64xf8E5M2>
    %1 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%arg0, %arg1 : tensor<64x64x64xf8E5M2>, tensor<64x64x64xf8E5M2>) outs(%0 : tensor<64x64x64xf8E5M2>) -> tensor<64x64x64xf8E5M2>
    return %1 : tensor<64x64x64xf8E5M2>
  }
  func.func @main() {
    %0 = tensor.empty() : tensor<64x64x64xf8E5M2>
    %1 = tensor.empty() : tensor<64x64x64xf8E5M2>
    %2 = call @kernel(%0, %1) {target = #furiosa.target<npu 0 pe 0 : 0>} : (tensor<64x64x64xf8E5M2>, tensor<64x64x64xf8E5M2>) -> tensor<64x64x64xf8E5M2>
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:1 = transform.structured.tile_using_forall %0 num_threads [64] { mapping = [ #furiosa.mapping ] } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
