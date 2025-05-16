// RUN: furiosa-mlir-opt -transform-interpreter

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @kernel(%arg0: tensor<64x64x64xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<128x64xf32>) -> tensor<64x64x64xf32> attributes { target = #furiosa.target<npu 0 pe 0:0> } {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x64x128xf32>
    %2 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%arg0, %arg1 : tensor<64x64x64xf32>, tensor<64x128xf32>) outs(%0 : tensor<64x64x128xf32>) -> tensor<64x64x128xf32>
    %3 = tensor.empty() : tensor<64x64x64xf32>
    %5 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%2, %arg2 : tensor<64x64x128xf32>, tensor<128x64xf32>) outs(%3 : tensor<64x64x64xf32>) -> tensor<64x64x64xf32>
    return %5 : tensor<64x64x64xf32>
  }
  func.func @main() {
    %0 = tensor.empty() : tensor<64x64x64xf32>
    %1 = tensor.empty() : tensor<64x128xf32>
    %2 = tensor.empty() : tensor<128x64xf32>
    %3 = call @kernel(%0, %1, %2) { target = #furiosa.target<npu 0 pe 0:0> } : (tensor<64x64x64xf32>, tensor<64x128xf32>, tensor<128x64xf32>) -> tensor<64x64x64xf32>
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:1 = transform.structured.tile_using_forall %0 num_threads [32, 0, 0, 2] { mapping = [ #furiosa.mapping, #furiosa.mapping ] } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
