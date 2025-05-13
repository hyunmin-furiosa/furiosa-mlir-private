#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @kernel(%arg0: tensor<1x5x3xf32>, %arg1: tensor<1x3x6xf32>, %arg2: tensor<1x6x3xf32>) -> tensor<1x5x3xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x5x6xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x5x6xf32>) -> tensor<1x5x6xf32>
    %2 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%arg0, %arg1 : tensor<1x5x3xf32>, tensor<1x3x6xf32>) outs(%1 : tensor<1x5x6xf32>) -> tensor<1x5x6xf32>
    %3 = tensor.empty() : tensor<1x5x3xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x5x3xf32>) -> tensor<1x5x3xf32>
    %5 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%2, %arg2 : tensor<1x5x6xf32>, tensor<1x6x3xf32>) outs(%4 : tensor<1x5x3xf32>) -> tensor<1x5x3xf32>
    return %5 : tensor<1x5x3xf32>
  }
  func.func @main() {
    %0 = tensor.empty() : tensor<1x5x3xf32>
    %1 = tensor.empty() : tensor<1x3x6xf32>
    %2 = tensor.empty() : tensor<1x6x3xf32>
    %3 = call @kernel(%0, %1, %2) : (tensor<1x5x3xf32>, tensor<1x3x6xf32>, tensor<1x6x3xf32>) -> tensor<1x5x3xf32>
    return
  }
}
