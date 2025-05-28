// RUN: furiosa-mlir-opt -convert-linalg-to-furiosa

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @kernel(%arg0: tensor<64x64x64xf8E5M2>, %arg1: tensor<64x64x64xf8E5M2>) -> tensor<64x64x64xf8E5M2> attributes {target = #furiosa.target<npu 0 pe 0 : 0>} {
    %cst = arith.constant 0.000000e+00 : f8E5M2
    %0 = tensor.empty() : tensor<64x64x64xf8E5M2>
    %1 = scf.forall (%arg2) in (64) shared_outs(%arg3 = %0) -> (tensor<64x64x64xf8E5M2>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg2, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<64x64x64xf8E5M2> to tensor<1x64x64xf8E5M2>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<64x64x64xf8E5M2> to tensor<1x64x64xf8E5M2>
      %extracted_slice_1 = tensor.extract_slice %arg3[%arg2, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<64x64x64xf8E5M2> to tensor<1x64x64xf8E5M2>
      %2 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%extracted_slice, %extracted_slice_0 : tensor<1x64x64xf8E5M2>, tensor<1x64x64xf8E5M2>) outs(%extracted_slice_1 : tensor<1x64x64xf8E5M2>) -> tensor<1x64x64xf8E5M2>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %2 into %arg3[%arg2, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<1x64x64xf8E5M2> into tensor<64x64x64xf8E5M2>
      }
    } {mapping = [#furiosa.mapping]}
    return %1 : tensor<64x64x64xf8E5M2>
  }
  func.func @main() {
    %0 = tensor.empty() : tensor<64x64x64xf8E5M2>
    %1 = tensor.empty() : tensor<64x64x64xf8E5M2>
    %2 = call @kernel(%0, %1) {target = #furiosa.target<npu 0 pe 0 : 0>} : (tensor<64x64x64xf8E5M2>, tensor<64x64x64xf8E5M2>) -> tensor<64x64x64xf8E5M2>
    return
  }
}
