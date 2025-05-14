#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  module {
    func.func @kernel(%arg0: tensor<64x64x64xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<128x64xf32>) -> tensor<64x64x64xf32> attributes {target = #furiosa.target<npu 0 pe 0 : 0>} {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = tensor.empty() : tensor<64x64x128xf32>
      %1 = scf.forall (%arg3) in (64) shared_outs(%arg4 = %0) -> (tensor<64x64x128xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg3, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<64x64x64xf32> to tensor<1x64x64xf32>
        %extracted_slice_0 = tensor.extract_slice %arg1[0, 0] [64, 128] [1, 1] : tensor<64x128xf32> to tensor<64x128xf32>
        %extracted_slice_1 = tensor.extract_slice %arg4[%arg3, 0, 0] [1, 64, 128] [1, 1, 1] : tensor<64x64x128xf32> to tensor<1x64x128xf32>
        %4 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%extracted_slice, %extracted_slice_0 : tensor<1x64x64xf32>, tensor<64x128xf32>) outs(%extracted_slice_1 : tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %4 into %arg4[%arg3, 0, 0] [1, 64, 128] [1, 1, 1] : tensor<1x64x128xf32> into tensor<64x64x128xf32>
        }
      } {mapping = [#furiosa.mapping]}
      %2 = tensor.empty() : tensor<64x64x64xf32>
      %3 = scf.forall (%arg3) in (64) shared_outs(%arg4 = %2) -> (tensor<64x64x64xf32>) {
        %extracted_slice = tensor.extract_slice %1[%arg3, 0, 0] [1, 64, 128] [1, 1, 1] : tensor<64x64x128xf32> to tensor<1x64x128xf32>
        %extracted_slice_0 = tensor.extract_slice %arg2[0, 0] [128, 64] [1, 1] : tensor<128x64xf32> to tensor<128x64xf32>
        %extracted_slice_1 = tensor.extract_slice %arg4[%arg3, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<64x64x64xf32> to tensor<1x64x64xf32>
        %4 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%extracted_slice, %extracted_slice_0 : tensor<1x64x128xf32>, tensor<128x64xf32>) outs(%extracted_slice_1 : tensor<1x64x64xf32>) -> tensor<1x64x64xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %4 into %arg4[%arg3, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<1x64x64xf32> into tensor<64x64x64xf32>
        }
      } {mapping = [#furiosa.mapping]}
      return %3 : tensor<64x64x64xf32>
    }
    func.func @main() {
      %0 = tensor.empty() : tensor<64x64x64xf32>
      %1 = tensor.empty() : tensor<64x128xf32>
      %2 = tensor.empty() : tensor<128x64xf32>
      %3 = call @kernel(%0, %1, %2) {target = #furiosa.target<npu 0 pe 0 : 0>} : (tensor<64x64x64xf32>, tensor<64x128xf32>, tensor<128x64xf32>) -> tensor<64x64x64xf32>
      return
    }
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.contract"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %tiled_op, %forall_op = transform.structured.tile_using_forall %0 num_threads [64, 0, 0](mapping = [#furiosa.mapping]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield 
    }
  }
}

