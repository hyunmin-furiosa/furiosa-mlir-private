// RUN: furiosa-mlir-opt -furiosa-promote-slice-partition-loop

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  module {
    func.func @kernel(%arg0: tensor<64x64x64xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<128x64xf32>) -> tensor<64x64x64xf32> attributes {target = #furiosa.target<npu 0 pe 0 : 0>} {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = tensor.empty() : tensor<64x64x128xf32>
      %1 = scf.forall (%arg3, %arg4) in (32, 2) shared_outs(%arg5 = %0) -> (tensor<64x64x128xf32>) {
        %4 = furiosa.dma %arg0 : tensor<64x64x64xf32> -> tensor<2x32x64xf32, #furiosa.memory_type<sram>> {destination_limits = [64, 32, 2, 1, 2, 32], destination_strides = [1, 64, 2048, 4194304, 4194304, 8388608], source_limits = [64, 32, 2, 1, 2, 32], source_strides = [1, 64, 4096, 0, 2048, 8192]}
        %5 = furiosa.dma %arg1 : tensor<64x128xf32> -> tensor<64x128xf32, #furiosa.memory_type<trf>> {destination_limits = [128, 64, 1, 1, 64], destination_strides = [1, 128, 4194304, 4194304, 0], source_limits = [128, 64, 1, 1, 64], source_strides = [1, 128, 0, 0, 0]}
        %6 = furiosa.dma %arg5 : tensor<64x64x128xf32> -> tensor<2x32x128xf32, #furiosa.memory_type<sram>> {destination_limits = [128, 32, 2, 1, 2, 32], destination_strides = [1, 128, 4096, 4194304, 4194304, 8388608], source_limits = [128, 32, 2, 1, 2, 32], source_strides = [1, 128, 8192, 0, 4096, 16384]}
        %7 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%4, %5 : tensor<2x32x64xf32, #furiosa.memory_type<sram>>, tensor<64x128xf32, #furiosa.memory_type<trf>>) outs(%6 : tensor<2x32x128xf32, #furiosa.memory_type<sram>>) -> tensor<2x32x128xf32, #furiosa.memory_type<sram>>
        furiosa.dma %7 : tensor<2x32x128xf32, #furiosa.memory_type<sram>> -> %arg5 : tensor<64x64x128xf32>  {destination_limits = [128, 32, 2, 1, 2, 32], destination_strides = [1, 128, 8192, 0, 4096, 16384], source_limits = [128, 32, 2, 1, 2, 32], source_strides = [1, 128, 4096, 4194304, 4194304, 8388608]}
        scf.forall.in_parallel {
        }
      } {mapping = [#furiosa.mapping, #furiosa.mapping]}
      %2 = tensor.empty() : tensor<64x64x64xf32>
      %3 = scf.forall (%arg3, %arg4) in (32, 2) shared_outs(%arg5 = %2) -> (tensor<64x64x64xf32>) {
        %4 = furiosa.dma %1 : tensor<64x64x128xf32> -> tensor<2x32x128xf32, #furiosa.memory_type<sram>> {destination_limits = [128, 32, 2, 1, 2, 32], destination_strides = [1, 128, 4096, 4194304, 4194304, 8388608], source_limits = [128, 32, 2, 1, 2, 32], source_strides = [1, 128, 8192, 0, 4096, 16384]}
        %5 = furiosa.dma %arg2 : tensor<128x64xf32> -> tensor<128x64xf32, #furiosa.memory_type<trf>> {destination_limits = [64, 128, 1, 1, 64], destination_strides = [1, 64, 4194304, 4194304, 0], source_limits = [64, 128, 1, 1, 64], source_strides = [1, 64, 0, 0, 0]}
        %6 = furiosa.dma %arg5 : tensor<64x64x64xf32> -> tensor<2x32x64xf32, #furiosa.memory_type<sram>> {destination_limits = [64, 32, 2, 1, 2, 32], destination_strides = [1, 64, 2048, 4194304, 4194304, 8388608], source_limits = [64, 32, 2, 1, 2, 32], source_strides = [1, 64, 4096, 0, 2048, 8192]}
        %7 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%4, %5 : tensor<2x32x128xf32, #furiosa.memory_type<sram>>, tensor<128x64xf32, #furiosa.memory_type<trf>>) outs(%6 : tensor<2x32x64xf32, #furiosa.memory_type<sram>>) -> tensor<2x32x64xf32, #furiosa.memory_type<sram>>
        furiosa.dma %7 : tensor<2x32x64xf32, #furiosa.memory_type<sram>> -> %arg5 : tensor<64x64x64xf32>  {destination_limits = [64, 32, 2, 1, 2, 32], destination_strides = [1, 64, 4096, 0, 2048, 8192], source_limits = [64, 32, 2, 1, 2, 32], source_strides = [1, 64, 2048, 4194304, 4194304, 8388608]}
        scf.forall.in_parallel {
        }
      } {mapping = [#furiosa.mapping, #furiosa.mapping]}
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
      %tiled_op, %forall_op = transform.structured.tile_using_forall %0 num_threads [32, 2, 0, 0](mapping = [#furiosa.mapping, #furiosa.mapping]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield 
    }
  }
}
