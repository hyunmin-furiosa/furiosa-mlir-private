// RUN: furiosa-mlir-opt -furiosa-promote-slice-partition-loop

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @kernel(%arg0: tensor<64x64x64xf8E5M2>, %arg1: tensor<64x64x64xf8E5M2>) -> tensor<64x64x64xf8E5M2> attributes {target = #furiosa.target<npu 0 pe 0 : 0>} {
    %cst = arith.constant 0.000000e+00 : f8E5M2
    %0 = furiosa.alloc : tensor<64x64x64xf8E5M2>
    %1 = scf.forall (%arg2) in (64) shared_outs(%arg3 = %0) -> (tensor<64x64x64xf8E5M2>) {
      %2 = furiosa.alloc : tensor<1x64x64xf8E5M2, #furiosa.memory_type<sram>>
      furiosa.dma %arg0 -> %2 : tensor<64x64x64xf8E5M2> -> tensor<1x64x64xf8E5M2, #furiosa.memory_type<sram>> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 4194304, 4194304, 4194304], source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 0, 0, 4096]}
      %3 = furiosa.alloc : tensor<1x64x64xf8E5M2, #furiosa.memory_type<trf>>
      furiosa.load_trf %2 -> %3 : tensor<1x64x64xf8E5M2, #furiosa.memory_type<sram>> -> tensor<1x64x64xf8E5M2, #furiosa.memory_type<trf>>
      %4 = furiosa.alloc : tensor<1x64x64xf8E5M2, #furiosa.memory_type<sram>>
      furiosa.dma %arg1 -> %4 : tensor<64x64x64xf8E5M2> -> tensor<1x64x64xf8E5M2, #furiosa.memory_type<sram>> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 4194304, 4194304, 4194304], source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 0, 0, 4096]}
      %5 = furiosa.alloc : tensor<1x64x64xf8E5M2, #furiosa.memory_type<sram>>
      furiosa.dma %arg3 -> %5 : tensor<64x64x64xf8E5M2> -> tensor<1x64x64xf8E5M2, #furiosa.memory_type<sram>> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 4194304, 4194304, 4194304], source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 0, 0, 4096]}
      %6 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%3, %4 : tensor<1x64x64xf8E5M2, #furiosa.memory_type<trf>>, tensor<1x64x64xf8E5M2, #furiosa.memory_type<sram>>) outs(%5 : tensor<1x64x64xf8E5M2, #furiosa.memory_type<sram>>) -> tensor<1x64x64xf8E5M2, #furiosa.memory_type<sram>>
      furiosa.dma %6 -> %arg3 : tensor<1x64x64xf8E5M2, #furiosa.memory_type<sram>> -> tensor<64x64x64xf8E5M2> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 0, 0, 4096], source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 4194304, 4194304, 4194304]}
      scf.forall.in_parallel {
      }
    } {mapping = [#furiosa.mapping]}
    return %1 : tensor<64x64x64xf8E5M2>
  }
  func.func @main() {
    %0 = furiosa.alloc : tensor<64x64x64xf8E5M2>
    %1 = furiosa.alloc : tensor<64x64x64xf8E5M2>
    %2 = call @kernel(%0, %1) {target = #furiosa.target<npu 0 pe 0 : 0>} : (tensor<64x64x64xf8E5M2>, tensor<64x64x64xf8E5M2>) -> tensor<64x64x64xf8E5M2>
    return
  }
}
