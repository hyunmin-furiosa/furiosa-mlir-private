// RUN: furiosa-mlir-opt -furiosa-deallocation

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @kernel(%arg0: tensor<64x64x64xf32>, %arg1: tensor<64x64x64xf32>, %arg2: tensor<64x64x64xf32>) attributes {target = #furiosa.target<npu 0 pe 0 : 0>} {
    %0 = furiosa.alloc : tensor<1x64x64xf32, #furiosa.memory_type<sram>>
    furiosa.dma %arg0 -> %0 : tensor<64x64x64xf32> -> tensor<1x64x64xf32, #furiosa.memory_type<sram>> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 4194304, 4194304, 4194304], source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 0, 0, 4096]}
    %1 = furiosa.alloc : tensor<1x64x64xf32, #furiosa.memory_type<sram>>
    furiosa.dma %arg1 -> %1 : tensor<64x64x64xf32> -> tensor<1x64x64xf32, #furiosa.memory_type<sram>> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 4194304, 4194304, 4194304], source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 0, 0, 4096]}
    %2 = furiosa.alloc : tensor<1x64x64xf32, #furiosa.memory_type<trf>>
    furiosa.load_trf %1 -> %2 : tensor<1x64x64xf32, #furiosa.memory_type<sram>> -> tensor<1x64x64xf32, #furiosa.memory_type<trf>>
    %3 = furiosa.alloc : tensor<1x64x64xf32, #furiosa.memory_type<sram>>
    furiosa.dma %arg2 -> %3 : tensor<64x64x64xf32> -> tensor<1x64x64xf32, #furiosa.memory_type<sram>> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 4194304, 4194304, 4194304], source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 0, 0, 4096]}
    %4 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%0, %2 : tensor<1x64x64xf32, #furiosa.memory_type<sram>>, tensor<1x64x64xf32, #furiosa.memory_type<trf>>) outs(%3 : tensor<1x64x64xf32, #furiosa.memory_type<sram>>) -> tensor<1x64x64xf32, #furiosa.memory_type<sram>>
    furiosa.dma %4 -> %arg2 : tensor<1x64x64xf32, #furiosa.memory_type<sram>> -> tensor<64x64x64xf32> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 0, 0, 4096], source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 4194304, 4194304, 4194304]}
    return
  }
  func.func @main() {
    %0 = furiosa.alloc : tensor<64x64x64xf32>
    %1 = furiosa.alloc : tensor<64x64x64xf32>
    %2 = furiosa.alloc : tensor<64x64x64xf32>
    call @kernel(%0, %1, %2) : (tensor<64x64x64xf32>, tensor<64x64x64xf32>, tensor<64x64x64xf32>) -> ()
    return
  }
}
