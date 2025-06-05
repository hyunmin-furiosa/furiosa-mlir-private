// RUN: furiosa-mlir-opt -furiosa-deallocation

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @kernel(%arg0: tensor<64x64x64xi8>, %arg1: tensor<64x64x64xi8>, %arg2: tensor<64x64x64xi8>) attributes {target = #furiosa.target<npu 0 pe 0 : 0>} {
    %0 = furiosa.alloc : tensor<1x64x64xi8, #furiosa.memory_type<sram>>
    furiosa.dma %arg0 -> %0 : tensor<64x64x64xi8> -> tensor<1x64x64xi8, #furiosa.memory_type<sram>> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 4194304, 4194304, 4194304], source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 0, 0, 4096]}
    %1 = furiosa.alloc : tensor<1x64x64xi8, #furiosa.memory_type<trf>>
    furiosa.load_trf %0 -> %1 : tensor<1x64x64xi8, #furiosa.memory_type<sram>> -> tensor<1x64x64xi8, #furiosa.memory_type<trf>>
    %2 = furiosa.alloc : tensor<1x64x64xi8, #furiosa.memory_type<sram>>
    furiosa.dma %arg1 -> %2 : tensor<64x64x64xi8> -> tensor<1x64x64xi8, #furiosa.memory_type<sram>> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 4194304, 4194304, 4194304], source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 0, 0, 4096]}
    %3 = furiosa.alloc : tensor<1x64x64xi8, #furiosa.memory_type<sram>>
    furiosa.dma %arg2 -> %3 : tensor<64x64x64xi8> -> tensor<1x64x64xi8, #furiosa.memory_type<sram>> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 4194304, 4194304, 4194304], source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 0, 0, 4096]}
    %4 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%1, %2 : tensor<1x64x64xi8, #furiosa.memory_type<trf>>, tensor<1x64x64xi8, #furiosa.memory_type<sram>>) outs(%3 : tensor<1x64x64xi8, #furiosa.memory_type<sram>>) -> tensor<1x64x64xi8, #furiosa.memory_type<sram>>
    furiosa.dma %4 -> %arg2 : tensor<1x64x64xi8, #furiosa.memory_type<sram>> -> tensor<64x64x64xi8> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 0, 0, 4096], source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 4194304, 4194304, 4194304]}
    return
  }
  func.func @main() {
    %0 = furiosa.alloc {argument} : tensor<64x64x64xi8>
    %1 = furiosa.alloc {argument} : tensor<64x64x64xi8>
    %2 = furiosa.alloc {result} : tensor<64x64x64xi8>
    call @kernel(%0, %1, %2) {target = #furiosa.target<npu 0 pe 0 : 0>} : (tensor<64x64x64xi8>, tensor<64x64x64xi8>, tensor<64x64x64xi8>) -> ()
    return
  }
}
