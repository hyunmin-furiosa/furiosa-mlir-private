module {
  func.func @kernel(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> () attributes { target = #furiosa.target<npu 0 pe 0:0> } {
    furiosa.tuc.rtosfr {sfr_address = 0 : i64, size = 1 : i64, value = 12424 : i64}
    furiosa.tuc.wait {dma_tag_id = 0 : i32, target_context = false, type = false}
    %0 = furiosa.task.dma_descriptor source %arg0 : tensor<256xf32> {opcode = 0, indirect = 0, destination_base = 0x0010000000, source_limits = [4,1,1,1,1,1,1,1], source_strides = [256,0,0,0,0,0,0,0], destination_limits = [4,1,1,1,1,1,1,1], destination_strides = [256,0,0,0,0,0,0,0]}
    furiosa.task.dmaw %0 {dma_tag_id = 0, profile = false, profile_id = 0}
    furiosa.tuc.wait {dma_tag_id = 0 : i32, type = true, target_context = false}
    %1 = furiosa.task.sfr.sub_fetch_unit {base = 0x0, type_conversion = 0, num_zero_points = 0, zero_point0 = 0, zero_point1 = 0, limits = [128,1,1,1,1,1,1,1], strides = [8,0,0,0,0,0,0,0], flit_count = 128, words_per_packet = 1, topology = 0 }
    %2 = furiosa.task.sfr.sub_commit_unit {mode = 0, packet_valid_count = 8, base = 0x10000, commit_in_size = 8, commit_data = 0, limits = [128,1,1,1,1,1,1,1], strides = [8,0,0,0,0,0,0,0], slice_enable_bitmap_mask = [0xffffffffffffffff,0xffffffffffffffff,0xffffffffffffffff,0xffffffffffffffff] }
    %3 = furiosa.task.sfr.sub_data_path_unit {route = 0x08}
    furiosa.task.mtosfr %1 {sfr_address = 0xff0100}
    furiosa.task.mtosfr %2 {sfr_address = 0xff0198}
    furiosa.task.mtosfr %3 {sfr_address = 0xff0170}
    furiosa.tuc.exec {subunit_bitmap = 0x0c1 : i32, context_id = false, target_context = true}
    furiosa.tuc.wait {dma_tag_id = 0 : i32, type = false, target_context = true}
    %4 = furiosa.task.dma_descriptor destination %arg1 : tensor<256xf32> {opcode = 0, indirect = 0, source_base = 0x0010010000, source_limits = [4,1,1,1,1,1,1,1], source_strides = [256,0,0,0,0,0,0,0], destination_limits = [4,1,1,1,1,1,1,1], destination_strides = [256,0,0,0,0,0,0,0]}
    furiosa.task.dmaw %4 {dma_tag_id = 1, profile = false, profile_id = 0}
    furiosa.tuc.wait {dma_tag_id = 1 : i32, type = true, target_context = false}
    return
  }
  func.func @main() {
    %arg0 = tensor.empty() { dram_address = 0x10000, argument } : tensor<256xf32>
    %res0 = tensor.empty() { dram_address = 0x20000, result } : tensor<256xf32>
    func.call @kernel(%arg0, %res0) { dram_address = 0x0, spm_address = 0x0, target = #furiosa.target<npu 0 pe 0:0> } : (tensor<256xf32>, tensor<256xf32>) -> ()
    furiosa_host.print %res0 : tensor<256xf32>
    return
  }
}
