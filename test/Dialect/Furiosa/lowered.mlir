// RUN: furiosa-mlir-opt -convert-func-to-furiosa-host

module {
  func.func @kernel(%arg0: tensor<64x256xf32>, %arg1: tensor<64x256xf32>) -> () attributes { target = #furiosa.target<npu 0 pe 0:0> } {
    %desc0 = furiosa_task.dma_descriptor source %arg0 : tensor<64x256xf32> {opcode = 0, indirect = 0, destination_base = 0x0010000000, source_limits = [0x4,0x40,1,1,1,1,1,1], source_strides = [0x100,0x400,0,0,0,0,0,0], destination_limits = [0x4,0x40,1,1,1,1,1,1], destination_strides = [0x100,0x400000,0,0,0,0,0,0]}
    %dma0 = furiosa_task.dmaw %desc0 {dma_tag_id = 0, profile = false, profile_id = 0}
    %dma0_wait = furiosa_task.tuc.wait %dma0 {dma_tag_id = 0 : i32, type = true, target_context = false}
    %sub_fetch_sfr = furiosa_task.sfr.sub_fetch_unit {base = 0x0, type_conversion = 0, num_zero_points = 0, zero_point0 = 0, zero_point1 = 0, limits = [128,1,1,1,1,1,1,1], strides = [8,0,0,0,0,0,0,0], flit_count = 128, words_per_packet = 1, topology = 0 }
    %sub_commit_sfr = furiosa_task.sfr.sub_commit_unit {mode = 0, packet_valid_count = 8, base = 0x10000, commit_in_size = 8, commit_data = 0, limits = [128,1,1,1,1,1,1,1], strides = [8,0,0,0,0,0,0,0], slice_enable_bitmap_mask = [0xffffffffffffffff,0xffffffffffffffff,0xffffffffffffffff,0xffffffffffffffff] }
    %sub_data_path_sfr = furiosa_task.sfr.sub_data_path_unit {route = 0x08}
    %mtosfr0 = furiosa_task.mtosfr %sub_fetch_sfr {sfr_address = 0xff0100}
    %mtosfr1 = furiosa_task.mtosfr %sub_commit_sfr {sfr_address = 0xff0198}
    %mtosfr2 = furiosa_task.mtosfr %sub_data_path_sfr {sfr_address = 0xff0170}
    %exec = furiosa_task.tuc.exec %dma0_wait, %mtosfr0, %mtosfr1, %mtosfr2 {subunit_bitmap = 0x0c1 : i32, context_id = false, target_context = true}
    %exec_wait = furiosa_task.tuc.wait %exec {dma_tag_id = 0 : i32, type = false, target_context = true}
    %desc1 = furiosa_task.dma_descriptor destination %arg1 : tensor<64x256xf32> {opcode = 0, indirect = 0, source_base = 0x0010010000, source_limits = [0x4,0x40,1,1,1,1,1,1], source_strides = [0x100,0x400000,0,0,0,0,0,0], destination_limits = [0x4,0x40,1,1,1,1,1,1], destination_strides = [0x100,0x400,0,0,0,0,0,0]}
    %dma1 = furiosa_task.dmaw %desc1 %exec_wait {dma_tag_id = 1, profile = false, profile_id = 0}
    %dma1_wait = furiosa_task.tuc.wait %dma1 {dma_tag_id = 1 : i32, type = true, target_context = false}
    return
  }
  func.func @main() {
    %arg0 = tensor.empty() { dram_address = 0x10000, argument, data = [1] } : tensor<64x256xf32>
    %res0 = tensor.empty() { dram_address = 0x20000, result } : tensor<64x256xf32>
    func.call @kernel(%arg0, %res0) { dram_address = 0x0, spm_address = 0x0, target = #furiosa.target<npu 0 pe 0:0> } : (tensor<64x256xf32>, tensor<64x256xf32>) -> ()
    furiosa_host.print %res0 : tensor<64x256xf32>
    return
  }
}
