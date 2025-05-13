// RUN: furiosa-mlir-runner

module {
  func.func @kernel(%arg0: i64, %arg1: i64) attributes { target = #furiosa.target<npu 0 pe 0:0> } {
    furiosa_task.tuc.rtosfr {sfr_address = 0 : i64, size = 1 : i64, value = 12424 : i64}
    furiosa_task.tuc.wait {dma_tag_id = 0 : i32, target_context = false, type = false}
    %0 = furiosa_task.dma_descriptor source %arg0 : i64 {opcode = 0, indirect = 0, destination_base = 0x0010000000, source_limits = [4,1,1,1,1,1,1,1], source_strides = [256,0,0,0,0,0,0,0], destination_limits = [4,1,1,1,1,1,1,1], destination_strides = [256,0,0,0,0,0,0,0]}
    furiosa_task.dmaw %0 {dma_tag_id = 0, profile = false, profile_id = 0}
    furiosa_task.tuc.wait {dma_tag_id = 0 : i32, type = true, target_context = false}
    %1 = furiosa_task.sfr.sub_fetch_unit {base = 0x0, type_conversion = 0, num_zero_points = 0, zero_point0 = 0, zero_point1 = 0, limits = [128,1,1,1,1,1,1,1], strides = [8,0,0,0,0,0,0,0], flit_count = 128, words_per_packet = 1, topology = 0 }
    %2 = furiosa_task.sfr.sub_commit_unit {mode = 0, packet_valid_count = 8, base = 0x10000, commit_in_size = 8, commit_data = 0, limits = [128,1,1,1,1,1,1,1], strides = [8,0,0,0,0,0,0,0], slice_enable_bitmap_mask = [0xffffffffffffffff,0xffffffffffffffff,0xffffffffffffffff,0xffffffffffffffff] }
    %3 = furiosa_task.sfr.sub_data_path_unit {route = 0x08}
    furiosa_task.mtosfr %1 {sfr_address = 0xff0100}
    furiosa_task.mtosfr %2 {sfr_address = 0xff0198}
    furiosa_task.mtosfr %3 {sfr_address = 0xff0170}
    furiosa_task.tuc.exec {subunit_bitmap = 0x0c1 : i32, context_id = false, target_context = true}
    furiosa_task.tuc.wait {dma_tag_id = 0 : i32, type = false, target_context = true}
    %4 = furiosa_task.dma_descriptor destination %arg1 : i64 {opcode = 0, indirect = 0, source_base = 0x0010010000, source_limits = [4,1,1,1,1,1,1,1], source_strides = [256,0,0,0,0,0,0,0], destination_limits = [4,1,1,1,1,1,1,1], destination_strides = [256,0,0,0,0,0,0,0]}
    furiosa_task.dmaw %4 {dma_tag_id = 1, profile = false, profile_id = 0}
    furiosa_task.tuc.wait {dma_tag_id = 1 : i32, type = true, target_context = false}
    return
  }
  func.func @main() {
    %binary = furiosa_host.func_alloc { function = @kernel }
    %arg0 = furiosa_host.alloc { size = 0x400, data = [1] }
    %res0 = furiosa_host.alloc { size = 0x400, data = [0] }
    %pe0 = furiosa_host.pe_program_load_inst %binary { dram_address = 0x0, spm_address = 0x0 }
    %pe1 = furiosa_host.pe_program_launch { spm_address = 0x0, operands = [0x10000, 0x20000] }
    %pe = furiosa_host.pe_program_seq %pe0, %pe1
    %hal0 = furiosa_host.hal_program_write_at %binary { dram_address = 0x0 }
    %hal1 = furiosa_host.hal_program_write_at %arg0 { dram_address = 0x10000 }
    %hal2 = furiosa_host.hal_program_execute %pe
    %hal3 = furiosa_host.hal_program_read_at %res0 { dram_address = 0x20000 }
    %hal = furiosa_host.hal_program_seq %hal0, %hal1, %hal2, %hal3
    %dev = furiosa_host.device_new { target = #furiosa.target<npu 0 pe 0:0> }
    %exec = furiosa_host.device_execute %dev %hal
    furiosa_host.device_execution_wait %exec
    %comp = furiosa_host.compare %arg0 %res0
    furiosa_host.print %comp : i1
    furiosa_host.print %res0 : !furiosa_host.buffer
    return
  }
}
