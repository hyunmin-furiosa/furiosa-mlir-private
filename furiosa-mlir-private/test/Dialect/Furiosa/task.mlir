// RUN: furiosa-mlir-opt -convert-func-to-furiosa-host

module {
  func.func @kernel(%arg0: tensor<64x64x64xi8>, %arg1: tensor<64x64x64xi8>, %arg2: tensor<64x64x64xi8>) attributes {address_allocated, target = #furiosa.target<npu 0 pe 0 : 0>} {
    %0 = furiosa_task.dma_descriptor source %arg0 : tensor<64x64x64xi8> {destination_base = 268435456 : i64, destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 4194304, 4194304, 4194304], indirect = 0 : i64, opcode = 0 : i64, source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 0, 0, 4096]}
    %1 = furiosa_task.dmaw %0  {dma_tag_id = 0 : i64, profile = false, profile_id = 0 : i64}
    %2 = furiosa_task.tuc.wait %1 {dma_tag_id = 0 : i32, target_context = false, type = true}
    %3 = furiosa_task.sfr.sub_fetch_unit {flit_count = 128 : i64, limits = [4, 4, 2, 2, 1, 8, 1, 1], strides = [8, 1024, 32, 64, 4096, 128, 0, 0], words_per_packet = 4 : i64}
    %4 = furiosa_task.mtosfr %3  {sfr_address = 16711936 : i64}
    %5 = furiosa_task.sfr.tensor_register_file {write_flits_per_period = 1 : i64, write_mac_rows = 8 : i64, write_row_count = 16 : i64, write_valid_flits_per_period = 1 : i64}
    %6 = furiosa_task.mtosfr %5  {sfr_address = 16712056 : i64}
    %7 = furiosa_task.sfr.sub_data_path_unit {data_path_route_sub_context = 16 : i64}
    %8 = furiosa_task.mtosfr %7  {sfr_address = 16712048 : i64}
    %9 = furiosa_task.tuc.exec %4, %6, %8 {context_id = false, subunit_bitmap = 321 : i32, target_context = true}
    %10 = furiosa_task.tuc.wait %9 {dma_tag_id = 0 : i32, target_context = true, type = false}
    %11 = furiosa_task.dma_descriptor source %arg1 : tensor<64x64x64xi8> {destination_base = 268439552 : i64, destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 4194304, 4194304, 4194304], indirect = 0 : i64, opcode = 0 : i64, source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 0, 0, 4096]}
    %12 = furiosa_task.dmaw %11  {dma_tag_id = 0 : i64, profile = false, profile_id = 0 : i64}
    %13 = furiosa_task.tuc.wait %12 {dma_tag_id = 0 : i32, target_context = false, type = true}
    %14 = furiosa_task.dma_descriptor source %arg2 : tensor<64x64x64xi8> {destination_base = 268443648 : i64, destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 4194304, 4194304, 4194304], indirect = 0 : i64, opcode = 0 : i64, source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 0, 0, 4096]}
    %15 = furiosa_task.dmaw %14  {dma_tag_id = 0 : i64, profile = false, profile_id = 0 : i64}
    %16 = furiosa_task.tuc.wait %15 {dma_tag_id = 0 : i32, target_context = false, type = true}
    %17 = furiosa_task.sfr.main_fetch_unit {base = 4096 : i64, fetch_size = 16 : i64, flit_count = 256 : i64, limits = [2, 2, 32, 2, 1, 2, 1, 1], strides = [64, 16, 128, 0, 0, 32, 0, 0], words_per_packet = 4 : i64}
    %18 = furiosa_task.mtosfr %17  {sfr_address = 16715776 : i64}
    %19 = furiosa_task.sfr.dot_product_engine {acc_cols = 32 : i64, acc_indexer_strides = [1, 0, 0, 0, 0, 0, 0, 0], acc_limit = 32 : i64, data_type = 2 : i64, feed_data_type = 1 : i64, feed_input_transpose = 1 : i64, flits_per_input = 2 : i64, iter_seq_limits = [4, 16, 4, 1, 2, 1, 1, 1], mac_rows = 8 : i64, mac_tree_depth = 1 : i64, mac_type = 1 : i64, pop_dim = 1 : i64, reg_indexer_strides = [32, 2, 128, 256, 0, 0, 0, 0], reg_read_log_size = 1 : i64}
    %20 = furiosa_task.mtosfr %19  {sfr_address = 16715912 : i64}
    %21 = furiosa_task.sfr.vector_route_unit {cast_compaction_mode = 2 : i64, compaction_mode_cast_compaction_count = 4 : i64, route_info_data_out_source = 1 : i64}
    %22 = furiosa_task.mtosfr %21  {sfr_address = 16716128 : i64}
    %23 = furiosa_task.sfr.main_commit_unit {base = 8192 : i64, commit_in_size = 32 : i64, commit_size = 32 : i64, limits = [8, 4, 2, 1, 2, 1, 1, 1], slice_enable_bitmap_mask = [-1, -1, -1, -1], strides = [128, 1024, 64, 4096, 32, 0, 0, 0]}
    %24 = furiosa_task.mtosfr %23  {sfr_address = 16716032 : i64}
    %25 = furiosa_task.sfr.main_data_path_unit {main_context = 11 : i64}
    %26 = furiosa_task.mtosfr %25  {sfr_address = 16715904 : i64}
    %27 = furiosa_task.tuc.exec %18, %20, %22, %24, %26 {context_id = false, subunit_bitmap = 47 : i32, target_context = false}
    %28 = furiosa_task.tuc.wait %27 {dma_tag_id = 0 : i32, target_context = false, type = false}
    %29 = furiosa_task.dma_descriptor destination %arg2 : tensor<64x64x64xi8> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 0, 0, 4096], indirect = 0 : i64, opcode = 0 : i64, source_base = 268443648 : i64, source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 4194304, 4194304, 4194304]}
    %30 = furiosa_task.dmaw %29  {dma_tag_id = 0 : i64, profile = false, profile_id = 0 : i64}
    %31 = furiosa_task.tuc.wait %30 {dma_tag_id = 0 : i32, target_context = false, type = true}
    return
  }
  func.func @main() attributes {address_allocated} {
    %0 = furiosa.alloc {address = 4194304 : i64, argument} : tensor<64x64x64xi8>
    %1 = furiosa.alloc {address = 4456448 : i64, argument} : tensor<64x64x64xi8>
    %2 = furiosa.alloc {address = 4718592 : i64, result} : tensor<64x64x64xi8>
    call @kernel(%0, %1, %2) {target = #furiosa.target<npu 0 pe 0 : 0>} : (tensor<64x64x64xi8>, tensor<64x64x64xi8>, tensor<64x64x64xi8>) -> ()
    furiosa.dealloc %2 : tensor<64x64x64xi8>
    furiosa.dealloc %1 : tensor<64x64x64xi8>
    furiosa.dealloc %0 : tensor<64x64x64xi8>
    return
  }
}
