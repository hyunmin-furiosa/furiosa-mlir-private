from furiosa_mlir import *
from furiosa_mlir.execution_engine import *
from furiosa_mlir.ir import *
from furiosa_mlir.passmanager import *
from furiosa_mlir.runtime.np_to_tensor import *

import furiosa_mlir.dialects.furiosa as furiosa

def test_execution_engine():
    with Context() as ctx, Location.unknown():
        module = Module.parse(
r"""
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
    %0 = furiosa_host.func_alloc {function = @kernel}
    %1 = furiosa_host.pe_program_load_inst %0 {dram_address = 0 : i64, spm_address = 0 : i64}
    %2 = furiosa_host.pe_program_launch {operands = [4194304, 4456448, 4718592], spm_address = 0 : i64}
    %3 = furiosa_host.hal_program_write_at %0 {dram_address = 0 : i64}
    %4 = furiosa_host.alloc {size = 262144 : i64}
    %5 = furiosa_host.hal_program_write_at %4 {dram_address = 4194304 : i64}
    %6 = furiosa_host.alloc {size = 262144 : i64}
    %7 = furiosa_host.hal_program_write_at %6 {dram_address = 4456448 : i64}
    %8 = furiosa_host.alloc {size = 262144 : i64}
    %9 = furiosa_host.hal_program_read_at %8 {dram_address = 4718592 : i64}
    %10 = furiosa_host.pe_program_seq %1, %2
    %11 = furiosa_host.hal_program_execute %10
    %12 = furiosa_host.hal_program_seq %3, %5, %7, %11, %9
    %13 = furiosa_host.device_new {target = #furiosa.target<npu 0 pe 0 : 0>}
    %14 = furiosa_host.device_execute %13 %12
    furiosa_host.device_execution_wait %14
    return
  }
}
"""
        )

        target = furiosa.TargetAttr.get(npu=0, pe_begin=0, pe_end=0)
        execution_engine = ExecutionEngine(module, target)
        execution_engine.invoke("main")

def test_kernel_task():
    with Context() as ctx, Location.unknown():
        module = Module.parse(
r"""
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
}
"""
        )

        arr0 = np.random.randint(0, 2, size=(64, 64, 64), dtype=np.int8)
        arr1 = np.random.randint(0, 2, size=(64, 64, 64), dtype=np.int8)
        arr2 = np.zeros((64, 64, 64), dtype=np.int8)
        arr0_desc = ctypes.pointer(get_ranked_tensor_descriptor(arr0))
        arr1_desc = ctypes.pointer(get_ranked_tensor_descriptor(arr1))
        arr2_desc = ctypes.pointer(get_ranked_tensor_descriptor(arr2))

        target = furiosa.TargetAttr.get(npu=0, pe_begin=0, pe_end=0)
        execution_engine = ExecutionEngine(module, target)
        execution_engine.invoke("kernel", [arr0_desc, arr1_desc], [arr2_desc])

        expected = np.einsum("nij,njk->nik", arr0, arr1)
        print(np.array_equal(arr2, expected))

def test_kernel_low_level():
    with Context() as ctx, Location.unknown():
        module = Module.parse(
r"""
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @kernel(%arg0: tensor<64x64x64xi8>, %arg1: tensor<64x64x64xi8>, %arg2: tensor<64x64x64xi8>) attributes {address_allocated, target = #furiosa.target<npu 0 pe 0 : 0>} {
    %0 = furiosa.alloc {address = 0 : i64} : tensor<1x64x64xi8, #furiosa.memory_type<sram>>
    furiosa.dma %arg0 -> %0 : tensor<64x64x64xi8> -> tensor<1x64x64xi8, #furiosa.memory_type<sram>> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 4194304, 4194304, 4194304], source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 0, 0, 4096]}
    %1 = furiosa.alloc {address = 0 : i64} : tensor<1x64x64xi8, #furiosa.memory_type<trf>>
    furiosa.load_trf %0 -> %1 : tensor<1x64x64xi8, #furiosa.memory_type<sram>> -> tensor<1x64x64xi8, #furiosa.memory_type<trf>>
    furiosa.dealloc %0 : tensor<1x64x64xi8, #furiosa.memory_type<sram>>
    %2 = furiosa.alloc {address = 4096 : i64} : tensor<1x64x64xi8, #furiosa.memory_type<sram>>
    furiosa.dma %arg1 -> %2 : tensor<64x64x64xi8> -> tensor<1x64x64xi8, #furiosa.memory_type<sram>> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 4194304, 4194304, 4194304], source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 0, 0, 4096]}
    %3 = furiosa.alloc {address = 8192 : i64} : tensor<1x64x64xi8, #furiosa.memory_type<sram>>
    furiosa.dma %arg2 -> %3 : tensor<64x64x64xi8> -> tensor<1x64x64xi8, #furiosa.memory_type<sram>> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 4194304, 4194304, 4194304], source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 0, 0, 4096]}
    %4 = linalg.contract indexing_maps = [#map, #map1, #map2] {context_id = false} ins(%1, %2 : tensor<1x64x64xi8, #furiosa.memory_type<trf>>, tensor<1x64x64xi8, #furiosa.memory_type<sram>>) outs(%3 : tensor<1x64x64xi8, #furiosa.memory_type<sram>>) -> tensor<1x64x64xi8, #furiosa.memory_type<sram>>
    furiosa.dealloc %3 : tensor<1x64x64xi8, #furiosa.memory_type<sram>>
    furiosa.dealloc %2 : tensor<1x64x64xi8, #furiosa.memory_type<sram>>
    furiosa.dealloc %1 : tensor<1x64x64xi8, #furiosa.memory_type<trf>>
    furiosa.dma %4 -> %arg2 : tensor<1x64x64xi8, #furiosa.memory_type<sram>> -> tensor<64x64x64xi8> {destination_limits = [64, 64, 1, 1, 1, 64], destination_strides = [1, 64, 4096, 0, 0, 4096], source_limits = [64, 64, 1, 1, 1, 64], source_strides = [1, 64, 4096, 4194304, 4194304, 4194304]}
    return
  }
}
"""
        )
        pm = PassManager.parse("any(convert-furiosa-to-furiosa-task)")
        pm.run(module.operation)

        arr0 = np.random.randint(0, 2, size=(64, 64, 64), dtype=np.int8)
        arr1 = np.random.randint(0, 2, size=(64, 64, 64), dtype=np.int8)
        arr2 = np.zeros((64, 64, 64), dtype=np.int8)
        arr0_desc = ctypes.pointer(get_ranked_tensor_descriptor(arr0))
        arr1_desc = ctypes.pointer(get_ranked_tensor_descriptor(arr1))
        arr2_desc = ctypes.pointer(get_ranked_tensor_descriptor(arr2))

        target = furiosa.TargetAttr.get(npu=0, pe_begin=0, pe_end=0)
        execution_engine = ExecutionEngine(module, target)
        execution_engine.invoke("kernel", [arr0_desc, arr1_desc], [arr2_desc])

        expected = np.einsum("nij,njk->nik", arr0, arr1)
        print(np.array_equal(arr2, expected))

def test_kernel_high_level():
    with Context() as ctx, Location.unknown():
        module = Module.parse(
r"""
module {
  func.func @kernel(%arg0: tensor<64x64x64xi8>, %arg1: tensor<64x64x64xi8>) -> (tensor<64x64x64xi8>) attributes { target = #furiosa.target<npu 0 pe 0:0> } {
    %0 = tensor.empty() : tensor<64x64x64xi8>
    %a_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %b_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp : (tensor<64x64x64xi8>, tensor<64x64x64xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<64x64x64xi8>
    return %1 : tensor<64x64x64xi8>
  }
}
"""
        )
        pm = PassManager.parse("builtin.module(func.func(tosa-to-linalg-named),linalg-generalize-to-contract-ops)")
        pm.run(module.operation)

        transform_module = Module.parse(
r"""
module {
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:1 = transform.structured.tile_using_forall %0 num_threads [64] { mapping = [ #furiosa.mapping ] } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}
"""
        )

        module.body.append(transform_module.body.operations[0])

        pm = PassManager.parse("builtin.module(transform-interpreter)")
        pm.run(module.operation)
        module.body.operations[-1].erase()

        pm = PassManager.parse("builtin.module(convert-linalg-to-furiosa,furiosa-promote-slice-partition-loop,func-results-to-params,furiosa-deallocation,func.func(optimize-allocation-liveness),furiosa-allocate-address,convert-furiosa-to-furiosa-task)")
        pm.run(module.operation)

        arr0 = np.random.randint(0, 2, size=(64, 64, 64), dtype=np.int8)
        arr1 = np.random.randint(0, 2, size=(64, 64, 64), dtype=np.int8)
        arr2 = np.zeros((64, 64, 64), dtype=np.int8)
        arr0_desc = ctypes.pointer(get_ranked_tensor_descriptor(arr0))
        arr1_desc = ctypes.pointer(get_ranked_tensor_descriptor(arr1))
        arr2_desc = ctypes.pointer(get_ranked_tensor_descriptor(arr2))

        target = furiosa.TargetAttr.get(npu=0, pe_begin=0, pe_end=0)
        execution_engine = ExecutionEngine(module, target)
        execution_engine.invoke("kernel", [arr0_desc, arr1_desc], [arr2_desc])

        expected = np.einsum("nij,njk->nik", arr0, arr1)
        print(np.array_equal(arr2, expected))

test_execution_engine()
test_kernel_task()
test_kernel_low_level()
test_kernel_high_level()
