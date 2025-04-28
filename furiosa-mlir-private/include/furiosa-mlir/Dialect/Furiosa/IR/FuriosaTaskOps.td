//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef FURIOSA_TASK_OPS_TD
#define FURIOSA_TASK_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaBase.td"
include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaTypes.td"

//===----------------------------------------------------------------------===//
// Shared dag type definitions
//===----------------------------------------------------------------------===//

class NamedDag<string m, dag d> {
  string mnemonic = m;
  dag data = d;
}

defset list<NamedDag> Sfr = {
  def MainFetchUnit
      : NamedDag<
            "main_fetch_unit",
            (ins
                DefaultValuedOptionalAttr<I64Attr, "0">:$fetch_mode,
                DefaultValuedOptionalAttr<I64Attr, "0">:$num_zero_points,
                DefaultValuedOptionalAttr<I64Attr, "0">:$zero_point0,
                DefaultValuedOptionalAttr<I64Attr, "0">:$zero_point1,
                DefaultValuedOptionalAttr<I64Attr, "0">:$table_entry_size,
                DefaultValuedOptionalAttr<I64Attr, "0">:$tables,
                DefaultValuedOptionalAttr<I64Attr, "0">:$indirect_base,
                DefaultValuedOptionalAttr<I64Attr, "0">:$indirect_dim,
                DefaultValuedOptionalAttr<I64Attr, "0">:$table_base_mode,
                DefaultValuedOptionalAttr<I64Attr, "0">:$indirect_pointer_size,
                DefaultValuedOptionalAttr<I64Attr, "0">:$zeropoint_tail_mode,
                DefaultValuedOptionalAttr<I64Attr, "0">:$last_dim_pad_value,
                DefaultValuedOptionalAttr<I64Attr, "0">:$last_dim,
                DefaultValuedOptionalAttr<I64Attr, "0">:$pad_order,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$last_dim_rightmost_valid_count_dim,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$last_dim_left_pad_count,
                DefaultValuedOptionalAttr<I64Attr, "0">:$type_conversion,
                DefaultValuedOptionalAttr<I64Attr, "0">:$last_dim_left_pad_mode,
                DefaultValuedOptionalAttr<I64Attr, "0">:$zeropoint_dims,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr,
                    "{0,0,0,0,0,0,0,0}">:$last_dim_rightmost_valid_count,
                DefaultValuedOptionalAttr<I64Attr, "0">:$base,
                DefaultValuedOptionalAttr<I64Attr, "0">:$fetch_size,
                DefaultValuedOptionalAttr<I64ArrayAttr,
                                          "{0,0,0,0,0,0,0,0}">:$limits,
                DefaultValuedOptionalAttr<I64ArrayAttr,
                                          "{0,0,0,0,0,0,0,0}">:$strides,
                DefaultValuedOptionalAttr<I64Attr, "0">:$flit_count,
                DefaultValuedOptionalAttr<I64Attr, "0">:$words_per_packet,
                DefaultValuedOptionalAttr<I64Attr, "0">:$zeropoint_fetch_limit,
                DefaultValuedOptionalAttr<I64Attr, "0">:$topology,
                DefaultValuedOptionalAttr<I64Attr, "0">:$channel_config,
                DefaultValuedOptionalAttr<I64Attr, "0">:$outer_slice_log_size,
                DefaultValuedOptionalAttr<I64Attr, "0">:$outer_dim0_log_size,
                DefaultValuedOptionalAttr<I64Attr, "0">:$outer_dim1_log_size,
                DefaultValuedOptionalAttr<I64Attr, "0">:$outer_dim0_chunk_size,
                DefaultValuedOptionalAttr<I64Attr, "0">:$outer_dim1_chunk_size,
                DefaultValuedOptionalAttr<I64ArrayAttr,
                                          "{0,0,0,0}">:$custom_snoop_bitmap_mask
            )>;

  def MainCommitUnit
      : NamedDag<"main_commit_unit",
                 (ins
                     DefaultValuedOptionalAttr<I64Attr, "0">:$type_conversion,
                     DefaultValuedOptionalAttr<I64Attr, "0">:$base,
                     DefaultValuedOptionalAttr<I64Attr, "0">:$commit_in_size,
                     DefaultValuedOptionalAttr<I64Attr, "0">:$commit_size,
                     DefaultValuedOptionalAttr<I64ArrayAttr,
                                               "{0,0,0,0,0,0,0,0}">:$limits,
                     DefaultValuedOptionalAttr<I64ArrayAttr,
                                               "{0,0,0,0,0,0,0,0}">:$strides,
                     DefaultValuedOptionalAttr<
                         I64ArrayAttr, "{0,0,0,0}">:$slice_enable_bitmap_mask
                 )>;

  def DotProductEngine
      : NamedDag<
            "dot_product_engine",
            (ins
                DefaultValuedOptionalAttr<I64Attr, "0">:$reg_indexer_base,
                DefaultValuedOptionalAttr<I64Attr, "0">:$acc_indexer_base,
                DefaultValuedOptionalAttr<I64Attr, "0">:$flits_per_input,
                DefaultValuedOptionalAttr<I64Attr, "0">:$feed_input_transpose,
                DefaultValuedOptionalAttr<I64Attr, "0">:$initial_shift_dim,
                DefaultValuedOptionalAttr<I64Attr, "0">:$shift_stride,
                DefaultValuedOptionalAttr<I64Attr, "0">:$pop_dim,
                DefaultValuedOptionalAttr<I64Attr, "0">:$shift_dim,
                DefaultValuedOptionalAttr<I64Attr, "0">:$channel_config,
                DefaultValuedOptionalAttr<I64Attr, "0">:$feed_data_type,
                DefaultValuedOptionalAttr<I64ArrayAttr,
                                          "{0,0,0,0,0,0,0,0}">:$initial_shift,
                DefaultValuedOptionalAttr<I64ArrayAttr,
                                          "{0,0,0,0,0,0,0,0}">:$iter_seq_limits,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0,0,0,0,0}">:$reg_indexer_strides,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0,0,0,0,0}">:$acc_indexer_strides,
                DefaultValuedOptionalAttr<I64Attr, "0">:$acc_limit,
                DefaultValuedOptionalAttr<I64Attr, "0">:$acc_cols,
                DefaultValuedOptionalAttr<I64Attr, "0">:$acc_reset,
                DefaultValuedOptionalAttr<I64Attr, "0">:$output_major,
                DefaultValuedOptionalAttr<I64Attr, "0">:$acc_init_value,
                DefaultValuedOptionalAttr<I64Attr, "0">:$mac_tree_operation,
                DefaultValuedOptionalAttr<I64Attr, "0">:$mac_tree_depth,
                DefaultValuedOptionalAttr<I64Attr, "0">:$mac_type,
                DefaultValuedOptionalAttr<I64Attr, "0">:$mac_rows,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_ieee_nan_multiplication,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_shift_rounding_mode,
                DefaultValuedOptionalAttr<I64Attr, "0">:$data_type,
                DefaultValuedOptionalAttr<I64Attr, "0">:$reg_read_log_size,
                DefaultValuedOptionalAttr<I64Attr, "0">:$reg_read_mode,
                DefaultValuedOptionalAttr<I64Attr, "0">:$reg_read_cache_mode
            )>;

  def VectorRouteUnit
      : NamedDag<
            "vector_route_unit",
            (ins
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$route_info_data_out_source,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$route_info_reduce_channel_out_source,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$route_info_reduce_unit_in_source,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$route_info_arithmetic_unit_in_source,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$route_info_valid_generator_mode,
                DefaultValuedOptionalAttr<I64Attr, "0">:$route_info_route_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$route_info_route_group_size,
                DefaultValuedOptionalAttr<I64Attr, "0">:$route_info_index_base,
                DefaultValuedOptionalAttr<I64ArrayAttr,
                                          "{0,0,0,0,0,0,0,0}">:$indexer_limits,
                DefaultValuedOptionalAttr<I64ArrayAttr,
                                          "{0,0,0,0,0,0,0,0}">:$indexer_strides,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr,
                    "{0,0,0,0,0,0,0,0}">:$valid_generator_lowered_limits,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr,
                    "{0,0,0,0,0,0,0,0}">:$valid_generator_lowered_strides,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0,0,0,0,"
                                  "0}">:$valid_generator_allocated_original_dim,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr,
                    "{0,0,0,0}">:$valid_generator_original_dim_partition_config,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr,
                    "{0,0,0,0}">:$valid_generator_original_dim_valid_count,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0}">:$valid_generator_slice_mask,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0}">:$valid_generator_slice_id_match,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$collect_compaction_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$compaction_mode_collect_compaction_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$compaction_mode_collect_compaction_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$compaction_mode_collect_compaction_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr, "0">:$cast_compaction_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$compaction_mode_cast_compaction_count
            )>;

  def VectorReduceUnit
      : NamedDag<
            "vector_reduce_unit",
            (ins
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$cluster_route_add_source,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$cluster_route_max_source,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$cluster_route_min_source,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$cluster_route_mul_source,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$cluster_route_cluster_source,
                DefaultValuedOptionalAttr<I64Attr, "0">:$mul_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr, "0">:$mul_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$mul_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$mul_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$mul_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr, "0">:$mul_control_rhs_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$mul_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$mul_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$mul_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$mul_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$mul_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$mul_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$mul_control_io_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$mul_control_io_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$mul_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$mul_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$mul_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<I64Attr, "0">:$add_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr, "0">:$add_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$add_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$add_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$add_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr, "0">:$add_control_rhs_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$add_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$add_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$add_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$add_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$add_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$add_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$add_control_io_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$add_control_io_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$add_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$add_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$add_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<I64Attr, "0">:$max_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr, "0">:$max_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$max_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$max_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$max_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr, "0">:$max_control_rhs_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$max_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$max_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$max_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$max_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$max_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$max_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$max_control_io_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$max_control_io_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$max_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$max_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$max_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<I64Attr, "0">:$min_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr, "0">:$min_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$min_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$min_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$min_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr, "0">:$min_control_rhs_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$min_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$min_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$min_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$min_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$min_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$min_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$min_control_io_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$min_control_io_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$min_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$min_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$min_data1_scalar_register_element2
            )>;

  def VectorArithmeticUnit
      : NamedDag<
            "vector_arithmetic_unit",
            (ins
                DefaultValuedOptionalAttr<I64Attr, "0">:$branch_mode_mode,
                DefaultValuedOptionalAttr<I64Attr, "0">:$branch_mode_format,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$branch_mode_compare_operation0,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$branch_mode_compare_operation1,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$branch_mode_compare_operation2,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$branch_mode_compare_operation3,
                DefaultValuedOptionalAttr<I64Attr, "0">:$branch_mode_group_size,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$branch_mode_branch_read_base,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$branch_mode_branch_read_limit,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$branch_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$branch_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$branch_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$branch_data1_scalar_register_element3,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$register_file_write_mode_branch_write_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$register_file_write_mode_branch_write_base,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$register_file_write_mode_branch_write_limit,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$register_file_write_mode_write_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$register_file_write_mode_write_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$register_file_write_mode_write_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_cluster_route_logic_and_source,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_cluster_route_logic_or_source,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_cluster_route_logic_xor_source,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_cluster_route_logic_left_shift_source,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_cluster_route_logic_right_shift_source,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_cluster_route_logic_cluster_source,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_and_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_and_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_and_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_and_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_and_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_and_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_and_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_and_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_and_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_and_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_and_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_and_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_and_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_and_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_and_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_and_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_and_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_or_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_or_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_or_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_or_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_or_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_or_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_or_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_or_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_or_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_or_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_or_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_or_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_or_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_or_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_or_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_or_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_or_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_xor_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_xor_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_xor_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_xor_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_xor_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$logic_xor_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_xor_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_xor_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_xor_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_xor_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_xor_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_xor_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_xor_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_xor_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_xor_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_xor_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_xor_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_left_shift_control_op_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_left_shift_control_arg_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_left_shift_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_left_shift_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_left_shift_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_left_shift_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_left_shift_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$logic_left_shift_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_left_shift_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$logic_left_shift_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_left_shift_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$logic_left_shift_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_left_shift_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$logic_left_shift_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$logic_left_shift_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$logic_left_shift_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$logic_left_shift_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_right_shift_control_op_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_right_shift_control_arg_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_right_shift_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_right_shift_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_right_shift_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_right_shift_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_right_shift_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$logic_right_shift_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_right_shift_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$logic_right_shift_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_right_shift_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$logic_right_shift_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$logic_right_shift_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$logic_right_shift_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$logic_right_shift_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$logic_right_shift_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$logic_right_shift_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_cluster_route_fxp_add_source,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_cluster_route_fxp_left_shift_source,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_cluster_route_fxp_mul_source,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_cluster_route_fxp_right_shift_source,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_cluster_route_fxp_cluster_source,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_add_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_add_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_add_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_add_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_add_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_add_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_add_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_add_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_add_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_add_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_add_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_add_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_add_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_add_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_add_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_add_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_add_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_left_shift_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_left_shift_control_arg_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_left_shift_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_left_shift_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_left_shift_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_left_shift_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_left_shift_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$fxp_left_shift_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_left_shift_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$fxp_left_shift_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_left_shift_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$fxp_left_shift_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_left_shift_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_left_shift_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$fxp_left_shift_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$fxp_left_shift_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$fxp_left_shift_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_mul_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_mul_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_mul_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_mul_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_mul_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_mul_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_mul_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_mul_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_mul_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_mul_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_mul_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_mul_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_mul_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_mul_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_mul_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_mul_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_mul_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fxp_right_shift_control_op_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_right_shift_control_arg_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_right_shift_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_right_shift_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_right_shift_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_right_shift_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_right_shift_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$fxp_right_shift_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_right_shift_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$fxp_right_shift_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_right_shift_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$fxp_right_shift_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_right_shift_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fxp_right_shift_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$fxp_right_shift_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$fxp_right_shift_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$fxp_right_shift_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_cluster_route_fp_fma_source,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_cluster_route_fp_fpu_source,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_cluster_route_fp_exp_source,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_cluster_route_fp_mul0_source,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_cluster_route_fp_mul1_source,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_cluster_route_fp_cluster_source,
                DefaultValuedOptionalAttr<I64Attr, "0">:$fp_fma_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_fma_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_fma_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_fma_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_fma_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_fma_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fma_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fma_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fma_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fma_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fma_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fma_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_fma_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fma_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fma_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fma_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fma_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$fp_fma_data1_secondary_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$fp_fma_data2_secondary_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr,
                    "0">:$fp_fma_data2_secondary_scalar_register_element2,
                DefaultValuedOptionalAttr<I64Attr, "0">:$fp_fpu_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_fpu_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_fpu_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_fpu_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_fpu_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_fpu_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fpu_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fpu_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fpu_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fpu_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fpu_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fpu_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_fpu_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fpu_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fpu_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fpu_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_fpu_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<I64Attr, "0">:$fp_exp_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_exp_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_exp_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_exp_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_exp_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_exp_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_exp_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_exp_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_exp_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_exp_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_exp_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_exp_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_exp_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_exp_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_mul0_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_mul0_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_mul0_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_mul0_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_mul0_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_mul0_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul0_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul0_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul0_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul0_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul0_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul0_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_mul0_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul0_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul0_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul0_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul0_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_mul1_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_mul1_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_mul1_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_mul1_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_mul1_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_mul1_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul1_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul1_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul1_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul1_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul1_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul1_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_mul1_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul1_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul1_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul1_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_mul1_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$reduce_layer_mode_reduce_data_path,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$reduce_layer_mode_reduce_rows,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$reduce_layer_mode_reduce_tree_depth,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$reduce_layer_mode_acc_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$reduce_layer_mode_acc_indexer_proceed,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$reduce_layer_mode_fxp_shift_rounding_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$reduce_layer_mode_reduce_row0_op_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$reduce_layer_mode_reduce_row1_op_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$reduce_layer_mode_accumulation_limit,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$reduce_layer_mode_acc_indexer_base,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$reduce_layer_acc_init_reduce_row0_acc_init,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$reduce_layer_acc_init_reduce_row1_acc_init,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr,
                    "{0,0,0,0,0,0,0,0}">:$reduce_layer_acc_indexer_limits,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr,
                    "{0,0,0,0,0,0,0,0}">:$reduce_layer_acc_indexer_strides,
                DefaultValuedOptionalAttr<I64Attr, "0">:$fp_div_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_div_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_div_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_div_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_div_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_div_control_rf_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_div_control_acc_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_div_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_div_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_div_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_div_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_div_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_div_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_div_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_div_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$fp_div_control_acc_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_div_control_acc_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_div_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_div_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$fp_div_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$float_adapter_fxp_to_fp_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$float_adapter_fxp_to_fp_int_width,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$float_adapter_fxp_to_fp_round_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$float_adapter_fp_to_fxp_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$float_adapter_fp_to_fxp_int_width,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$float_adapter_fp_to_fxp_round_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$float_adapter_split_layer_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$float_adapter_concat_layer_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_cluster_route_clip_add_source,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_cluster_route_clip_max_source,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_cluster_route_clip_min_source,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_cluster_route_clip_cluster_source,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_add_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_add_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_add_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_add_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_add_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_add_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_add_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_add_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_add_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_add_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_add_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_add_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_add_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_add_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_add_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_add_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_add_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_max_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_max_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_max_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_max_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_max_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_max_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_max_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_max_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_max_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_max_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_max_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_max_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_max_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_max_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_max_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_max_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_max_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_min_control_op_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_min_control_arg_mode,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_min_control_reg0_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_min_control_reg1_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_min_control_reg2_cmp_op,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$clip_min_control_rf_cmp_op,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_min_control_reg0_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_min_control_reg0_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_min_control_reg1_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_min_control_reg1_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_min_control_reg2_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_min_control_reg2_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_min_control_rf_execution_id,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_min_control_rf_execution_id_mask,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_min_data0_scalar_register_element0,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_min_data0_scalar_register_element1,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$clip_min_data1_scalar_register_element2,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$alloc_indexer_read_indexer0_module,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$alloc_indexer_read_indexer1_module,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$alloc_indexer_read_indexer2_module,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$alloc_indexer_read_indexer3_module,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$alloc_indexer_operand_indexer_module,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$alloc_indexer_write_indexer_module,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$read_indexer0_operation,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$operation_read_indexer0_proceed,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$operation_read_indexer0_element_size,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$read_indexer1_operation,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$operation_read_indexer1_proceed,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$operation_read_indexer1_element_size,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$read_indexer2_operation,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$operation_read_indexer2_proceed,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$operation_read_indexer2_element_size,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$read_indexer3_operation,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$operation_read_indexer3_proceed,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$operation_read_indexer3_element_size,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$operand_indexer_operation,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$operation_operand_indexer_proceed,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$operation_operand_indexer_update_mode,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$operation_operand_indexer_element_size,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$write_indexer_operation,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$indexer_base0_read_indexer0_base,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$indexer_base0_read_indexer1_base,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$indexer_base0_read_indexer2_base,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$indexer_base0_read_indexer3_base,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$indexer_base1_operand_indexer_base,
                DefaultValuedOptionalAttr<
                    I64Attr, "0">:$indexer_base1_write_indexer_base,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0,0,0,0,0}">:$read_indexer0_limits,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0,0,0,0,0}">:$read_indexer0_strides,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0,0,0,0,0}">:$read_indexer1_limits,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0,0,0,0,0}">:$read_indexer1_strides,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0,0,0,0,0}">:$read_indexer2_limits,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0,0,0,0,0}">:$read_indexer2_strides,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0,0,0,0,0}">:$read_indexer3_limits,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0,0,0,0,0}">:$read_indexer3_strides,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0,0,0,0,0}">:$operand_indexer_limits,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0,0,0,0,0}">:$operand_indexer_strides,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0,0,0,0,0}">:$write_indexer_limits,
                DefaultValuedOptionalAttr<
                    I64ArrayAttr, "{0,0,0,0,0,0,0,0}">:$write_indexer_strides
            )>;

  def TransposeEngine
      : NamedDag<
            "transpose_engine",
            (ins
                DefaultValuedOptionalAttr<I64Attr, "0">:$fetch_in_cols,
                DefaultValuedOptionalAttr<I64Attr, "0">:$fetch_in_rows,
                DefaultValuedOptionalAttr<I64Attr, "0">:$fetch_out_rows,
                DefaultValuedOptionalAttr<I64Attr, "0">:$data_type,
                DefaultValuedOptionalAttr<I64Attr, "0">:$fetch_in_width_shift
            )>;

  def MainDataPathUnit
      : NamedDag<"main_data_path_unit",
                 (ins
                     DefaultValuedOptionalAttr<I64Attr, "0">:$main_context,
                     DefaultValuedOptionalAttr<I64Attr, "0">:$channel_config
                 )>;

  def SubFetchUnit
      : NamedDag<
            "sub_fetch_unit",
            (ins
                DefaultValuedOptionalAttr<I64Attr, "0">:$base,
                DefaultValuedOptionalAttr<I64Attr, "0">:$type_conversion,
                DefaultValuedOptionalAttr<I64Attr, "0">:$num_zero_points,
                DefaultValuedOptionalAttr<I64Attr, "0">:$zero_point0,
                DefaultValuedOptionalAttr<I64Attr, "0">:$zero_point1,
                DefaultValuedOptionalAttr<I64ArrayAttr,
                                          "{0,0,0,0,0,0,0,0}">:$limits,
                DefaultValuedOptionalAttr<I64ArrayAttr,
                                          "{0,0,0,0,0,0,0,0}">:$strides,
                DefaultValuedOptionalAttr<I64Attr, "0">:$flit_count,
                DefaultValuedOptionalAttr<I64Attr, "0">:$words_per_packet,
                DefaultValuedOptionalAttr<I64Attr, "0">:$topology,
                DefaultValuedOptionalAttr<I64Attr, "0">:$outer_slice_log_size,
                DefaultValuedOptionalAttr<I64Attr, "0">:$outer_dim0_log_size,
                DefaultValuedOptionalAttr<I64Attr, "0">:$outer_dim1_log_size,
                DefaultValuedOptionalAttr<I64Attr, "0">:$outer_dim0_chunk_size,
                DefaultValuedOptionalAttr<I64Attr, "0">:$outer_dim1_chunk_size,
                DefaultValuedOptionalAttr<I64ArrayAttr,
                                          "{0,0,0,0}">:$custom_snoop_bitmap_mask
            )>;

  def SubCommitUnit
      : NamedDag<
            "sub_commit_unit",
            (ins
                DefaultValuedOptionalAttr<I64Attr, "0">:$mode,
                DefaultValuedOptionalAttr<I64Attr, "0">:$packet_valid_count,
                DefaultValuedOptionalAttr<I64Attr, "0">:$base,
                DefaultValuedOptionalAttr<I64Attr, "0">:$commit_in_size,
                DefaultValuedOptionalAttr<I64Attr, "0">:$commit_data,
                DefaultValuedOptionalAttr<I64ArrayAttr,
                                          "{0,0,0,0,0,0,0,0}">:$limits,
                DefaultValuedOptionalAttr<I64ArrayAttr,
                                          "{0,0,0,0,0,0,0,0}">:$strides,
                DefaultValuedOptionalAttr<I64ArrayAttr,
                                          "{0,0,0,0}">:$slice_enable_bitmap_mask
            )>;

  def TensorRegisterFile
      : NamedDag<
            "tensor_register_file",
            (ins
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$write_interleaving_flit_count,
                DefaultValuedOptionalAttr<I64Attr, "0">:$write_mode,
                DefaultValuedOptionalAttr<I64Attr, "0">:$write_mac_rows,
                DefaultValuedOptionalAttr<I64Attr, "0">:$write_skip_flit_count,
                DefaultValuedOptionalAttr<I64Attr, "0">:$write_row_base,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$write_mac_row_interleaving,
                DefaultValuedOptionalAttr<I64Attr, "0">:$write_row_count,
                DefaultValuedOptionalAttr<I64Attr, "0">:$write_flits_per_period,
                DefaultValuedOptionalAttr<I64Attr,
                                          "0">:$write_valid_flits_per_period
            )>;

  def VectorRegisterFile
      : NamedDag<
            "vector_register_file",
            (ins
                DefaultValuedOptionalAttr<I64Attr, "0">:$write_row_base,
                DefaultValuedOptionalAttr<I64Attr, "0">:$write_row_count,
                DefaultValuedOptionalAttr<I64Attr, "0">:$write_skip_flit_count,
                DefaultValuedOptionalAttr<I64Attr, "0">:$write_row_stride
            )>;

  def RegisterConfigUnit
      : NamedDag<"register_config_unit",
                 (ins
                     DefaultValuedOptionalAttr<I64Attr, "0">:$base,
                     DefaultValuedOptionalAttr<I64Attr, "0">:$size,
                     DefaultValuedOptionalAttr<I64Attr, "0">:$access_type,
                     DefaultValuedOptionalAttr<I64Attr, "0">:$words_per_input,
                     DefaultValuedOptionalAttr<I64Attr, "0">:$data_offset
                 )>;

  def SubDataPathUnit
      : NamedDag<"sub_data_path_unit",
                 (ins
                     DefaultValuedOptionalAttr<I64Attr,
                                               "0">:$data_path_route_sub_context
                 )>;
}

def DmaDescriptor
    : NamedDag<"dma_descriptor", (ins
                                     I64Attr:$opcode,
                                     I64Attr:$source_base,
                                     I64Attr:$destination_base,
                                     I64ArrayAttr:$source_limits,
                                     I64ArrayAttr:$source_strides,
                                     I64ArrayAttr:$destination_limits,
                                     I64ArrayAttr:$destination_strides
                                 )>;

//===----------------------------------------------------------------------===//
// Task operations.
//===----------------------------------------------------------------------===//

class TaskOp<string mnemonic, list<Trait> traits = []>
    : Furiosa_Op<mnemonic, traits> {}

class TaskStaticSfrOp<string mnemonic, list<Trait> traits = []>
    : TaskOp<mnemonic, traits> {
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

class TaskSfrOp<string mnemonic, list<Trait> traits = []>
    : TaskOp<mnemonic, traits> {
  let results = (outs
      SfrType:$sfr
  );
  let assemblyFormat = "attr-dict";
}

foreach nd = Sfr in {
  def TaskStaticSfr#!cast<string>(nd)#Op
      : TaskStaticSfrOp<"task.static.sfr."#nd.mnemonic> {
    let summary = "Static special function register for "#!cast<string>(nd);
    let description = [{}];
    let arguments = !con((ins
                             I64Attr:$sfr_addr
                         ),
                         nd.data);
  }

  def TaskSfr#!cast<string>(nd)#Op : TaskSfrOp<"task.sfr."#nd.mnemonic> {
    let summary = "Special function register for "#!cast<string>(nd);
    let description = [{}];
    let arguments = nd.data;
  }
}

def TaskStaticDmaDescriptorOp : TaskOp<"task.static.dma_descriptor"> {
  let summary = "Static direct memory access descriptor";
  let description = [{}];
  let arguments = !con((ins
                           I64Attr:$desc_addr
                       ),
                       DmaDescriptor.data);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def TaskDmaDescriptorOp : TaskOp<"task.dma_descriptor"> {
  let summary = "Direct memory access descriptor";
  let description = [{}];
  let arguments = DmaDescriptor.data;
  let results = (outs
      DmaDescriptorType:$desc
  );
  let assemblyFormat = "attr-dict";
}

def TaskMtosfrOp : TaskOp<"task.mtosfr"> {
  let summary = "Dynamic memory to special register file";
  let description = [{}];
  let arguments = (ins
      SfrType:$sfr,
      I64Attr:$sfr_address
  );
  let results = (outs);
  let assemblyFormat = "$sfr attr-dict";
}

def TaskDmawOp : TaskOp<"task.dmaw"> {
  let summary = "Dynamic direct memory access wide";
  let description = [{}];
  let arguments = (ins
      DmaDescriptorType:$desc,
      I64Attr:$dma_tag_id,
      I1Attr:$profile,
      I64Attr:$profile_id
  );
  let results = (outs);
  let assemblyFormat = "$desc attr-dict";
}

#endif // FURIOSA_TASK_OPS_TD
