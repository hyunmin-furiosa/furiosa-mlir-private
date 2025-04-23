//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef FURIOSA_OPS_TD
#define FURIOSA_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaBase.td"
include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaTypes.td"

//===----------------------------------------------------------------------===//
// Tensor unit commands.
//===----------------------------------------------------------------------===//

class TensorUnitCommandOp<string mnemonic, list<Trait> traits = []>
    : Furiosa_Op<mnemonic, traits> {
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def TucItosfrOp : TensorUnitCommandOp<"tuc.itosfr"> {
  let summary = "Immediate to special register file";
  let description = [{}];
  let arguments = (ins
      I32Attr:$value,
      I64Attr:$sfr_address,
      I64Attr:$size
  );
}

def TucRtosfrOp : TensorUnitCommandOp<"tuc.rtosfr"> {
  let summary = "Register to special register file";
  let description = [{}];
  let arguments = (ins
      I64Attr:$value,
      I64Attr:$sfr_address,
      I64Attr:$size
  );
}

def TucRtosfriOp : TensorUnitCommandOp<"tuc.rtosfri"> {
  let summary = "Register to special register file immediate";
  let description = [{}];
  let arguments = (ins
      I32Attr:$sfr_address,
      I32Attr:$log_size,
      I64Attr:$value
  );
}

def TucMtosfrOp : TensorUnitCommandOp<"tuc.mtosfr"> {
  let summary = "Memory to special register file";
  let description = [{}];
  let arguments = (ins
      I64Attr:$spm_address,
      I64Attr:$size,
      I64Attr:$sfr_address
  );
}

def TucStosfrOp : TensorUnitCommandOp<"tuc.stosfr"> {
  let summary = "SRAM to special register file";
  let description = [{}];
  let arguments = (ins
      I64Attr:$fetch_base,
      I64Attr:$fetch_size,
      I64Attr:$sfr_address,
      I64Attr:$topology,
      I64Attr:$slice_log_size,
      I64Attr:$dim0_log_size,
      I64Attr:$dim1_log_size,
      I64Attr:$data_offset,
      I64Attr:$size,
      I64Attr:$words_per_packet
  );
}

def TucSfrtosOp : TensorUnitCommandOp<"tuc.sfrtos"> {
  let summary = "special register file to SRAM";
  let description = [{}];
  let arguments = (ins
      I64Attr:$commit_base,
      I64Attr:$commit_limit,
      I64Attr:$sfr_address
  );
}

def TucStallOp : TensorUnitCommandOp<"tuc.stall"> {
  let summary = "Stall";
  let description = [{}];
  let arguments = (ins
      I64Attr:$cycle
  );
}

def TucItosOp : TensorUnitCommandOp<"tuc.itos"> {
  let summary = "Immediate to SRAM";
  let description = [{}];
  let arguments = (ins
      I64Attr:$address_begin,
      I64Attr:$address_end,
      I64Attr:$value,
      I64Attr:$dim1_log_size,
      I64Attr:$limit1,
      I64Attr:$stride1
  );
}

def TucItosiOp : TensorUnitCommandOp<"tuc.itosi"> {
  let summary = "Immediate to SRAM immediate";
  let description = [{}];
  let arguments = (ins
      I64Attr:$address_begin,
      I64Attr:$address_end,
      I64Attr:$value
  );
}

def TucStosOp : TensorUnitCommandOp<"tuc.stos"> {
  let summary = "SRAM to SRAM";
  let description = [{}];
  let arguments = (ins
      I64Attr:$address_begin,
      I64Attr:$address_end,
      I64Attr:$destination_begin,
      I64Attr:$slice_log_size,
      I64Attr:$dim1_log_size,
      I64Attr:$words_per_packet
  );
}

def TucStotabOp : TensorUnitCommandOp<"tuc.stotab"> {
  let summary = "SRAM to table";
  let description = [{}];
  let arguments = (ins
      I64Attr:$fetch_base,
      I64Attr:$fetch_limit,
      I64Attr:$tables,
      I64Attr:$commit_base,
      I64Attr:$topology,
      I64Attr:$slice_log_size,
      I64Attr:$dim0_log_size,
      I64Attr:$dim1_log_size,
      I64Attr:$words_per_packet
  );
}

def TucStotrfOp : TensorUnitCommandOp<"tuc.stotrf"> {
  let summary = "SRAM to tensor register file";
  let description = [{}];
  let arguments = (ins
      I64Attr:$fetch_base,
      I64Attr:$fetch_limit,
      I64Attr:$type_conversion,
      I64Attr:$write_mode,
      I64Attr:$zeropoint,
      I64Attr:$topology,
      I64Attr:$slice_log_size,
      I64Attr:$dim0_log_size,
      I64Attr:$dim1_log_size,
      I64Attr:$flits_per_packet,
      I64Attr:$dim0_chunk_size,
      I64Attr:$skip_flit_count,
      I64Attr:$write_row_base,
      I64Attr:$write_row_count,
      I64Attr:$write_mac_row,
      I64Attr:$flits_per_period,
      I64Attr:$valid_flits_per_period
  );
}

def TucStovrfOp : TensorUnitCommandOp<"tuc.stovrf"> {
  let summary = "SRAM to vector register file";
  let description = [{}];
  let arguments = (ins
      I64Attr:$fetch_base,
      I64Attr:$fetch_limit,
      I64Attr:$type_conversion,
      I64Attr:$topology,
      I64Attr:$slice_log_size,
      I64Attr:$dim0_log_size,
      I64Attr:$dim1_log_size,
      I64Attr:$words_per_packet,
      I64Attr:$skip_flit_count,
      I64Attr:$write_row_base,
      I64Attr:$write_row_count,
      I64Attr:$write_row_stride
  );
}

def TucExecutionOp : TensorUnitCommandOp<"tuc.exec"> {
  let summary = "Tensor unit execution";
  let description = [{}];
  let arguments = (ins
      I32Attr:$subunit_bitmap,
      I1Attr:$context_id,
      I1Attr:$target_context
  );
}

def TucWaitOp : TensorUnitCommandOp<"tuc.wait"> {
  let summary = "Wait";
  let description = [{}];
  let arguments = (ins
      I32Attr:$dma_tag_id,
      I1Attr:$type,
      I1Attr:$target_context
  );
}

def TucWaitiOp : TensorUnitCommandOp<"tuc.waiti"> {
  let summary = "Wait interrupt";
  let description = [{}];
  let arguments = (ins
      I32Attr:$dma_tag_id,
      I1Attr:$type,
      I1Attr:$target_context
  );
}

def TucInterruptOp : TensorUnitCommandOp<"tuc.interrupt"> {
  let summary = "SRAM to tensor register file";
  let description = [{}];
  let arguments = (ins);
}

def TucDmaOp : TensorUnitCommandOp<"tuc.dma"> {
  let summary = "Direct memory access";
  let description = [{}];
  let arguments = (ins
      I64Attr:$pe0_desc_addr,
      I64Attr:$pe1_desc_addr,
      I64Attr:$pe2_desc_addr,
      I64Attr:$pe3_desc_addr,
      I64Attr:$dma_tag_id,
      I1Attr:$profile,
      I64Attr:$profile_id
  );
}

def TucDma1Op : TensorUnitCommandOp<"tuc.dma1"> {
  let summary = "Direct memory access 1";
  let description = [{}];
  let arguments = (ins
      I64Attr:$desc_addr,
      I64Attr:$pe_valid_bitmap,
      I64Attr:$dma_tag_id,
      I1Attr:$profile,
      I64Attr:$profile_id
  );
}

def TucDmawOp : TensorUnitCommandOp<"tuc.dmaw"> {
  let summary = "Direct memory access wide";
  let description = [{}];
  let arguments = (ins
      I64Attr:$pe0_desc_addr,
      I64Attr:$pe1_desc_addr,
      I64Attr:$pe2_desc_addr,
      I64Attr:$pe3_desc_addr,
      I64Attr:$dma_tag_id,
      I1Attr:$profile,
      I64Attr:$profile_id
  );
}

def TucProfileOp : TensorUnitCommandOp<"tuc.profile"> {
  let summary = "Profile";
  let description = [{}];
  let arguments = (ins
      I64Attr:$profile_id
  );
}

def TucProfileiOp : TensorUnitCommandOp<"tuc.profilei"> {
  let summary = "Profile immediate";
  let description = [{}];
  let arguments = (ins
      I64Attr:$profile_id
  );
}

def TucPrflushOp : TensorUnitCommandOp<"tuc.prflush"> {
  let summary = "Profile flush";
  let description = [{}];
  let arguments = (ins);
}

//===----------------------------------------------------------------------===//
// Shared dag type definitions
//===----------------------------------------------------------------------===//

class NamedDag<string m, dag d> {
  string mnemonic = m;
  dag data = d;
}

defset list<NamedDag> Sfr = {
  def SfrMainFetchUnit
      : NamedDag<"main_fetch_unit",
                 (ins
                     I64Attr:$fetch_mode,
                     I64Attr:$num_zero_points,
                     I64Attr:$zero_point0,
                     I64Attr:$zero_point1,
                     I64Attr:$table_entry_size,
                     I64Attr:$tables,
                     I64Attr:$indirect_base,
                     I64Attr:$indirect_dim,
                     I64Attr:$table_base_mode,
                     I64Attr:$indirect_pointer_size,
                     I64Attr:$zeropoint_tail_mode,
                     I64Attr:$last_dim_pad_value,
                     I64Attr:$last_dim,
                     I64Attr:$pad_order,
                     I64Attr:$last_dim_rightmost_valid_count_dim,
                     I64Attr:$last_dim_left_pad_count,
                     I64Attr:$type_conversion,
                     I64Attr:$last_dim_left_pad_mode,
                     I64Attr:$zeropoint_dims,
                     ArrayAttr:$last_dim_rightmost_valid_count,
                     I64Attr:$base,
                     I64Attr:$fetch_size,
                     ArrayAttr:$limits,
                     ArrayAttr:$strides,
                     I64Attr:$flit_count,
                     I64Attr:$words_per_packet,
                     I64Attr:$zeropoint_fetch_limit,
                     I64Attr:$topology,
                     I64Attr:$channel_config,
                     I64Attr:$outer_slice_log_size,
                     I64Attr:$outer_dim0_log_size,
                     I64Attr:$outer_dim1_log_size,
                     I64Attr:$outer_dim0_chunk_size,
                     I64Attr:$outer_dim1_chunk_size,
                     ArrayAttr:$custom_snoop_bitmap_mask
                 )>;

  def MainCommitUnit
      : NamedDag<"main_commit_unit", (ins
                                         I64Attr:$type_conversion,
                                         I64Attr:$base,
                                         I64Attr:$commit_in_size,
                                         I64Attr:$commit_size,
                                         ArrayAttr:$limits,
                                         ArrayAttr:$strides,
                                         ArrayAttr:$slice_enable_bitmap_mask
                                     )>;

  def DotProductEngine
      : NamedDag<"dot_product_engine", (ins
                                           I64Attr:$reg_indexer_base,
                                           I64Attr:$acc_indexer_base,
                                           I64Attr:$flits_per_input,
                                           I64Attr:$feed_input_transpose,
                                           I64Attr:$initial_shift_dim,
                                           I64Attr:$shift_stride,
                                           I64Attr:$pop_dim,
                                           I64Attr:$shift_dim,
                                           I64Attr:$channel_config,
                                           I64Attr:$feed_data_type,
                                           ArrayAttr:$initial_shift,
                                           ArrayAttr:$iter_seq_limits,
                                           ArrayAttr:$reg_indexer_strides,
                                           ArrayAttr:$acc_indexer_strides,
                                           I64Attr:$acc_limit,
                                           I64Attr:$acc_cols,
                                           I64Attr:$acc_reset,
                                           I64Attr:$output_major,
                                           I64Attr:$acc_init_value,
                                           I64Attr:$mac_tree_operation,
                                           I64Attr:$mac_tree_depth,
                                           I64Attr:$mac_type,
                                           I64Attr:$mac_rows,
                                           I64Attr:$fp_ieee_nan_multiplication,
                                           I64Attr:$fxp_shift_rounding_mode,
                                           I64Attr:$data_type,
                                           I64Attr:$reg_read_log_size,
                                           I64Attr:$reg_read_mode,
                                           I64Attr:$reg_read_cache_mode
                                       )>;

  def TransposeEngine
      : NamedDag<"transpose_engine", (ins
                                         I64Attr:$fetch_in_cols,
                                         I64Attr:$fetch_in_rows,
                                         I64Attr:$fetch_out_rows,
                                         I64Attr:$data_type,
                                         I64Attr:$fetch_in_width_shift
                                     )>;

  def MainDataPathUnit
      : NamedDag<"main_data_path_unit", (ins
                                            I64Attr:$main_context,
                                            I64Attr:$channel_config
                                        )>;

  def SubFetchUnit
      : NamedDag<"sub_fetch_unit", (ins
                                       I64Attr:$base,
                                       I64Attr:$type_conversion,
                                       I64Attr:$num_zero_points,
                                       I64Attr:$zero_point0,
                                       I64Attr:$zero_point1,
                                       ArrayAttr:$limits,
                                       ArrayAttr:$strides,
                                       I64Attr:$flit_count,
                                       I64Attr:$words_per_packet,
                                       I64Attr:$topology,
                                       I64Attr:$outer_slice_log_size,
                                       I64Attr:$outer_dim0_log_size,
                                       I64Attr:$outer_dim1_log_size,
                                       I64Attr:$outer_dim0_chunk_size,
                                       I64Attr:$outer_dim1_chunk_size,
                                       ArrayAttr:$custom_snoop_bitmap_mask
                                   )>;

  def SubCommitUnit
      : NamedDag<"sub_commit_unit", (ins
                                        I64Attr:$mode,
                                        I64Attr:$packet_valid_count,
                                        I64Attr:$base,
                                        I64Attr:$commit_in_size,
                                        I64Attr:$commit_data,
                                        ArrayAttr:$limits,
                                        ArrayAttr:$strides,
                                        ArrayAttr:$slice_enable_bitmap_mask
                                    )>;

  def TensorRegisterFile : NamedDag<"tensor_register_file",
                                    (ins
                                        I64Attr:$write_interleaving_flit_count,
                                        I64Attr:$write_mode,
                                        I64Attr:$write_mac_rows,
                                        I64Attr:$write_skip_flit_count,
                                        I64Attr:$write_row_base,
                                        I64Attr:$write_mac_row_interleaving,
                                        I64Attr:$write_row_count,
                                        I64Attr:$write_flits_per_period,
                                        I64Attr:$write_valid_flits_per_period
                                    )>;

  def VectorRegisterFile
      : NamedDag<"vector_register_file", (ins
                                             I64Attr:$write_row_base,
                                             I64Attr:$write_row_count,
                                             I64Attr:$write_skip_flit_count,
                                             I64Attr:$write_row_stride
                                         )>;

  def RegisterConfigUnit
      : NamedDag<"register_config_unit", (ins
                                             I64Attr:$base,
                                             I64Attr:$size,
                                             I64Attr:$access_type,
                                             I64Attr:$words_per_input,
                                             I64Attr:$data_offset
                                         )>;

  def SubDataPathUnit : NamedDag<"sub_data_path_unit", (ins
                                                           I64Attr:$route
                                                       )>;
}

def DmaDescriptor
    : NamedDag<"dma_descriptor", (ins
                                     I64Attr:$opcode,
                                     I64Attr:$source_base,
                                     I64Attr:$destination_base,
                                     ArrayAttr:$source_limits,
                                     ArrayAttr:$source_strides,
                                     ArrayAttr:$destination_limits,
                                     ArrayAttr:$destination_strides
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

#endif // FURIOSA_OPS_TD
