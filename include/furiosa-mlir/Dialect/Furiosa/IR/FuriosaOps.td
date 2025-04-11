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

//===----------------------------------------------------------------------===//
// Tensor unit commands.
//===----------------------------------------------------------------------===//

def Furiosa_ItosfrOp : Furiosa_TensorUnitCommand<"itosfr"> {
  let summary = "Immediate to special register file";
  let description = [{}];
  let arguments = (ins
      I32Attr:$value,
      I64Attr:$sfr_address,
      I64Attr:$size
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_RtosfrOp : Furiosa_TensorUnitCommand<"rtosfr"> {
  let summary = "Register to special register file";
  let description = [{}];
  let arguments = (ins
      I64Attr:$value,
      I64Attr:$sfr_address,
      I64Attr:$size
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_RtosfriOp : Furiosa_TensorUnitCommand<"rtosfri"> {
  let summary = "Register to special register file immediate";
  let description = [{}];
  let arguments = (ins
      I32Attr:$sfr_address,
      I32Attr:$log_size,
      I64Attr:$value
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_MtosfrOp : Furiosa_TensorUnitCommand<"mtosfr"> {
  let summary = "Memory to special register file";
  let description = [{}];
  let arguments = (ins
      I64Attr:$spm_address,
      I64Attr:$size,
      I64Attr:$sfr_address
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_StosfrOp : Furiosa_TensorUnitCommand<"stosfr"> {
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
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_SfrtosOp : Furiosa_TensorUnitCommand<"sfrtos"> {
  let summary = "special register file to SRAM";
  let description = [{}];
  let arguments = (ins
      I64Attr:$commit_base,
      I64Attr:$commit_limit,
      I64Attr:$sfr_address
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_StallOp : Furiosa_TensorUnitCommand<"stall"> {
  let summary = "Stall";
  let description = [{}];
  let arguments = (ins
      I64Attr:$cycle
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_ItosOp : Furiosa_TensorUnitCommand<"itos"> {
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
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_ItosiOp : Furiosa_TensorUnitCommand<"itosi"> {
  let summary = "Immediate to SRAM immediate";
  let description = [{}];
  let arguments = (ins
      I64Attr:$address_begin,
      I64Attr:$address_end,
      I64Attr:$value
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_StosOp : Furiosa_TensorUnitCommand<"stos"> {
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
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_StotabOp : Furiosa_TensorUnitCommand<"stotab"> {
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
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_StotrfOp : Furiosa_TensorUnitCommand<"stotrf"> {
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
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_StovrfOp : Furiosa_TensorUnitCommand<"stovrf"> {
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
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_ExecutionOp : Furiosa_TensorUnitCommand<"exec"> {
  let summary = "Tensor unit execution";
  let description = [{}];
  let arguments = (ins
      I32Attr:$subunit_bitmap,
      I1Attr:$context_id,
      I1Attr:$target_context
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_WaitOp : Furiosa_TensorUnitCommand<"wait"> {
  let summary = "Wait";
  let description = [{}];
  let arguments = (ins
      I32Attr:$dma_tag_id,
      I1Attr:$type,
      I1Attr:$target_context
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_WaitiOp : Furiosa_TensorUnitCommand<"waiti"> {
  let summary = "Wait interrupt";
  let description = [{}];
  let arguments = (ins
      I32Attr:$dma_tag_id,
      I1Attr:$type,
      I1Attr:$target_context
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_InterruptOp : Furiosa_TensorUnitCommand<"interrupt"> {
  let summary = "SRAM to tensor register file";
  let description = [{}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_DmaOp : Furiosa_TensorUnitCommand<"dma"> {
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
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_Dma1Op : Furiosa_TensorUnitCommand<"dma1"> {
  let summary = "Direct memory access 1";
  let description = [{}];
  let arguments = (ins
      I64Attr:$desc_addr,
      I64Attr:$pe_valid_bitmap,
      I64Attr:$dma_tag_id,
      I1Attr:$profile,
      I64Attr:$profile_id
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_DmawOp : Furiosa_TensorUnitCommand<"dmaw"> {
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
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_ProfileOp : Furiosa_TensorUnitCommand<"profile"> {
  let summary = "Profile";
  let description = [{}];
  let arguments = (ins
      I64Attr:$profile_id
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_ProfileiOp : Furiosa_TensorUnitCommand<"profilei"> {
  let summary = "Profile immediate";
  let description = [{}];
  let arguments = (ins
      I64Attr:$profile_id
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_PrflushOp : Furiosa_TensorUnitCommand<"prflush"> {
  let summary = "Profile flush";
  let description = [{}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Task binary helpers.
//===----------------------------------------------------------------------===//

def Furiosa_SfrOp : Furiosa_TensorUnitCommand<"sfr"> {
  let summary = "Special function register";
  let description = [{}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_DmaDescriptorOp : Furiosa_TensorUnitCommand<"dma_descriptor"> {
  let summary = "Direct memory access descriptor";
  let description = [{}];
  let arguments = (ins
      I64Attr:$desc_addr,
      I64Attr:$opcode,
      I64Attr:$source_base,
      I64Attr:$destination_base,
      ArrayAttr:$source_limit,
      ArrayAttr:$source_stride,
      ArrayAttr:$destination_limit,
      ArrayAttr:$destination_stride
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

#endif // FURIOSA_OPS_TD
