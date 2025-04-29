//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef FURIOSA_TUC_OPS_TD
#define FURIOSA_TUC_OPS_TD

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
    : Furiosa_Op<"tuc."#mnemonic, traits> {
  let cppNamespace = Furiosa_Dialect.cppNamespace#"::tuc";
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def ItosfrOp : TensorUnitCommandOp<"itosfr"> {
  let summary = "Immediate to special register file";
  let description = [{}];
  let arguments = (ins
      I32Attr:$value,
      I64Attr:$sfr_address,
      I64Attr:$size
  );
}

def RtosfrOp : TensorUnitCommandOp<"rtosfr"> {
  let summary = "Register to special register file";
  let description = [{}];
  let arguments = (ins
      I64Attr:$value,
      I64Attr:$sfr_address,
      I64Attr:$size
  );
}

def RtosfriOp : TensorUnitCommandOp<"rtosfri"> {
  let summary = "Register to special register file immediate";
  let description = [{}];
  let arguments = (ins
      I32Attr:$sfr_address,
      I32Attr:$log_size,
      I64Attr:$value
  );
}

def MtosfrOp : TensorUnitCommandOp<"mtosfr"> {
  let summary = "Memory to special register file";
  let description = [{}];
  let arguments = (ins
      I64Attr:$spm_address,
      I64Attr:$size,
      I64Attr:$sfr_address
  );
}

def StosfrOp : TensorUnitCommandOp<"stosfr"> {
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

def SfrtosOp : TensorUnitCommandOp<"sfrtos"> {
  let summary = "special register file to SRAM";
  let description = [{}];
  let arguments = (ins
      I64Attr:$commit_base,
      I64Attr:$commit_limit,
      I64Attr:$sfr_address
  );
}

def StallOp : TensorUnitCommandOp<"stall"> {
  let summary = "Stall";
  let description = [{}];
  let arguments = (ins
      I64Attr:$cycle
  );
}

def ItosOp : TensorUnitCommandOp<"itos"> {
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

def ItosiOp : TensorUnitCommandOp<"itosi"> {
  let summary = "Immediate to SRAM immediate";
  let description = [{}];
  let arguments = (ins
      I64Attr:$address_begin,
      I64Attr:$address_end,
      I64Attr:$value
  );
}

def StosOp : TensorUnitCommandOp<"stos"> {
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

def StotabOp : TensorUnitCommandOp<"stotab"> {
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

def StotrfOp : TensorUnitCommandOp<"stotrf"> {
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

def StovrfOp : TensorUnitCommandOp<"stovrf"> {
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

def ExecutionOp : TensorUnitCommandOp<"exec"> {
  let summary = "Tensor unit execution";
  let description = [{}];
  let arguments = (ins
      I32Attr:$subunit_bitmap,
      I1Attr:$context_id,
      I1Attr:$target_context
  );
}

def WaitOp : TensorUnitCommandOp<"wait"> {
  let summary = "Wait";
  let description = [{}];
  let arguments = (ins
      I32Attr:$dma_tag_id,
      I1Attr:$type,
      I1Attr:$target_context
  );
}

def WaitiOp : TensorUnitCommandOp<"waiti"> {
  let summary = "Wait interrupt";
  let description = [{}];
  let arguments = (ins
      I32Attr:$dma_tag_id,
      I1Attr:$type,
      I1Attr:$target_context
  );
}

def InterruptOp : TensorUnitCommandOp<"interrupt"> {
  let summary = "SRAM to tensor register file";
  let description = [{}];
  let arguments = (ins);
}

def DmaOp : TensorUnitCommandOp<"dma"> {
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

def Dma1Op : TensorUnitCommandOp<"dma1"> {
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

def DmawOp : TensorUnitCommandOp<"dmaw"> {
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

def ProfileOp : TensorUnitCommandOp<"profile"> {
  let summary = "Profile";
  let description = [{}];
  let arguments = (ins
      I64Attr:$profile_id
  );
}

def ProfileiOp : TensorUnitCommandOp<"profilei"> {
  let summary = "Profile immediate";
  let description = [{}];
  let arguments = (ins
      I64Attr:$profile_id
  );
}

def PrflushOp : TensorUnitCommandOp<"prflush"> {
  let summary = "Profile flush";
  let description = [{}];
  let arguments = (ins);
}

#endif // FURIOSA_TUC_OPS_TD
