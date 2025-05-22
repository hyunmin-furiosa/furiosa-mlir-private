//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TASK_OPS_TD
#define TASK_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "furiosa-mlir/Dialect/Task/IR/RenegadeSfr.td"
include "furiosa-mlir/Dialect/Task/IR/TaskBase.td"
include "furiosa-mlir/Dialect/Task/IR/TaskTypes.td"

//===----------------------------------------------------------------------===//
// Task operation definitions
//===----------------------------------------------------------------------===//

class Task_Op<string mnemonic, list<Trait> traits = []>
    : Op<Task_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Tensor unit commands.
//===----------------------------------------------------------------------===//

class TensorUnitCommandOp<string mnemonic, list<Trait> traits = []>
    : Task_Op<"tuc."#mnemonic, traits> {
  let cppNamespace = Task_Dialect.cppNamespace#"::tuc";
  dag dependency = (ins
      Variadic<CommandType>:$dependencies
  );
  let results = (outs
      CommandType:$command
  );
  let assemblyFormat = "$dependencies attr-dict";
}

def ItosfrOp : TensorUnitCommandOp<"itosfr"> {
  let summary = "Immediate to special register file";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I32Attr:$value,
                                       I64Attr:$sfr_address,
                                       I64Attr:$size
                                   ));
}

def RtosfrOp : TensorUnitCommandOp<"rtosfr"> {
  let summary = "Register to special register file";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I64Attr:$value,
                                       I64Attr:$sfr_address,
                                       I64Attr:$size
                                   ));
}

def RtosfriOp : TensorUnitCommandOp<"rtosfri"> {
  let summary = "Register to special register file immediate";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I32Attr:$sfr_address,
                                       I32Attr:$log_size,
                                       I64Attr:$value
                                   ));
}

def MtosfrOp : TensorUnitCommandOp<"mtosfr"> {
  let summary = "Memory to special register file";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I64Attr:$spm_address,
                                       I64Attr:$size,
                                       I64Attr:$sfr_address
                                   ));
}

def StosfrOp : TensorUnitCommandOp<"stosfr"> {
  let summary = "SRAM to special register file";
  let description = [{}];
  let arguments = !con(dependency, (ins
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
                                   ));
}

def SfrtosOp : TensorUnitCommandOp<"sfrtos"> {
  let summary = "special register file to SRAM";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I64Attr:$commit_base,
                                       I64Attr:$commit_limit,
                                       I64Attr:$sfr_address
                                   ));
}

def StallOp : TensorUnitCommandOp<"stall"> {
  let summary = "Stall";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I64Attr:$cycle
                                   ));
}

def ItosOp : TensorUnitCommandOp<"itos"> {
  let summary = "Immediate to SRAM";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I64Attr:$address_begin,
                                       I64Attr:$address_end,
                                       I64Attr:$value,
                                       I64Attr:$dim1_log_size,
                                       I64Attr:$limit1,
                                       I64Attr:$stride1
                                   ));
}

def ItosiOp : TensorUnitCommandOp<"itosi"> {
  let summary = "Immediate to SRAM immediate";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I64Attr:$address_begin,
                                       I64Attr:$address_end,
                                       I64Attr:$value
                                   ));
}

def StosOp : TensorUnitCommandOp<"stos"> {
  let summary = "SRAM to SRAM";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I64Attr:$address_begin,
                                       I64Attr:$address_end,
                                       I64Attr:$destination_begin,
                                       I64Attr:$slice_log_size,
                                       I64Attr:$dim1_log_size,
                                       I64Attr:$words_per_packet
                                   ));
}

def StotabOp : TensorUnitCommandOp<"stotab"> {
  let summary = "SRAM to table";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I64Attr:$fetch_base,
                                       I64Attr:$fetch_limit,
                                       I64Attr:$tables,
                                       I64Attr:$commit_base,
                                       I64Attr:$topology,
                                       I64Attr:$slice_log_size,
                                       I64Attr:$dim0_log_size,
                                       I64Attr:$dim1_log_size,
                                       I64Attr:$words_per_packet
                                   ));
}

def StotrfOp : TensorUnitCommandOp<"stotrf"> {
  let summary = "SRAM to tensor register file";
  let description = [{}];
  let arguments = !con(dependency, (ins
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
                                   ));
}

def StovrfOp : TensorUnitCommandOp<"stovrf"> {
  let summary = "SRAM to vector register file";
  let description = [{}];
  let arguments = !con(dependency, (ins
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
                                   ));
}

def ExecutionOp : TensorUnitCommandOp<"exec"> {
  let summary = "Tensor unit execution";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I32Attr:$subunit_bitmap,
                                       I1Attr:$context_id,
                                       I1Attr:$target_context
                                   ));
}

def WaitOp : TensorUnitCommandOp<"wait"> {
  let summary = "Wait";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I32Attr:$dma_tag_id,
                                       I1Attr:$type,
                                       I1Attr:$target_context
                                   ));
}

def WaitiOp : TensorUnitCommandOp<"waiti"> {
  let summary = "Wait interrupt";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I32Attr:$dma_tag_id,
                                       I1Attr:$type,
                                       I1Attr:$target_context
                                   ));
}

def InterruptOp : TensorUnitCommandOp<"interrupt"> {
  let summary = "SRAM to tensor register file";
  let description = [{}];
  let arguments = !con(dependency, (ins));
}

def DmaOp : TensorUnitCommandOp<"dma"> {
  let summary = "Direct memory access";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I64Attr:$pe0_desc_addr,
                                       I64Attr:$pe1_desc_addr,
                                       I64Attr:$pe2_desc_addr,
                                       I64Attr:$pe3_desc_addr,
                                       I64Attr:$dma_tag_id,
                                       I1Attr:$profile,
                                       I64Attr:$profile_id
                                   ));
}

def Dma1Op : TensorUnitCommandOp<"dma1"> {
  let summary = "Direct memory access 1";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I64Attr:$desc_addr,
                                       I64Attr:$pe_valid_bitmap,
                                       I64Attr:$dma_tag_id,
                                       I1Attr:$profile,
                                       I64Attr:$profile_id
                                   ));
}

def DmawOp : TensorUnitCommandOp<"dmaw"> {
  let summary = "Direct memory access wide";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I64Attr:$pe0_desc_addr,
                                       I64Attr:$pe1_desc_addr,
                                       I64Attr:$pe2_desc_addr,
                                       I64Attr:$pe3_desc_addr,
                                       I64Attr:$dma_tag_id,
                                       I1Attr:$profile,
                                       I64Attr:$profile_id
                                   ));
}

def ProfileOp : TensorUnitCommandOp<"profile"> {
  let summary = "Profile";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I64Attr:$profile_id
                                   ));
}

def ProfileiOp : TensorUnitCommandOp<"profilei"> {
  let summary = "Profile immediate";
  let description = [{}];
  let arguments = !con(dependency, (ins
                                       I64Attr:$profile_id
                                   ));
}

def PrflushOp : TensorUnitCommandOp<"prflush"> {
  let summary = "Profile flush";
  let description = [{}];
  let arguments = !con(dependency, (ins));
}

//===----------------------------------------------------------------------===//
// Task SFR operations.
//===----------------------------------------------------------------------===//

class StaticSfrOp<string mnemonic, list<Trait> traits = []>
    : Task_Op<"static.sfr."#mnemonic, traits> {
  let cppNamespace = Task_Dialect.cppNamespace#"::sfr";
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

class SfrOp<string mnemonic, list<Trait> traits = []>
    : Task_Op<"sfr."#mnemonic, traits> {
  let cppNamespace = Task_Dialect.cppNamespace#"::sfr";
  let results = (outs
      SfrType:$sfr
  );
  let assemblyFormat = "attr-dict";
}

foreach nd = Sfr in {
  def StaticSfr#!cast<string>(nd)#Op : StaticSfrOp<nd.mnemonic> {
    let summary = "Static special function register for "#!cast<string>(nd);
    let description = [{}];
    let arguments = !con((ins
                             I64Attr:$sfr_addr
                         ),
                         nd.data);
  }

  def Sfr#!cast<string>(nd)#Op : SfrOp<nd.mnemonic> {
    let summary = "Special function register for "#!cast<string>(nd);
    let description = [{}];
    let arguments = nd.data;
  }
}

//===----------------------------------------------------------------------===//
// Task operations.
//===----------------------------------------------------------------------===//

def StaticDmaDescriptorOp : Task_Op<"static.dma_descriptor"> {
  let summary = "Static direct memory access descriptor";
  let description = [{}];
  let arguments = (ins
      I64Attr:$desc_addr,
      I64Attr:$opcode,
      I64Attr:$indirect,
      I64Attr:$source_base,
      I64Attr:$destination_base,
      I64ArrayAttr:$source_limits,
      I64ArrayAttr:$source_strides,
      I64ArrayAttr:$destination_limits,
      I64ArrayAttr:$destination_strides
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def DmaDescriptorOp : Task_Op<"dma_descriptor", [AttrSizedOperandSegments]> {
  let summary = "Direct memory access descriptor";
  let description = [{}];
  let arguments = (ins
      Optional<AnyType>:$source,
      Optional<AnyType>:$destination,
      I64Attr:$opcode,
      I64Attr:$indirect,
      OptionalAttr<I64Attr>:$source_base,
      OptionalAttr<I64Attr>:$destination_base,
      I64ArrayAttr:$source_limits,
      I64ArrayAttr:$source_strides,
      I64ArrayAttr:$destination_limits,
      I64ArrayAttr:$destination_strides
  );
  let results = (outs
      DmaDescriptorType:$desc
  );
  let assemblyFormat = "(`source` $source^ `:` type($source))? (`destination` "
                       "$destination^ `:` type($destination))? attr-dict";
}

def DynamicMtosfrOp : Task_Op<"mtosfr"> {
  let summary = "Dynamic memory to special register file";
  let description = [{}];
  let arguments = (ins
      SfrType:$sfr,
      Variadic<CommandType>:$dependencies,
      I64Attr:$sfr_address
  );
  let results = (outs
      CommandType:$command
  );
  let assemblyFormat = "$sfr $dependencies attr-dict";
}

def DynamicDmawOp : Task_Op<"dmaw"> {
  let summary = "Dynamic direct memory access wide";
  let description = [{}];
  let arguments = (ins
      DmaDescriptorType:$desc,
      Variadic<CommandType>:$dependencies,
      I64Attr:$dma_tag_id,
      I1Attr:$profile,
      I64Attr:$profile_id
  );
  let results = (outs
      CommandType:$command
  );
  let assemblyFormat = "$desc $dependencies attr-dict";
}

#endif // TASK_OPS_TD
