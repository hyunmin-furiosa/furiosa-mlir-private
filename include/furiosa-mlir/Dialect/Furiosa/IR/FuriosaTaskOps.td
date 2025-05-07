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
// Task operations.
//===----------------------------------------------------------------------===//

def StaticDmaDescriptorOp : TaskOp<"static.dma_descriptor"> {
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

def DmaDescriptorOp : TaskOp<"dma_descriptor", [AttrSizedOperandSegments]> {
  let summary = "Direct memory access descriptor";
  let description = [{}];
  let arguments = (ins
      Optional<I64>:$source,
      Optional<I64>:$destination,
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
  let assemblyFormat = "(`source` $source^)? (`destination` "
                       "$destination^)? attr-dict";
}

def MtosfrOp : TaskOp<"mtosfr"> {
  let summary = "Dynamic memory to special register file";
  let description = [{}];
  let arguments = (ins
      SfrType:$sfr,
      I64Attr:$sfr_address
  );
  let results = (outs);
  let assemblyFormat = "$sfr attr-dict";
}

def DmawOp : TaskOp<"dmaw"> {
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
