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
      I64Attr:$size,
      I64Attr:$sfr_address
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_RtosfrOp : Furiosa_TensorUnitCommand<"rtosfr"> {
  let summary = "Register to special register file";
  let description = [{}];
  let arguments = (ins
      I64Attr:$value,
      I64Attr:$size,
      I64Attr:$sfr_address
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_RtosfriOp : Furiosa_TensorUnitCommand<"rtosfri"> {
  let summary = "Register to special register file immediate";
  let description = [{}];
  let arguments = (ins
      I64Attr:$value,
      I32Attr:$size,
      I32Attr:$sfr_address
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_MtosfrOp : Furiosa_TensorUnitCommand<"mtosfr"> {
  let summary = "Memory to special register file";
  let description = [{}];
  let arguments = (ins
      I64Attr:$sfr_address,
      I64Attr:$size,
      I64Attr:$spm_address
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_StosfrOp : Furiosa_TensorUnitCommand<"stosfr"> {
  let summary = "SRAM to special register file";
  let description = [{}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_SfrtosOp : Furiosa_TensorUnitCommand<"sfrtos"> {
  let summary = "special register file to SRAM";
  let description = [{}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_StallOp : Furiosa_TensorUnitCommand<"stall"> {
  let summary = "Stall";
  let description = [{}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_ItosOp : Furiosa_TensorUnitCommand<"itos"> {
  let summary = "Immediate to SRAM";
  let description = [{}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_ItosiOp : Furiosa_TensorUnitCommand<"itosi"> {
  let summary = "Immediate to SRAM immediate";
  let description = [{}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_StosOp : Furiosa_TensorUnitCommand<"stos"> {
  let summary = "SRAM to SRAM";
  let description = [{}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_StotabOp : Furiosa_TensorUnitCommand<"stotab"> {
  let summary = "SRAM to table";
  let description = [{}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_StotrfOp : Furiosa_TensorUnitCommand<"stotrf"> {
  let summary = "SRAM to tensor register file";
  let description = [{}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_StovrfOp : Furiosa_TensorUnitCommand<"stovrf"> {
  let summary = "SRAM to vector register file";
  let description = [{}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_ExecutionOp : Furiosa_TensorUnitCommand<"exec"> {
  let summary = "Tensor unit execution";
  let description = [{}];
  let arguments = (ins
      I1Attr:$target_context,
      I1Attr:$context_id,
      I32Attr:$subunit_bitmap
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_WaitOp : Furiosa_TensorUnitCommand<"wait"> {
  let summary = "Wait";
  let description = [{}];
  let arguments = (ins
      I1Attr:$target_context
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_WaitiOp : Furiosa_TensorUnitCommand<"waiti"> {
  let summary = "Wait immediate";
  let description = [{}];
  let arguments = (ins);
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
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_Dma1Op : Furiosa_TensorUnitCommand<"dma1"> {
  let summary = "Direct memory access 1";
  let description = [{}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_DmawOp : Furiosa_TensorUnitCommand<"dmaw"> {
  let summary = "Direct memory access wide";
  let description = [{}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_ProfileOp : Furiosa_TensorUnitCommand<"profile"> {
  let summary = "Profile";
  let description = [{}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_ProfileiOp : Furiosa_TensorUnitCommand<"profilei"> {
  let summary = "Profile immediate";
  let description = [{}];
  let arguments = (ins);
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

#endif // FURIOSA_OPS_TD
