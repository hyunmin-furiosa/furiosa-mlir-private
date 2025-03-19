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

def Furiosa_ExecutionOp : Furiosa_TensorUnitCommand<"exec"> {
  let summary = "Tensor unit execution";
  let description = [{}];
  let arguments = (ins I1Attr
                   : $target_context, I1Attr
                   : $context_id, I32Attr
                   : $subunit_bitmap);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Furiosa_WaitOp : Furiosa_TensorUnitCommand<"wait"> {
  let summary = "Wait";
  let description = [{}];
  let arguments = (ins I1Attr : $target_context);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

#endif // FURIOSA_OPS_TD
