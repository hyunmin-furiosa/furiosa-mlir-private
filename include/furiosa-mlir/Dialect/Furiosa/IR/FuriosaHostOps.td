//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef FURIOSA_HOST_OPS_TD
#define FURIOSA_HOST_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaBase.td"
include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaTypes.td"

//===----------------------------------------------------------------------===//
// Host operations.
//===----------------------------------------------------------------------===//

def DeviceExecuteOp : HostOp<"device_execute"> {
  let summary = "Host device execute";
  let description = [{}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

#endif // FURIOSA_HOST_OPS_TD
