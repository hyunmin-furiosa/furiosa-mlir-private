//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FURIOSA_MLIR_CONVERSION_PASSES
#define FURIOSA_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// FuncToFuriosaHost
//===----------------------------------------------------------------------===//

def ConvertFuncToFuriosaHostPass
    : Pass<"convert-func-to-furiosa-host", "ModuleOp"> {
  let summary = "Convert func dialect to furiosa host dialect";
  let dependentDialects = ["func::FuncDialect", "furiosa::host::HostDialect"];
}

#endif // FURIOSA_MLIR_CONVERSION_PASSES
