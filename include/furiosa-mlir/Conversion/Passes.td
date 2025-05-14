//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FURIOSA_MLIR_CONVERSION_PASSES_TD
#define FURIOSA_MLIR_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// FuncToFuriosaHost
//===----------------------------------------------------------------------===//

def ConvertFuncToFuriosaHostPass
    : Pass<"convert-func-to-furiosa-host", "ModuleOp"> {
  let summary = "Convert func dialect to furiosa host dialect";
  let dependentDialects = ["func::FuncDialect", "furiosa::host::HostDialect"];
}

//===----------------------------------------------------------------------===//
// LinalgToFuriosa
//===----------------------------------------------------------------------===//

def ConvertLinalgToFuriosaPass : Pass<"convert-linalg-to-furiosa", "ModuleOp"> {
  let summary = "Convert linalg dialect to furiosa dialect";
  let dependentDialects = ["scf::SCFDialect", "linalg::LinalgDialect",
                           "furiosa::FuriosaDialect"];
}

#endif // FURIOSA_MLIR_CONVERSION_PASSES_TD
