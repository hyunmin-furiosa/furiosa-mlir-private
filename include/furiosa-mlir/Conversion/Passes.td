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

def ConvertFuncToFuriosaHostPass : Pass<"convert-func-to-furiosa-host"> {
  let summary = "Convert func dialect to furiosa host dialect";
  let dependentDialects = ["func::FuncDialect", "furiosa::host::HostDialect"];
}

//===----------------------------------------------------------------------===//
// LinalgToFuriosa
//===----------------------------------------------------------------------===//

def ConvertLinalgToFuriosaPass : Pass<"convert-linalg-to-furiosa"> {
  let summary = "Convert linalg dialect to furiosa dialect";
  let dependentDialects = ["bufferization::BufferizationDialect",
                           "scf::SCFDialect", "linalg::LinalgDialect",
                           "furiosa::FuriosaDialect"];
}

//===----------------------------------------------------------------------===//
// FuriosaToFuriosaTask
//===----------------------------------------------------------------------===//

def ConvertFuriosaToFuriosaTaskPass : Pass<"convert-furiosa-to-furiosa-task"> {
  let summary = "Convert linalg dialect to furiosa dialect";
  let dependentDialects = ["furiosa::FuriosaDialect",
                           "furiosa::task::TaskDialect"];
}

#endif // FURIOSA_MLIR_CONVERSION_PASSES_TD
