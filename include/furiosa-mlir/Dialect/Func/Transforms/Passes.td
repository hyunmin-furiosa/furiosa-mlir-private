//===-- Passes.td - Func pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FURIOSA_MLIR_FUNC_PASSES_TD
#define FURIOSA_MLIR_FUNC_PASSES_TD

include "mlir/Pass/PassBase.td"

def FuncResultsToParamsPass : Pass<"func-results-to-params"> {
  let summary = "Convert function results to function parameters";
  let dependentDialects = ["func::FuncDialect"];
}

#endif // FURIOSA_MLIR_FUNC_PASSES_TD
