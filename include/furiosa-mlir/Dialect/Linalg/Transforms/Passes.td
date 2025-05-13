//===-- Passes.td - Linalg pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FURIOSA_MLIR_LINALG_PASSES_TD
#define FURIOSA_MLIR_LINALG_PASSES_TD

include "mlir/Pass/PassBase.td"

def LinalgGeneralizeToContractOpsPass
    : Pass<"linalg-generalize-to-contract-ops"> {
  let summary = "Convert named ops into contract ops";
  let dependentDialects = ["linalg::LinalgDialect"];
}

#endif // FURIOSA_MLIR_LINALG_PASSES_TD
