//===- mlir-translate.cpp - MLIR Translate Driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include <furiosa-mlir/InitAll.h>
#include <mlir/IR/DialectRegistry.h>

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
#include "stablehlo/dialect/Register.h"
#endif

using namespace mlir;

namespace mlir::furiosa {
void registerFuriosaToBinary();
} // namespace mlir::furiosa

int main(int argc, char **argv) {
  mlir::furiosa::registerFuriosaToBinary();

  return failed(mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
