//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "furiosa-mlir/InitAll.h"

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/IR/Dialect.h"

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaDialect.h"
#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.h"

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"
#endif

#ifdef TORCH_MLIR_ENABLE_TOSA
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#endif

void mlir::furiosa::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::furiosa::FuriosaDialect>();
}

void mlir::furiosa::registerAllExtensions(mlir::DialectRegistry &registry) {
  mlir::func::registerInlinerExtension(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
}

void mlir::furiosa::registerAllPasses() {
  mlir::furiosa::registerFuriosaPasses();

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
  mlir::stablehlo::registerStablehloLegalizeToLinalgPass();
  mlir::stablehlo::registerStablehloAggressiveSimplificationPass();
  mlir::stablehlo::registerStablehloRefineShapesPass();
  mlir::stablehlo::registerStablehloConvertToSignlessPass();
  mlir::stablehlo::registerShapeLegalizeToStablehloPass();
  mlir::stablehlo::registerStablehloLegalizeDeprecatedOpsPass();
#endif

#ifdef TORCH_MLIR_ENABLE_REFBACKEND
  mlir::torch::RefBackend::registerRefBackendPasses();
#endif
}
