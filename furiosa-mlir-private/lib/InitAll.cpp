//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "furiosa-mlir/InitAll.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/Passes.h"

#include "furiosa-mlir/Conversion/Passes.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaDialect.h"
#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.h"
#include "furiosa-mlir/Dialect/Host/IR/HostDialect.h"
#include "furiosa-mlir/Dialect/Linalg/Transforms/Passes.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskDialect.h"

void mlir::furiosa::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                  mlir::tosa::TosaDialect, mlir::transform::TransformDialect>();
  mlir::furiosa::registerFuriosaDialect(registry);
  mlir::furiosa::host::registerHostDialect(registry);
  mlir::furiosa::task::registerTaskDialect(registry);
}

void mlir::furiosa::registerAllExtensions(mlir::DialectRegistry &registry) {
  mlir::linalg::registerTransformDialectExtension(registry);
}

void mlir::furiosa::registerAllPasses() {
  // General passes
  mlir::registerTransformsPasses();

  // Conversion passes
  mlir::registerTosaToLinalg();
  mlir::registerTosaToLinalgNamed();
  mlir::furiosa::registerConvertFuncToFuriosaHostPass();

  // Transform passes
  mlir::registerLinalgPasses();
  mlir::furiosa::registerLinalgPasses();
}
