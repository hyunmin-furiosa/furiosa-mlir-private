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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/Passes.h"

#include "furiosa-mlir/Conversion/Passes.h"
#include "furiosa-mlir/Dialect/Func/Transforms/Passes.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaDialect.h"
#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.h"
#include "furiosa-mlir/Dialect/Host/IR/HostDialect.h"
#include "furiosa-mlir/Dialect/Linalg/Transforms/Passes.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskDialect.h"

void mlir::furiosa::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  mlir::tosa::TosaDialect, mlir::transform::TransformDialect>();
  mlir::furiosa::registerFuriosaDialect(registry);
  mlir::furiosa::host::registerHostDialect(registry);
  mlir::furiosa::task::registerTaskDialect(registry);

  // Register all external models.
  arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  linalg::registerAllDialectInterfaceImplementations(registry);
  scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
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
  mlir::furiosa::registerConvertFuriosaToFuriosaTaskPass();
  mlir::furiosa::registerConvertLinalgToFuriosaPass();

  // Transform passes
  mlir::bufferization::registerBufferizationPasses();
  mlir::registerLinalgPasses();
  mlir::registerSCFPasses();
  mlir::transform::registerTransformPasses();
  mlir::furiosa::registerFuncPasses();
  mlir::furiosa::registerFuriosaPasses();
  mlir::furiosa::registerLinalgPasses();
}
