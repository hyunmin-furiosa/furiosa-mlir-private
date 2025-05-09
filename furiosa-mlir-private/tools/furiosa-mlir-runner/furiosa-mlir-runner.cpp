//===- mlir-runner.cpp - MLIR Translate Driver -------------------------===//
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

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaDialect.h"
#include "furiosa-mlir/Dialect/Host/IR/HostDialect.h"
#include "furiosa-mlir/ExecutionEngine/JitRunner.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();
  mlir::furiosa::registerFuriosaDialect(registry);
  mlir::furiosa::host::registerHostDialect(registry);

  mlir::furiosa::JitRunnerConfig jitRunnerConfig;
  return mlir::furiosa::JitRunnerMain(argc, argv, registry, jitRunnerConfig);
}
