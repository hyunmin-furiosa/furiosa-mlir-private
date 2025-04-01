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

#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::JitRunnerConfig jitRunnerConfig;
  return mlir::JitRunnerMain(argc, argv, registry, jitRunnerConfig);
}
