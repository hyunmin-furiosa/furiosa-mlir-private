//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class ModuleOp;

namespace furiosa {

#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.h.inc"

/// Registers all Furiosa transformation passes.
void registerFuriosaPasses();

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.h.inc"

} // namespace furiosa
} // namespace mlir
