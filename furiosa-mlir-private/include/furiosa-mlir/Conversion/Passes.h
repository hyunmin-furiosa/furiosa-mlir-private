//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "furiosa-mlir/Conversion/FuncToFuriosaHost/FuncToFuriosaHost.h"
#include "furiosa-mlir/Conversion/FuriosaToFuriosaTask/FuriosaToFuriosaTask.h"
#include "furiosa-mlir/Conversion/LinalgToFuriosa/LinalgToFuriosa.h"

namespace mlir::furiosa {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "furiosa-mlir/Conversion/Passes.h.inc"

} // namespace mlir::furiosa
