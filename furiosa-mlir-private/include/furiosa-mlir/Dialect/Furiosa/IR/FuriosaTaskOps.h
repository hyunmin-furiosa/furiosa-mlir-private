//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaTypes.h"

#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaTaskOps.h.inc"
