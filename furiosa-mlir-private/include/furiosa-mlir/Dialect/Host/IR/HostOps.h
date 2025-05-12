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
#include "furiosa-mlir/Dialect/Host/IR/HostDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_ATTRDEF_CLASSES
#include "furiosa-mlir/Dialect/Host/IR/HostAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "furiosa-mlir/Dialect/Host/IR/HostTypes.h.inc"

#define GET_OP_CLASSES
#include "furiosa-mlir/Dialect/Host/IR/HostOps.h.inc"
