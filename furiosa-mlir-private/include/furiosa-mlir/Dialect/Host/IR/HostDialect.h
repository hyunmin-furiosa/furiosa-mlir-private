//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Dialect.h"

#include "furiosa-mlir/Dialect/Host/IR/HostDialect.h.inc"

namespace mlir::furiosa::host {

void registerHostDialect(DialectRegistry &registry);

} // namespace mlir::furiosa::host
