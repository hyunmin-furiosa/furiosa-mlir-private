//===-- mlir-c/Dialect/GPU.h - C API for GPU dialect -------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef FURIOSA_MLIR_C_DIALECT_FURIOSA_H
#define FURIOSA_MLIR_C_DIALECT_FURIOSA_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Furiosa, furiosa);

//===---------------------------------------------------------------------===//
// BufferType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAFuriosaBufferType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirFuriosaBufferTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirFuriosaBufferTypeGet(MlirContext ctx);

#ifdef __cplusplus
}
#endif

#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.capi.h.inc"

#endif // FURIOSA_MLIR_C_DIALECT_FURIOSA_H
