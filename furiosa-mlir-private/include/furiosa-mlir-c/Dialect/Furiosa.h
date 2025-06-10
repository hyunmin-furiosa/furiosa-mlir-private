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

#include <cstdint>

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"

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

//===---------------------------------------------------------------------===//
// TargetAttr
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFuriosaTargetAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID mlirFuriosaTargetAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute mlirFuriosaTargetAttrGet(MlirContext ctx,
                                                          std::uint64_t npu,
                                                          std::uint64_t peBegin,
                                                          std::uint64_t peEnd);

//===---------------------------------------------------------------------===//
// MappingAttr
//===---------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED bool mlirAttributeIsAFuriosaMappingAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID mlirFuriosaMappingAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute mlirFuriosaMappingAttrGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// MemoryTypeAttr
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool
mlirAttributeIsAFuriosaMemoryTypeAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID mlirFuriosaMemoryTypeAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute
mlirFuriosaMemoryTypeAttrGet(MlirContext ctx, mlir::furiosa::MemoryType value);

#ifdef __cplusplus
}
#endif

#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.capi.h.inc"

#endif // FURIOSA_MLIR_C_DIALECT_FURIOSA_H
