#pragma once

#include <cstdint>

#include "furiosa-mlir/Dialect/Task/IR/TaskOps.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Task, task);

//===---------------------------------------------------------------------===//
// SfrType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsATaskSfrType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirTaskSfrTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirTaskSfrTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// DmaDescriptorType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsATaskDmaDescriptorType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirTaskDmaDescriptorTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirTaskDmaDescriptorTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// CommandType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsATaskCommandType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirTaskCommandTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirTaskCommandTypeGet(MlirContext ctx);

#ifdef __cplusplus
}
#endif
