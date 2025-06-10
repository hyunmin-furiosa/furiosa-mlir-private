#pragma once

#include <cstdint>

#include "furiosa-mlir/Dialect/Host/IR/HostOps.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Host, host);

//===---------------------------------------------------------------------===//
// PeProgramType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAHostPeProgramType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirHostPeProgramTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirHostPeProgramTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// HalProgramType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAHostHalProgramType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirHostHalProgramTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirHostHalProgramTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// DeviceType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAHostDeviceType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirHostDeviceTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirHostDeviceTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// ExecutionType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAHostExecutionType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirHostExecutionTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirHostExecutionTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// BufferType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAHostBufferType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirHostBufferTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirHostBufferTypeGet(MlirContext ctx);

#ifdef __cplusplus
}
#endif
