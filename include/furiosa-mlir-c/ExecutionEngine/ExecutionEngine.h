#pragma once

#include "mlir/CAPI/Wrap.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "furiosa-mlir/ExecutionEngine/ExecutionEngine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(FuriosaMlirExecutionEngine, void);

#undef DEFINE_C_API_STRUCT

MLIR_CAPI_EXPORTED FuriosaMlirExecutionEngine
furiosaMlirExecutionEngineCreate(MlirModule op);

/// Destroy an ExecutionEngine instance.
MLIR_CAPI_EXPORTED void
furiosaMlirExecutionEngineDestroy(FuriosaMlirExecutionEngine engine);

/// Checks whether an execution engine is null.
static inline bool
furiosaMlirExecutionEngineIsNull(FuriosaMlirExecutionEngine engine) {
  return !engine.ptr;
}

MLIR_CAPI_EXPORTED MlirLogicalResult furiosaMlirExecutionEngineInvokePacked(
    FuriosaMlirExecutionEngine engine, MlirStringRef name,
    std::int64_t num_args, std::int64_t num_inputs, void **args);

DEFINE_C_API_PTR_METHODS(FuriosaMlirExecutionEngine,
                         mlir::furiosa::ExecutionEngine)

#ifdef __cplusplus
}
#endif
