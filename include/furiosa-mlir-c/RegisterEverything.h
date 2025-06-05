#ifndef FURIOSA_MLIR_C_REGISTER_EVERYTHING_H
#define FURIOSA_MLIR_C_REGISTER_EVERYTHING_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Registers all dialects with a context.
 * This is needed before creating IR for these Dialects.
 */
MLIR_CAPI_EXPORTED void
furiosaMlirRegisterAllDialects(MlirDialectRegistry context);

/** Registers all passes for symbolic access with the global registry. */
MLIR_CAPI_EXPORTED void furiosaMlirRegisterAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif // FURIOSA_MLIR_C_REGISTER_EVERYTHING_H