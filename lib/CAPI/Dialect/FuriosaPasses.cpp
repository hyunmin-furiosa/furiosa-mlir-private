#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"

// Must include the declarations as they carry important visibility attributes.
#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.capi.h.inc"

using namespace mlir;
using namespace mlir::furiosa;

#ifdef __cplusplus
extern "C" {
#endif

#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
