#include "furiosa-mlir-c/RegisterEverything.h"

#include "furiosa-mlir/InitAll.h"

#include "mlir/CAPI/IR.h"

void furiosaMlirRegisterAllDialects(MlirDialectRegistry registry) {
  mlir::furiosa::registerAllDialects(*unwrap(registry));
  mlir::furiosa::registerAllExtensions(*unwrap(registry));
}

void furiosaMlirRegisterAllPasses() { mlir::furiosa::registerAllPasses(); }
