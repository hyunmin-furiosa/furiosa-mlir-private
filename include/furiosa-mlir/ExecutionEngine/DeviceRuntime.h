#pragma once

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"

#include "furiosa-mlir/Target/Furiosa/Binary.h"

void launchKernel(mlir::furiosa::FuriosaBinary furiosaBinary);

namespace mlir::furiosa {

LogicalResult executeFunction(Operation *module, StringRef entryPoint,
                              StringRef entryPointType);

} // namespace mlir::furiosa
