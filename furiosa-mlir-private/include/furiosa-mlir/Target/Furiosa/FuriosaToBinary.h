#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "furiosa-mlir/Target/Furiosa/Binary.h"

namespace mlir::furiosa {

FailureOr<binary_t> translateKernelToBinary(func::FuncOp functionOp);

LogicalResult translateFuriosaToBinary(Operation *op, llvm::raw_ostream &os);

} // namespace mlir::furiosa
