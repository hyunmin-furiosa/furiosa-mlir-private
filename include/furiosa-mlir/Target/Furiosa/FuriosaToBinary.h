#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::furiosa {

LogicalResult translateFuriosaToBinary(Operation *op, llvm::raw_ostream &os);

} // namespace mlir::furiosa
