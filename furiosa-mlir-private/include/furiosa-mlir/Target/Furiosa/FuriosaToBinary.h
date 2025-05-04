#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::furiosa {

static constexpr auto MIN_BINARY_SIZE = 256;
using binary_t = llvm::SmallString<MIN_BINARY_SIZE>;

FailureOr<binary_t> translateKernelFunctionToBinary(func::FuncOp functionOp);

LogicalResult translateFuriosaToArmC(Operation *op, llvm::raw_ostream &os);

} // namespace mlir::furiosa
