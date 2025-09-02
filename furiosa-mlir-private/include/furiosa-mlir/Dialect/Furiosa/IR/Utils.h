#pragma once

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"

namespace mlir::furiosa {

FailureOr<MemoryType> getMemoryType(Value value);

} // namespace mlir::furiosa
