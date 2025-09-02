#include "furiosa-mlir/Dialect/Furiosa/IR/Utils.h"

using namespace mlir;
using namespace mlir::furiosa;

FailureOr<MemoryType> mlir::furiosa::getMemoryType(Value value) {
  auto tensor_type = llvm::dyn_cast_or_null<RankedTensorType>(value.getType());
  if (!tensor_type) {
    return failure();
  }

  if (auto encoding = tensor_type.getEncoding()) {
    return llvm::dyn_cast_or_null<MemoryTypeAttr>(encoding).getValue();
  } else {
    return MemoryType::dram; // Default to DRAM if no encoding is specified
  }
}
