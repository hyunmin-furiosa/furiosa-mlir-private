#pragma once

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

namespace mlir::furiosa {

static constexpr auto MIN_BINARY_SIZE = 256;

struct FuriosaBinary {
  llvm::SmallVector<std::uint32_t> argumentSizes;
  llvm::SmallVector<std::uint32_t> resultSizes;
  llvm::SmallString<MIN_BINARY_SIZE> binBuffer;
};

LogicalResult writeFuriosaBinary(llvm::Twine filepath,
                                 FuriosaBinary furiosaBinary);
FailureOr<FuriosaBinary> readFuriosaBinary(llvm::Twine filepath);

} // namespace mlir::furiosa
