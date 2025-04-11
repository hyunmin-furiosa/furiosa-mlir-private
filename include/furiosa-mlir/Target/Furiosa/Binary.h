#pragma once

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

namespace mlir::furiosa {

static constexpr auto MIN_BINARY_SIZE = 256;

using address_size_t = std::pair<std::uint64_t, std::uint64_t>; // address, size

struct FuriosaBinaryMetadata {
  std::uint64_t npu;
  std::uint64_t peBegin;
  std::uint64_t peEnd;
  std::uint64_t argumentSize;
  std::uint64_t resultSize;
  std::uint64_t binaryAddress;
  std::uint64_t binarySize;
};

struct FuriosaBinary {
  FuriosaBinaryMetadata metadata;
  llvm::SmallVector<address_size_t> arguments;
  llvm::SmallVector<address_size_t> results;
  llvm::SmallString<MIN_BINARY_SIZE> binary;
};

LogicalResult writeFuriosaBinary(llvm::Twine filepath,
                                 FuriosaBinary furiosaBinary);
FailureOr<FuriosaBinary> readFuriosaBinary(llvm::Twine filepath);

} // namespace mlir::furiosa
