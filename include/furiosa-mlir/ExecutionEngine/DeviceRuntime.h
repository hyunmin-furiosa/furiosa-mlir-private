#pragma once

#include <any>

#include "furiosa_torch.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"

#include "furiosa-mlir/Target/Furiosa/Binary.h"

#define CEIL(a, b) (((a + b - 1) / b) * b)

namespace mlir::furiosa {

using byte_array_t = SmallVector<std::uint8_t, 256>;
using pe_program_t = SmallVector<furiosa_torch::PeProgram *>;
using hal_program_t = SmallVector<furiosa_torch::HalProgram *>;
using device_t = furiosa_torch::Device *;

struct ExecutionContext {
  Operation *module;

  void createValue(Value val, std::any data) {
    if (!valueMapper.count(val)) {
      valueMapper.insert_or_assign(val, data);
    }
  }

  /// get current data of value
  std::any &getValue(Value val) {
    if (!valueMapper.count(val)) {
      llvm::report_fatal_error(llvm::Twine("value does not exist"));
    }
    return valueMapper[val];
  }

private:
  using ValueMapper = llvm::DenseMap<Value, std::any>;

  /// Map from value to its data
  ValueMapper valueMapper;
};

LogicalResult executeFunction(Operation *module, StringRef entryPoint,
                              StringRef entryPointType);

} // namespace mlir::furiosa
