#pragma once

#include <any>

#include "furiosa_torch.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"

#include "furiosa-mlir/Target/Furiosa/FuriosaToBinary.h"

namespace mlir::furiosa {

#define CEIL(a, b) (((a + b - 1) / b) * b)

static constexpr auto MIN_BYTE_ARRAY_SIZE = 256;
static constexpr auto DRAM_ACCESS_WIDTH = 256;
using byte_array_t = SmallVector<std::uint8_t, MIN_BYTE_ARRAY_SIZE>;
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

LogicalResult executeFunction(Operation *module, StringRef entry_point,
                              StringRef entry_point_type);

} // namespace mlir::furiosa
