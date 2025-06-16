#pragma once

#include <random>

#include "device_runtime.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/Any.h"
#include "llvm/ADT/ArrayRef.h"

#include "furiosa-mlir/Target/C/FuriosaTaskToC.h"

namespace mlir::furiosa {

#define CEIL(a, b) (((a + b - 1) / b) * b)

static constexpr auto MIN_BYTE_ARRAY_SIZE = 256;
static constexpr auto DRAM_ACCESS_WIDTH = 256;
using byte_array_t = SmallVector<std::uint8_t, MIN_BYTE_ARRAY_SIZE>;
using pe_program_t = SmallVector<device_runtime::Stmt *>;
using hal_program_t = SmallVector<device_runtime::Program *>;
using device_t = device_runtime::Device *;
using execution_t = device_runtime::Execution *;

class ExecutionContext {
public:
  ExecutionContext(Operation *module)
      : module(module), randomNumberGenerator(), distribution() {}

  Operation *getModule() const { return module; }

  void createValue(Value val, llvm::Any data) {
    if (!valueMapper.count(val)) {
      valueMapper.insert_or_assign(val, data);
    }
  }

  /// get current data of value
  llvm::Any &getValue(Value val) {
    if (!valueMapper.count(val)) {
      llvm::report_fatal_error(llvm::Twine("value does not exist"));
    }
    return valueMapper[val];
  }

  std::uint64_t getRandomNumber() {
    return distribution(randomNumberGenerator);
  }

private:
  using ValueMapper = llvm::DenseMap<Value, llvm::Any>;

  Operation *module;

  /// Map from value to its data
  ValueMapper valueMapper;

  /// random number generator
  std::mt19937 randomNumberGenerator;
  std::uniform_int_distribution<std::uint64_t> distribution;
};

LogicalResult executeFunction(Operation *module, StringRef entry_point,
                              StringRef entry_point_type);

} // namespace mlir::furiosa
