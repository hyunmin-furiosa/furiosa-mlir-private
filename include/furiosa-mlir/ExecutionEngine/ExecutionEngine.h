#pragma once

#include <random>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/Any.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include "furiosa-mlir/ExecutionEngine/RenegadeRuntime.h"

namespace mlir::furiosa {

class ExecutionEngine {
public:
  ExecutionEngine(ModuleOp module, Attribute target)
      : module(module), target(target), randomNumberGenerator(),
        distribution() {}

  // execution engine functions for python binding
  static llvm::Expected<std::unique_ptr<ExecutionEngine>>
  create(ModuleOp module, Attribute target = Attribute());

  llvm::Error invokePacked(StringRef func_name, std::int64_t num_args,
                           std::int64_t num_inputs, void **args);

  ModuleOp getModule() const { return module; }

  Attribute getTarget() const { return target; }

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

  ModuleOp module;
  Attribute target;

  /// Map from value to its data
  ValueMapper valueMapper;

  /// random number generator
  std::mt19937 randomNumberGenerator;
  std::uniform_int_distribution<std::uint64_t> distribution;
};

} // namespace mlir::furiosa
