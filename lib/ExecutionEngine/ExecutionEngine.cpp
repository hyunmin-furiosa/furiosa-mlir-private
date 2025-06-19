#include "furiosa-mlir/ExecutionEngine/ExecutionEngine.h"

using namespace mlir;
using namespace mlir::furiosa;

llvm::Expected<std::unique_ptr<ExecutionEngine>>
ExecutionEngine::create(Operation *module) {
  return std::make_unique<ExecutionEngine>(module);
}

llvm::Error ExecutionEngine::invokePacked(StringRef func_name,
                                          MutableArrayRef<void *> args) {
  if (failed(executeFunction(*this, func_name))) {
    return llvm::createStringError("Failed to execute function: " + func_name);
  } else {
    return llvm::Error::success();
  }
}
