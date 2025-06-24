#include "furiosa-mlir-c/ExecutionEngine/ExecutionEngine.h"

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaDialect.h"
#include "furiosa-mlir/Dialect/Host/IR/HostDialect.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskDialect.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

using namespace mlir;
using namespace mlir::furiosa;

extern "C" FuriosaMlirExecutionEngine
furiosaMlirExecutionEngineCreate(MlirModule module, MlirAttribute target) {
  auto engine = ExecutionEngine::create(unwrap(module), unwrap(target));
  if (!engine) {
    consumeError(engine.takeError());
    return FuriosaMlirExecutionEngine{nullptr};
  }
  return wrap(engine->release());
}

extern "C" void
furiosaMlirExecutionEngineDestroy(FuriosaMlirExecutionEngine engine) {
  if (furiosaMlirExecutionEngineIsNull(engine)) {
    return;
  }
  delete unwrap(engine);
  engine.ptr = nullptr;
}

extern "C" MlirLogicalResult furiosaMlirExecutionEngineInvokePacked(
    FuriosaMlirExecutionEngine engine, MlirStringRef name,
    std::int64_t num_args, std::int64_t num_inputs, void **args) {
  llvm::Error error = unwrap(engine)->invokePacked(unwrap(name).str(), num_args,
                                                   num_inputs, args);
  if (error)
    return wrap(failure());
  return wrap(success());
}
