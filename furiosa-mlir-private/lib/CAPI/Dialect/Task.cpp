#include "furiosa-mlir-c/Dialect/Task.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskDialect.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskOps.h"

#include "mlir/CAPI/Registration.h"

#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::furiosa;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Task, task, task::TaskDialect)

//===----------------------------------------------------------------------===//
// SfrType
//===----------------------------------------------------------------------===//

bool mlirTypeIsATaskSfrType(MlirType type) {
  return llvm::isa<task::SfrType>(unwrap(type));
}

MlirTypeID mlirTaskSfrTypeGetTypeID(void) {
  return wrap(task::SfrType::getTypeID());
}

MlirType mlirTaskSfrTypeGet(MlirContext ctx) {
  return wrap(task::SfrType::get(unwrap(ctx)));
}

//===----------------------------------------------------------------------===//
// DmaDescriptorType
//===----------------------------------------------------------------------===//

bool mlirTypeIsATaskDmaDescriptorType(MlirType type) {
  return llvm::isa<task::DmaDescriptorType>(unwrap(type));
}

MlirTypeID mlirTaskDmaDescriptorTypeGetTypeID(void) {
  return wrap(task::DmaDescriptorType::getTypeID());
}

MlirType mlirTaskDmaDescriptorTypeGet(MlirContext ctx) {
  return wrap(task::DmaDescriptorType::get(unwrap(ctx)));
}

//===----------------------------------------------------------------------===//
// CommandType
//===----------------------------------------------------------------------===//

bool mlirTypeIsATaskCommandType(MlirType type) {
  return llvm::isa<task::CommandType>(unwrap(type));
}

MlirTypeID mlirTaskCommandTypeGetTypeID(void) {
  return wrap(task::CommandType::getTypeID());
}

MlirType mlirTaskCommandTypeGet(MlirContext ctx) {
  return wrap(task::CommandType::get(unwrap(ctx)));
}
