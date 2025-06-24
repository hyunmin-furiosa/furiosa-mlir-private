#include "furiosa-mlir-c/Dialect/Host.h"
#include "furiosa-mlir/Dialect/Host/IR/HostDialect.h"
#include "furiosa-mlir/Dialect/Host/IR/HostOps.h"

#include "mlir/CAPI/Registration.h"

#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::furiosa;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Host, host, host::HostDialect)

//===---------------------------------------------------------------------===//
// PeProgramType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAHostPeProgramType(MlirType type) {
  return isa<host::PeProgramType>(unwrap(type));
}

MlirTypeID mlirHostPeProgramTypeGetTypeID(void) {
  return wrap(host::PeProgramType::getTypeID());
}

MlirType mlirHostPeProgramTypeGet(MlirContext ctx) {
  return wrap(host::PeProgramType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// HalProgramType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAHostHalProgramType(MlirType type) {
  return isa<host::HalProgramType>(unwrap(type));
}

MlirTypeID mlirHostHalProgramTypeGetTypeID(void) {
  return wrap(host::HalProgramType::getTypeID());
}

MlirType mlirHostHalProgramTypeGet(MlirContext ctx) {
  return wrap(host::HalProgramType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// DeviceType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAHostDeviceType(MlirType type) {
  return isa<host::DeviceType>(unwrap(type));
}

MlirTypeID mlirHostDeviceTypeGetTypeID(void) {
  return wrap(host::DeviceType::getTypeID());
}

MlirType mlirHostDeviceTypeGet(MlirContext ctx) {
  return wrap(host::DeviceType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// ExecutionType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAHostExecutionType(MlirType type) {
  return isa<host::ExecutionType>(unwrap(type));
}

MlirTypeID mlirHostExecutionTypeGetTypeID(void) {
  return wrap(host::ExecutionType::getTypeID());
}

MlirType mlirHostExecutionTypeGet(MlirContext ctx) {
  return wrap(host::ExecutionType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// BufferType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAHostBufferType(MlirType type) {
  return isa<host::BufferType>(unwrap(type));
}

MlirTypeID mlirHostBufferTypeGetTypeID(void) {
  return wrap(host::BufferType::getTypeID());
}

MlirType mlirHostBufferTypeGet(MlirContext ctx) {
  return wrap(host::BufferType::get(unwrap(ctx)));
}
