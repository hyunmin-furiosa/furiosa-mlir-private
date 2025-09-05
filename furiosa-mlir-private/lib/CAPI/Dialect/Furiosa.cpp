#include "furiosa-mlir-c/Dialect/Furiosa.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaDialect.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"

#include "mlir/CAPI/Registration.h"

#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::furiosa;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Furiosa, furiosa, furiosa::FuriosaDialect)

//===---------------------------------------------------------------------===//
// BufferType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAFuriosaBufferType(MlirType type) {
  return isa<furiosa::BufferType>(unwrap(type));
}

MlirTypeID mlirFuriosaBufferTypeGetTypeID(void) {
  return wrap(furiosa::BufferType::getTypeID());
}

MlirType mlirFuriosaBufferTypeGet(MlirContext ctx) {
  return wrap(furiosa::BufferType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// TargetAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAFuriosaTargetAttr(MlirAttribute attr) {
  return isa<furiosa::TargetAttr>(unwrap(attr));
}

MlirTypeID mlirFuriosaTargetAttrGetTypeID(void) {
  return wrap(furiosa::TargetAttr::getTypeID());
}

MlirAttribute mlirFuriosaTargetAttrGet(MlirContext ctx, std::uint64_t npu,
                                       std::uint64_t peBegin,
                                       std::uint64_t peEnd) {
  return wrap(furiosa::TargetAttr::get(unwrap(ctx), npu, peBegin, peEnd));
}

//===---------------------------------------------------------------------===//
// MappingAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAFuriosaMappingAttr(MlirAttribute attr) {
  return isa<furiosa::MappingAttr>(unwrap(attr));
}

MlirTypeID mlirFuriosaMappingAttrGetTypeID(void) {
  return wrap(furiosa::MappingAttr::getTypeID());
}

MlirAttribute mlirFuriosaMappingAttrGet(MlirContext ctx) {
  return wrap(furiosa::MappingAttr::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// TensorAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAFuriosaTensorAttr(MlirAttribute attr) {
  return isa<furiosa::TensorAttr>(unwrap(attr));
}

MlirTypeID mlirFuriosaTensorAttrGetTypeID(void) {
  return wrap(furiosa::TensorAttr::getTypeID());
}

MlirAttribute mlirFuriosaTensorAttrGet(MlirContext ctx,
                                       mlir::furiosa::MemoryType memory_type,
                                       MlirAttribute memory_map) {
  return wrap(
      furiosa::TensorAttr::get(unwrap(ctx), memory_type, unwrap(memory_map)));
}
