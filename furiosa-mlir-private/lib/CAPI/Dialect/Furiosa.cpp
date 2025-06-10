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
