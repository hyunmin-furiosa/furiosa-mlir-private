#include "furiosa-mlir-c/Dialect/Furiosa.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaDialect.h"

#include "mlir/CAPI/Registration.h"

#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::furiosa;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Furiosa, furiosa, furiosa::FuriosaDialect)
