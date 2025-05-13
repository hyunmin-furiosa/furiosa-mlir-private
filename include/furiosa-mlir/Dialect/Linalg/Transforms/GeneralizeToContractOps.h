#pragma once

#include <memory>
#include <string>

namespace mlir {
class Pass;

namespace furiosa {

#define GEN_PASS_DECL_LINALGGENERALIZETOCONTRACTOPSPASS
#include "furiosa-mlir/Dialect/Linalg/Transforms/Passes.h.inc"

} // namespace furiosa
} // namespace mlir
