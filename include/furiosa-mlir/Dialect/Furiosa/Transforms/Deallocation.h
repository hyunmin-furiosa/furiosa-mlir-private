#pragma once

#include <memory>
#include <string>

namespace mlir {
class Pass;

namespace furiosa {

#define GEN_PASS_DECL_FURIOSADEALLOCATIONPASS
#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.h.inc"

} // namespace furiosa
} // namespace mlir
