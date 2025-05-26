#pragma once

#include <memory>
#include <string>

namespace mlir {
class Pass;

namespace furiosa {

#define GEN_PASS_DECL_FUNCRESULTSTOPARAMSPASS
#include "furiosa-mlir/Dialect/Func/Transforms/Passes.h.inc"

} // namespace furiosa
} // namespace mlir
