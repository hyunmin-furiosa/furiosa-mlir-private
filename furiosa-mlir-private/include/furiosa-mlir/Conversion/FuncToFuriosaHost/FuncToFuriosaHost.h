#pragma once

#include <memory>
#include <string>

namespace mlir::furiosa {
class Pass;

#define GEN_PASS_DECL_CONVERTFUNCTOFURIOSAHOSTPASS
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir::furiosa
