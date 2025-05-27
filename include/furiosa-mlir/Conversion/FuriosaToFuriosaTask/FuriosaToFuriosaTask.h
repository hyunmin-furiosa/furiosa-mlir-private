#pragma once

#include <memory>
#include <string>

namespace mlir {
class Pass;

namespace furiosa {

#define GEN_PASS_DECL_CONVERTFURIOSATOFURIOSATASKPASS
#include "mlir/Conversion/Passes.h.inc"

} // namespace furiosa
} // namespace mlir
