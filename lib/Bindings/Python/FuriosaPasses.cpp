#include "furiosa-mlir-c/Dialect/Furiosa.h"

#include "mlir/Bindings/Python/Nanobind.h"

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

NB_MODULE(_mlirFuriosaPasses, m) {
  m.doc() = "Furiosa-MLIR Furiosa Dialect Passes";

  // Register all GPU passes on load.
  mlirRegisterFuriosaPasses();
}
