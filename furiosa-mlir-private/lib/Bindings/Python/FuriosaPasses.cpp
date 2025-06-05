#include "furiosa-mlir-c/Dialect/Furiosa.h"

#include "mlir/Bindings/Python/Nanobind.h"

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

NB_MODULE(_furiosaMlirFuriosaPasses, m) {
  m.doc() = "Furiosa-MLIR Furiosa Dialect Passes";

  // Register all Furiosa passes on load.
  mlirRegisterFuriosaPasses();
}
