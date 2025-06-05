#include "furiosa-mlir-c/RegisterEverything.h"

#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

NB_MODULE(_furiosaMlirRegistration, m) {
  m.doc() = "Furiosa-MLIR All Dialects, Translations and Passes Registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    furiosaMlirRegisterAllDialects(registry);
  });

  // Register all passes on load.
  furiosaMlirRegisterAllPasses();
}
