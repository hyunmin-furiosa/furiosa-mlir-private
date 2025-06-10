#include "furiosa-mlir-c/Dialect/Furiosa.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace nanobind::literals;

using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

NB_MODULE(_furiosaMlirDialectsFuriosa, m) {
  m.doc() = "MLIR Furiosa Dialect";

  mlir_type_subclass(m, "BufferType", mlirTypeIsAFuriosaBufferType,
                     mlirFuriosaBufferTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirFuriosaBufferTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none());
}
