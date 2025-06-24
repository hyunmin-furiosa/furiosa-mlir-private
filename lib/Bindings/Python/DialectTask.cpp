#include "furiosa-mlir-c/Dialect/Task.h"

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

NB_MODULE(_furiosaMlirDialectsTask, m) {
  m.doc() = "Furiosa-MLIR Task Dialect";

  mlir_type_subclass(m, "SfrType", mlirTypeIsATaskSfrType,
                     mlirTaskSfrTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirTaskSfrTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none());

  mlir_type_subclass(m, "DmaDescriptorType", mlirTypeIsATaskDmaDescriptorType,
                     mlirTaskDmaDescriptorTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirTaskDmaDescriptorTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none());

  mlir_type_subclass(m, "CommandType", mlirTypeIsATaskCommandType,
                     mlirTaskCommandTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirTaskCommandTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none());
}
