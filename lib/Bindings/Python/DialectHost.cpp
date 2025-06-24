#include "furiosa-mlir-c/Dialect/Host.h"

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

NB_MODULE(_furiosaMlirDialectsHost, m) {
  m.doc() = "Furiosa-MLIR Host Dialect";

  mlir_type_subclass(m, "PeProgramType", mlirTypeIsAHostPeProgramType,
                     mlirHostPeProgramTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirHostPeProgramTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none());

  mlir_type_subclass(m, "HalProgramType", mlirTypeIsAHostHalProgramType,
                     mlirHostHalProgramTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirHostHalProgramTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none());

  mlir_type_subclass(m, "DeviceType", mlirTypeIsAHostDeviceType,
                     mlirHostDeviceTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirHostDeviceTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none());

  mlir_type_subclass(m, "ExecutionType", mlirTypeIsAHostExecutionType,
                     mlirHostExecutionTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirHostExecutionTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none());

  mlir_type_subclass(m, "BufferType", mlirTypeIsAHostBufferType,
                     mlirHostBufferTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirHostBufferTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none());
}