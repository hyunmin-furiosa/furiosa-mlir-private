#include <optional>

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
  m.doc() = "Furiosa-MLIR Furiosa Dialect";

  mlir_type_subclass(m, "BufferType", mlirTypeIsAFuriosaBufferType,
                     mlirFuriosaBufferTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirFuriosaBufferTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none());

  mlir_attribute_subclass(m, "TargetAttr", mlirAttributeIsAFuriosaTargetAttr)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context, std::uint64_t npu,
             std::uint64_t peBegin, std::uint64_t peEnd) {
            return cls(mlirFuriosaTargetAttrGet(context, npu, peBegin, peEnd));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none(),
          nb::arg("npu"), nb::arg("pe_begin"), nb::arg("pe_end"));

  mlir_attribute_subclass(m, "MappingAttr", mlirAttributeIsAFuriosaMappingAttr)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirFuriosaMappingAttrGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none());

  nb::enum_<mlir::furiosa::MemoryType>(m, "MemoryType")
      .value("dram", mlir::furiosa::MemoryType::dram)
      .value("sram", mlir::furiosa::MemoryType::sram)
      .value("trf", mlir::furiosa::MemoryType::trf)
      .value("vrf", mlir::furiosa::MemoryType::vrf);

  mlir_attribute_subclass(m, "TensorAttr", mlirAttributeIsAFuriosaTensorAttr)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context,
             mlir::furiosa::MemoryType memory_type,
             std::optional<MlirAttribute> memory_map) {
            return cls(mlirFuriosaTensorAttrGet(
                context, memory_type,
                memory_map.has_value() ? *memory_map : MlirAttribute{nullptr}));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none(),
          nb::arg("memory_type"), nb::arg("memory_map").none() = nb::none());
}
