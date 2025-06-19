#include <list>

#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "furiosa-mlir-c/Bindings/Python/Interop.h"
#include "furiosa-mlir-c/ExecutionEngine/ExecutionEngine.h"

namespace nb = nanobind;
using namespace mlir;
using namespace mlir::python;

namespace {

/// Owning Wrapper around an ExecutionEngine.
class PyExecutionEngine {
public:
  PyExecutionEngine(FuriosaMlirExecutionEngine executionEngine)
      : executionEngine(executionEngine) {}
  PyExecutionEngine(PyExecutionEngine &&other) noexcept
      : executionEngine(other.executionEngine) {
    other.executionEngine.ptr = nullptr;
  }
  ~PyExecutionEngine() {
    if (!furiosaMlirExecutionEngineIsNull(executionEngine))
      furiosaMlirExecutionEngineDestroy(executionEngine);
  }
  FuriosaMlirExecutionEngine get() { return executionEngine; }

  void release() {
    executionEngine.ptr = nullptr;
    referencedObjects.clear();
  }
  nb::object getCapsule() {
    return nb::steal<nb::object>(
        furiosaMlirPythonExecutionEngineToCapsule(get()));
  }

  // Add an object to the list of referenced objects whose lifetime must exceed
  // those of the ExecutionEngine.
  void addReferencedObject(const nb::object &obj) {
    referencedObjects.push_back(obj);
  }

  static nb::object createFromCapsule(nb::object capsule) {
    FuriosaMlirExecutionEngine rawPm =
        furiosaMlirPythonCapsuleToExecutionEngine(capsule.ptr());
    if (furiosaMlirExecutionEngineIsNull(rawPm))
      throw nb::python_error();
    return nb::cast(PyExecutionEngine(rawPm), nb::rv_policy::move);
  }

private:
  FuriosaMlirExecutionEngine executionEngine;
  // We support Python ctypes closures as callbacks. Keep a list of the objects
  // so that they don't get garbage collected. (The ExecutionEngine itself
  // just holds raw pointers with no lifetime semantics).
  std::vector<nb::object> referencedObjects;
};

} // namespace

NB_MODULE(_furiosaMlirExecutionEngine, m) {
  m.doc() = "Furiosa-MLIR Execution Engine";

  nb::class_<PyExecutionEngine>(m, "ExecutionEngine")
      .def(
          "__init__",
          [](PyExecutionEngine &self, MlirModule module) {
            FuriosaMlirExecutionEngine engine =
                furiosaMlirExecutionEngineCreate(module);
            if (furiosaMlirExecutionEngineIsNull(engine))
              throw std::runtime_error(
                  "Failure while creating the ExecutionEngine.");
            new (&self) PyExecutionEngine(engine);
          },
          nb::arg("module"))
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyExecutionEngine::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyExecutionEngine::createFromCapsule)
      .def(
          "raw_invoke",
          [](PyExecutionEngine &self, const std::string &name,
             std::int64_t num_args, std::int64_t args) {
            MlirLogicalResult result = furiosaMlirExecutionEngineInvokePacked(
                self.get(), mlirStringRefCreate(name.c_str(), name.size()),
                num_args, reinterpret_cast<void **>(args));
            if (mlirLogicalResultIsFailure(result))
              throw std::runtime_error("Invocation failed.");
          },
          nb::arg("name"), nb::arg("num_args"), nb::arg("args"));
}
