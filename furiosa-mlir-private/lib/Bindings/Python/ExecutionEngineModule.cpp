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
            new (&self) PyExecutionEngine(engine);
          },
          nb::arg("module"));
}
