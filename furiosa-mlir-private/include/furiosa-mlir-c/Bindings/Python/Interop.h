#pragma once

// clang-format off
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir-c/Bindings/Python/Interop.h"
// clang-format on

#include "furiosa-mlir-c/ExecutionEngine/ExecutionEngine.h"

#define FURIOSA_MLIR_PYTHON_CAPSULE_EXECUTION_ENGINE                           \
  MAKE_MLIR_PYTHON_QUALNAME("execution_engine.ExecutionEngine._CAPIPtr")

static inline PyObject *
furiosaMlirPythonExecutionEngineToCapsule(FuriosaMlirExecutionEngine engine) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(engine),
                       FURIOSA_MLIR_PYTHON_CAPSULE_EXECUTION_ENGINE, NULL);
}

static inline FuriosaMlirExecutionEngine
furiosaMlirPythonCapsuleToExecutionEngine(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(
      capsule, FURIOSA_MLIR_PYTHON_CAPSULE_EXECUTION_ENGINE);
  FuriosaMlirExecutionEngine engine = {ptr};
  return engine;
}
