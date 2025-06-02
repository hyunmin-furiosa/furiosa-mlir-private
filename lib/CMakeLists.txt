add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(ExecutionEngine)
add_subdirectory(Target)

set(LinkedLibs
  MLIRComplexDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRFuncToLLVM
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransformOps
  MLIRMLProgramDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRTosaToLinalg
  MLIRTransformDialect
  MLIRTransformDialectTransforms

  # Dialects.
  FuriosaMLIRHostDialect
  FuriosaMLIRFuriosaDialect
  FuriosaMLIRTaskDialect

  # Dialect passes.
  FuriosaMLIRFuncPasses
  FuriosaMLIRFuriosaPasses
  FuriosaMLIRLinalgPasses

  # Conversion passes.
  FuriosaMLIRFuncToFuriosaHost
  FuriosaMLIRFuriosaToFuriosaTask
  FuriosaMLIRLinalgToFuriosa
)

add_mlir_library(FuriosaMLIRInitAll
  InitAll.cpp

  LINK_LIBS PUBLIC
  ${LinkedLibs}
)

furiosa_mlir_target_includes(FuriosaMLIRInitAll)
