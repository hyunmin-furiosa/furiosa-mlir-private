torch_mlir_enable_werror()

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
  FuriosaMLIRLinalgToFuriosa
)

if(TORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND LinkedLibs StablehloLinalgTransforms StablehloPasses)
endif()

if(TORCH_MLIR_ENABLE_TOSA)
  list(APPEND LinkedLibs MLIRTosaDialect)
endif()

if(TORCH_MLIR_ENABLE_REFBACKEND)
  add_subdirectory(RefBackend)
  list(APPEND LinkedLibs TorchMLIRRefBackend)
endif()

add_mlir_library(FuriosaMLIRInitAll
  InitAll.cpp

  LINK_LIBS PUBLIC
  ${LinkedLibs}
)

furiosa_mlir_target_includes(FuriosaMLIRInitAll)
