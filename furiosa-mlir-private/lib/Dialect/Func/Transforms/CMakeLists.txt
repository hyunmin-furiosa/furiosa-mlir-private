add_mlir_library(FuriosaMLIRFuncPasses
  ResultsToParams.cpp

  DEPENDS
  FuriosaMLIRFuncPassesIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
)

furiosa_mlir_target_includes(FuriosaMLIRFuncPasses)
