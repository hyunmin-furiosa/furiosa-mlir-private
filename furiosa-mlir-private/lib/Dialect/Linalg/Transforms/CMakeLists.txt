add_mlir_library(FuriosaMLIRLinalgPasses
  GeneralizeToContractOps.cpp

  DEPENDS
  FuriosaMLIRLinalgPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
)

furiosa_mlir_target_includes(FuriosaMLIRLinalgPasses)
