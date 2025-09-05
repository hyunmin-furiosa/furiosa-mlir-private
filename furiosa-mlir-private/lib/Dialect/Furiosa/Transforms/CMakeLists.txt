add_mlir_library(FuriosaMLIRFuriosaPasses
  AllocateAddress.cpp
  Deallocation.cpp
  LoadTensorRegisterFile.cpp
  PromoteSlicePartitionLoop.cpp

  DEPENDS
  FuriosaMLIRFuriosaPassesIncGen

  LINK_LIBS PUBLIC
  FuriosaMLIRFuriosaDialect
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

furiosa_mlir_target_includes(FuriosaMLIRFuriosaPasses)
