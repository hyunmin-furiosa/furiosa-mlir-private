add_mlir_library(FuriosaMLIRFuriosaDialect
  FuriosaDialect.cpp

  DEPENDS
  FuriosaMLIRFuriosaOpsIncGen
  FuriosaMLIRFuriosaAttributesIncGen
  FuriosaMLIRFuriosaTypesIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRBytecodeOpInterface
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRViewLikeInterface
)

furiosa_mlir_target_includes(FuriosaMLIRFuriosaDialect)
