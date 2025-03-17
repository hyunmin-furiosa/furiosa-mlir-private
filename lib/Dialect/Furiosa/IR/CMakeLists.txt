add_mlir_library(FuriosaMLIRFuriosaDialect
  FuriosaDialect.cpp
  FuriosaOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/furiosa-mlir/Dialect/Furiosa

  DEPENDS
  FuriosaMLIRFuriosaOpsIncGen

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
