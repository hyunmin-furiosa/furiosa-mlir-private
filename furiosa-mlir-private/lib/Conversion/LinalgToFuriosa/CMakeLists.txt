add_mlir_conversion_library(FuriosaMLIRLinalgToFuriosa
  LinalgToFuriosa.cpp

  DEPENDS
  FuriosaMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRPass
)
