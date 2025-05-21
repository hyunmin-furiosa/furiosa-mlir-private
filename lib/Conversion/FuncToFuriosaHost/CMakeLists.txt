add_mlir_conversion_library(FuriosaMLIRFuncToFuriosaHost
  FuncToFuriosaHost.cpp

  DEPENDS
  FuriosaMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRPass
)
