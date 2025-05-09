add_mlir_conversion_library(FuriosaMLIRFuncToFuriosaHost
  FuncToFuriosaHost.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/furiosa-mlir/Conversion/FuncToLLVM

  DEPENDS
  FuriosaMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRPass
)
