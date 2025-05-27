add_mlir_conversion_library(FuriosaMLIRFuriosaToFuriosaTask
  FuriosaToFuriosaTask.cpp

  DEPENDS
  FuriosaMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
)
