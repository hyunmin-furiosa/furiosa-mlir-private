add_mlir_public_c_api_library(FuriosaMLIRCAPIRegisterEverything
  RegisterEverything.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/furiosa-mlir-c/

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  FuriosaMLIRFuriosaDialect
  FuriosaMLIRFuriosaPasses
  FuriosaMLIRInitAll
)
