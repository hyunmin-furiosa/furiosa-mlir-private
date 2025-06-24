add_mlir_public_c_api_library(FuriosaMLIRCAPIFuriosa
  Furiosa.cpp
  FuriosaPasses.cpp
  Host.cpp
  Task.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  FuriosaMLIRFuriosaPassesIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  FuriosaMLIRFuriosaDialect
  FuriosaMLIRFuriosaPasses
  FuriosaMLIRHostDialect
  FuriosaMLIRTaskDialect
  MLIRPass
)
