add_mlir_library(FuriosaMLIRHostDialect
  HostDialect.cpp
  HostOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/furiosa-mlir/Dialect/Host

  DEPENDS
  FuriosaMLIRHostOpsIncGen
  FuriosaMLIRHostAttributesIncGen
  FuriosaMLIRHostTypesIncGen

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRSupport
)

furiosa_mlir_target_includes(FuriosaMLIRHostDialect)
