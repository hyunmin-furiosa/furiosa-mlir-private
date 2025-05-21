add_mlir_library(FuriosaMLIRHostDialect
  HostDialect.cpp
  HostOps.cpp

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
