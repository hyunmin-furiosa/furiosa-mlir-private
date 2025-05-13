add_mlir_library(FuriosaMLIRTaskDialect
  TaskDialect.cpp
  TaskOps.cpp

  DEPENDS
  FuriosaMLIRTaskOpsIncGen
  FuriosaMLIRTaskAttributesIncGen
  FuriosaMLIRTaskTypesIncGen

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRSupport
)

furiosa_mlir_target_includes(FuriosaMLIRTaskDialect)
