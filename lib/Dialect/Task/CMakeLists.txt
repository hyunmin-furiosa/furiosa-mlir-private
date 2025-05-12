add_mlir_library(FuriosaMLIRTaskDialect
  TaskDialect.cpp
  TaskOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/furiosa-mlir/Dialect/Task

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
