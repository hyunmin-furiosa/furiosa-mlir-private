add_llvm_executable(furiosa-mlir-opt furiosa-mlir-opt.cpp)

install(TARGETS furiosa-mlir-opt
EXPORT TorchMLIRTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT furiosa-mlir-opt)

target_link_libraries(furiosa-mlir-opt PRIVATE
  MLIROptLib
  MLIRTransforms
  FuriosaMLIRInitAll

  # Dialects.
  FuriosaMLIRFuriosaDialect
  FuriosaMLIRTaskDialect

  # Dialect passes.
  FuriosaMLIRFuriosaPasses
  FuriosaMLIRLinalgPasses

  # Conversion passes.
  FuriosaMLIRFuncToFuriosaHost
)
