add_llvm_executable(furiosa-mlir-translate furiosa-mlir-translate.cpp)

install(TARGETS furiosa-mlir-translate
EXPORT TorchMLIRTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT furiosa-mlir-translate)

target_link_libraries(furiosa-mlir-translate PRIVATE
  MLIRTranslateLib
  FuriosaMLIRFuriosaDialect
  FuriosaMLIRFuriosaPasses
  FuriosaMLIRTargetFuriosa
)
