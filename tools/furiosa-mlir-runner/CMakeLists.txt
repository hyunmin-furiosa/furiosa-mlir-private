add_llvm_executable(furiosa-mlir-runner furiosa-mlir-runner.cpp)

install(TARGETS furiosa-mlir-runner
EXPORT TorchMLIRTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT furiosa-mlir-runner)

target_link_libraries(furiosa-mlir-runner PRIVATE
  MLIRExecutionEngine
  MLIRJitRunner
  FuriosaMLIRExecutionEngine
  FuriosaMLIRFuriosaDialect
  FuriosaMLIRHostDialect
  FuriosaMLIRTargetFuriosa
)
