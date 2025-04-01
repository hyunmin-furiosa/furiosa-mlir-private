add_llvm_executable(furiosa-runner furiosa-runner.cpp)

install(TARGETS furiosa-runner
EXPORT TorchMLIRTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT furiosa-runner)

target_link_libraries(furiosa-runner PRIVATE
  LLVMSupport
  FuriosaMLIRExecutionEngine
)
