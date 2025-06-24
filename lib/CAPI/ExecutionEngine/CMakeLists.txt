add_mlir_public_c_api_library(FuriosaMLIRCAPIExecutionEngine
  ExecutionEngine.cpp

  LINK_LIBS PUBLIC
  FuriosaMLIRExecutionEngine
  FuriosaMLIRTargetC
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
)
