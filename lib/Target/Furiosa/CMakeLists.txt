add_mlir_translation_library(FuriosaMLIRTargetFuriosa
  Binary.cpp
  FuriosaToBinary.cpp
  TranslateRegistration.cpp

  LINK_LIBS PUBLIC
  FuriosaMLIRFuriosaDialect
)

target_link_libraries(FuriosaMLIRTargetFuriosa PRIVATE
  clangCodeGen
  clangFrontend
  LLVMAArch64CodeGen
  LLVMObject
)

target_compile_definitions(obj.FuriosaMLIRTargetFuriosa PRIVATE
  CLANG_CROSS_COMPILE_INCLUDE_DIRS="${LLVM_BUILD_DIR}/lib/clang/${LLVM_VERSION_MAJOR}/include"
)
