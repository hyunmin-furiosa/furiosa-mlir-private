add_mlir_translation_library(FuriosaMLIRTargetC
  FuriosaTaskToC.cpp
  TranslateRegistration.cpp

  LINK_LIBS PUBLIC
  FuriosaMLIRFuriosaDialect
  FuriosaMLIRTaskDialect
  FuriosaMLIRInitAll
)

target_link_libraries(FuriosaMLIRTargetC PRIVATE
  clangCodeGen
  clangFrontend
  lldELF
  LLVMAArch64AsmParser
  LLVMAArch64CodeGen
  LLVMObject
)

target_compile_definitions(obj.FuriosaMLIRTargetC PRIVATE
  CLANG_CROSS_COMPILE_INCLUDE="${LLVM_BUILD_DIR}/lib/clang/${LLVM_VERSION_MAJOR}/include"
  CLANG_CROSS_COMPILE_LIBPE="${CMAKE_SOURCE_DIR}/include/furiosa-mlir/Target/C/libpe"
)
