add_mlir_translation_library(FuriosaMLIRTargetFuriosa
  FuriosaToBinary.cpp
  TranslateRegistration.cpp

  LINK_LIBS PUBLIC
  FuriosaMLIRFuriosaDialect
  FuriosaMLIRHostDialect
  FuriosaMLIRInitAll
)

target_link_libraries(FuriosaMLIRTargetFuriosa PRIVATE
  clangCodeGen
  clangFrontend
  lldELF
  LLVMAArch64AsmParser
  LLVMAArch64CodeGen
  LLVMObject
)

target_compile_definitions(obj.FuriosaMLIRTargetFuriosa PRIVATE
  CLANG_CROSS_COMPILE_INCLUDE="${LLVM_BUILD_DIR}/lib/clang/${LLVM_VERSION_MAJOR}/include"
  CLANG_CROSS_COMPILE_LIBPE="${CMAKE_SOURCE_DIR}/include/furiosa-mlir/Target/Furiosa/libpe"
)
