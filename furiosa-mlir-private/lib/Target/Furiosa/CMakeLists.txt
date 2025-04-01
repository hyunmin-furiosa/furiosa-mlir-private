add_mlir_translation_library(FuriosaMLIRTargetFuriosa
  FuriosaToBinary.cpp
  TranslateRegistration.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/furiosa-mlir/Target/Furiosa

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
