add_mlir_translation_library(FuriosaMLIRTargetFuriosa
    FuriosaToBinary.cpp
    TranslateRegistration.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/furiosa-mlir/Target/Furiosa

    LINK_LIBS PUBLIC
    FuriosaMLIRFuriosaDialect
)
