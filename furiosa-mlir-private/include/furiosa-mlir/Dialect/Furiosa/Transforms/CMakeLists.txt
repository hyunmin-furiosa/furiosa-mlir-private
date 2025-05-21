set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Furiosa)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Furiosa)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Furiosa)
add_public_tablegen_target(FuriosaMLIRFuriosaPassesIncGen)

add_mlir_doc(Passes FuriosaPasses Furiosa/ -gen-pass-doc)
