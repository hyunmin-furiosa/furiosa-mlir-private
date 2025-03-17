# The Furiosa-MLIR Project
The Furiosa-MLIR project is based on [torch-mlir](https://github.com/llvm/torch-mlir).

## Building Furiosa-MLIR

Build [llvm-project](https://github.com/llvm/llvm-project) at [6d847b1](https://github.com/llvm/llvm-project/commit/6d847b1aada50d59c3e29f2e7eff779c0ee8182c).
```shell
cmake -G Ninja -B build llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  `# use clang`\
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  `# use ccache to cache build results` \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  `# use LLD to link in seconds, rather than minutes` \
  `# if using clang <= 13, replace --ld-path=ld.lld with -fuse-ld=lld` \
  -DCMAKE_EXE_LINKER_FLAGS_INIT="--ld-path=ld.lld" \
  -DCMAKE_MODULE_LINKER_FLAGS_INIT="--ld-path=ld.lld" \
  -DCMAKE_SHARED_LINKER_FLAGS_INIT="--ld-path=ld.lld"
```

Build Furiosa-MLIR project.
```shell
export $LLVM_INSTALL_DIR=<llvm-project>/build

cmake -GNinja -Bbuild . \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir/" \
  -DLLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm/" \
  -DLLVM_TARGETS_TO_BUILD=host \
  `# use clang`\
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  `# use ccache to cache build results` \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  `# use LLD to link in seconds, rather than minutes` \
  `# if using clang <= 13, replace --ld-path=ld.lld with -fuse-ld=lld` \
  -DCMAKE_EXE_LINKER_FLAGS_INIT="--ld-path=ld.lld" \
  -DCMAKE_MODULE_LINKER_FLAGS_INIT="--ld-path=ld.lld" \
  -DCMAKE_SHARED_LINKER_FLAGS_INIT="--ld-path=ld.lld" \
  `# other options` \
  -DTORCH_MLIR_ENABLE_STABLEHLO=OFF \
  -DTORCH_MLIR_ENABLE_REFBACKEND=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
```
