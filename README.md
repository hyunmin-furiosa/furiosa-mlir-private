# The Furiosa-MLIR Project
The Furiosa-MLIR project aims to provide a compilation flow that converts arbitrary MLIR code into optimized hardware code that can run on [FuriosaAI Renegade](https://furiosa.ai/rngd).  

## Building Furiosa-MLIR

Build [llvm-project](https://github.com/llvm/llvm-project) at [6d847b1](https://github.com/llvm/llvm-project/commit/6d847b1aada50d59c3e29f2e7eff779c0ee8182c).
```shell
cmake -G Ninja -B build llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="clang;llvm;mlir" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DLLVM_TARGETS_TO_BUILD="host;AArch64" \
  `# use clang`\
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  `# use ccache to cache build results` \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  `# use LLD to link in seconds, rather than minutes` \
  `# if using clang <= 13, replace --ld-path=ld.lld with -fuse-ld=lld` \
  -DCMAKE_EXE_LINKER_FLAGS_INIT="--ld-path=ld.lld" \
  -DCMAKE_MODULE_LINKER_FLAGS_INIT="--ld-path=ld.lld" \
  -DCMAKE_SHARED_LINKER_FLAGS_INIT="--ld-path=ld.lld"
cmake --build build -j 16
```

Build Furiosa-MLIR project.
```shell
make LLVM_BUILD_DIR=<llvm-project>/build
```

## Using Furiosa-MLIR

Run generated binaries in `build/bin`.

```llvm
// example.mlir
module {
  func.func @kernel(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> (tensor<1024xf32>, tensor<1024xf32>) attributes {target = "renegade"} {
    furiosa.rtosfr {sfr_address = 0 : i64, size = 1 : i64, value = 12424 : i64}
    furiosa.wait {dma_tag_id = 0 : i32, target_context = false, type = false}
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = tensor.empty() : tensor<1024xf32>
    return %0, %1 : tensor<1024xf32>, tensor<1024xf32>
  }
  func.func @main() {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = tensor.empty() : tensor<1024xf32>
    %2:2 = call @kernel(%0, %1) : (tensor<1024xf32>, tensor<1024xf32>) -> (tensor<1024xf32>, tensor<1024xf32>)
    return
  }
}
```

```shell
furiosa-mlir-opt example.mlir | furiosa-mlir-translate -furiosa-to-binary
furiosa-runner furiosa.bin
```
