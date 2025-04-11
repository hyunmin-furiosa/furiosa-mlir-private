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
  func.func @kernel(%arg0: tensor<1024xf32, #furiosa.address<0x10000>>) -> (tensor<1024xf32, #furiosa.address<0x20000>>) attributes {address = #furiosa.address<0x0>, target = #furiosa.target<npu 0 pe 0:0>} {
    furiosa.rtosfr {sfr_address = 0 : i64, size = 1 : i64, value = 12424 : i64}
    furiosa.wait {dma_tag_id = 0 : i32, target_context = false, type = false}
    furiosa.dma_descriptor {desc_addr = 0x110000, opcode = 0, source_base = 0xC000010000, destination_base = 0x0010000000, source_limit = [4,1,1,1,1,1,1,1], source_stride = [256,0,0,0,0,0,0,0], destination_limit = [4,1,1,1,1,1,1,1], destination_stride = [256,0,0,0,0,0,0,0]}
    furiosa.dma {pe0_desc_addr = 0x110000, pe1_desc_addr = 0x110000, pe2_desc_addr = 0x110000, pe3_desc_addr = 0x110000, dma_tag_id = 0, profile = false, profile_id = 0}
    furiosa.dma_descriptor {desc_addr = 0x110100, opcode = 0, source_base = 0x0010000000, destination_base = 0xC000020000, source_limit = [4,1,1,1,1,1,1,1], source_stride = [256,0,0,0,0,0,0,0], destination_limit = [4,1,1,1,1,1,1,1], destination_stride = [256,0,0,0,0,0,0,0]}
    furiosa.dma {pe0_desc_addr = 0x110100, pe1_desc_addr = 0x110100, pe2_desc_addr = 0x110100, pe3_desc_addr = 0x110100, dma_tag_id = 0, profile = false, profile_id = 0}
    %0 = tensor.empty() : tensor<1024xf32, #furiosa.address<0x20000>>
    return %0 : tensor<1024xf32, #furiosa.address<0x20000>>
  }
}
```

```shell
furiosa-mlir-opt example.mlir | furiosa-mlir-translate -furiosa-to-binary
furiosa-runner furiosa.bin
```
