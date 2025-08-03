# The Furiosa-MLIR Project
The Furiosa-MLIR project aims to provide a compilation flow that converts arbitrary MLIR code into optimized hardware code that can run on [FuriosaAI Renegade](https://furiosa.ai/rngd).  

## Building Furiosa-MLIR

Build [device-runtime](https://github.com/furiosa-ai/device-runtime/) and [pert](https://github.com/furiosa-ai/device-runtime/tree/main/pert) at [22dd3fe](https://github.com/furiosa-ai/device-runtime/commit/22dd3fecea87965790f075cce12c19459e33ba78).
```shell
cargo build --release -p device-runtime-c # C bindings for device-runtime
cd pert
make pert
```

Build [npu-virtual-platform](https://github.com/furiosa-ai/npu-virtual-platform) at [9eba629](https://github.com/furiosa-ai/npu-virtual-platform/commit/9eba62989f72df68d5e755b227668e0f984dfd13).
```shell
make renegade DEFAULT_PERT_PATH=<device-runtime>/target/aarch64-unknown-none-softfloat/release/pert
```

Build [llvm-project](https://github.com/llvm/llvm-project) at [779868d](https://github.com/llvm/llvm-project/commit/779868de6975f6fd0ea17bb9a8e929037d3752d7).
```shell
cmake -G Ninja -B build llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="clang;lld;llvm;mlir" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
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
LIBRARY_PATH=$LIBRARY_PATH:<device-runtime>/target/release \
CPATH=$CPATH:<device-runtime>/target/release \
make LLVM_BUILD_DIR=<llvm-project>/build
```

## Using Furiosa-MLIR

Add required libraries and generated binaries to environment variables. 
```shell
export PATH=$PATH:<furiosa-mlir>/build/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<device-runtime>/target/release:<npu-virtual-platform>/build/renegade
```

Optimize and lower the example MLIR into furiosa host dialect. 
```shell
furiosa-mlir-opt test/Dialect/Furiosa/task.mlir -convert-func-to-furiosa-host
```

Translate the example MLIR into ARM C code. 
```shell
furiosa-mlir-translate test/Dialect/Furiosa/task.mlir -furiosa-to-arm-c
```

Run the example MLIR on target device. 
```shell
furiosa-mlir-runner test/Dialect/Host/host.mlir
```
