# The Furiosa-MLIR Project
The Furiosa-MLIR project aims to provide a compilation flow that converts arbitrary MLIR code into optimized hardware code that can run on [FuriosaAI Renegade](https://furiosa.ai/rngd).  

## Building Furiosa-MLIR

Build [furiosa-torch](https://github.com/furiosa-ai/furiosa-torch) at [0aa4cdf](https://github.com/furiosa-ai/furiosa-torch/commit/0aa4cdf5f29483abdded2b4d956d54cd423d6716).
```shell
cargo build --release
```

Build pert in [device-runtime](https://github.com/furiosa-ai/device-runtime/) at [6d67166](https://github.com/furiosa-ai/device-runtime/commit/6d671664f6823967e69a8c49b729ad0ef6ff1f80)
```shell
cd pert
make pert
```

Build [npu-virtual-platform](https://github.com/furiosa-ai/npu-virtual-platform) at [b95cd40](https://github.com/furiosa-ai/npu-virtual-platform/commit/b95cd408fc21d37389c19afa8111504160fb937e)
```shell
make renegade DEFAULT_PERT_PATH=<device-runtime>/target/aarch64-unknown-none-softfloat/release/pert
```

Build [llvm-project](https://github.com/llvm/llvm-project) at [dda4b96](https://github.com/llvm/llvm-project/commit/dda4b968e77e1bb2c319bf2d523de3b5c4ccbb23).
```shell
cmake -G Ninja -B build llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="clang;lld;llvm;mlir" \
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
LIBRARY_PATH=$LIBRARY_PATH:<furiosa-torch>/target/release \
CPATH=$CPATH:<furiosa-torch>/cpp_extensions/include \
make LLVM_BUILD_DIR=<llvm-project>/build
```

## Using Furiosa-MLIR

Generated binaries are located in `build/bin`.

Translate the example MLIR into ARM C code. 
```shell
furiosa-mlir-translate test/Dialect/Furiosa/example.mlir -furiosa-to-arm-c
```

Run the example MLIR on target device. 
```shell
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<furiosa-torch>/target/release:<npu-virtual-platform>/build/renegade \
furiosa-mlir-runner test/Dialect/Host/example.mlir
```
