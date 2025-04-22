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

Run generated binaries in `build/bin`.

```llvm
// example.mlir
module {
  func.func @kernel(%arg0: tensor<256xf32, #furiosa.address<0x10000>>) -> (tensor<256xf32, #furiosa.address<0x20000>>) attributes {address = #furiosa.address<0x0>, target = #furiosa.target<npu 0 pe 0:0>} {
    furiosa.tuc.rtosfr {sfr_address = 0 : i64, size = 1 : i64, value = 12424 : i64}
    furiosa.tuc.wait {dma_tag_id = 0 : i32, target_context = false, type = false}
    %0 = furiosa.task.dma_descriptor {opcode = 0, source_base = 0xC000010000, destination_base = 0x0010000000, source_limits = [4,1,1,1,1,1,1,1], source_strides = [256,0,0,0,0,0,0,0], destination_limits = [4,1,1,1,1,1,1,1], destination_strides = [256,0,0,0,0,0,0,0]}
    furiosa.task.dmaw %0 {dma_tag_id = 0, profile = false, profile_id = 0}
    furiosa.tuc.wait {dma_tag_id = 0 : i32, type = true, target_context = false}
    %1 = furiosa.task.sfr.sub_fetch {base = 0x0, type_conversion = 0, num_zero_points = 0, zero_point0 = 0, zero_point1 = 0, limits = [128,1,1,1,1,1,1,1], strides = [8,0,0,0,0,0,0,0], flit_count = 128, words_per_packet = 1, topology = 0, outer_slice_log_size = 0, outer_dim0_log_size = 0, outer_dim1_log_size = 0, outer_dim0_chunk_size = 0, outer_dim1_chunk_size = 0, custom_snoop_bitmap = [0,0,0,0] }
    %2 = furiosa.task.sfr.sub_commit {mode = 0, packet_valid_count = 8, base = 0x10000, commit_in_size = 8, commit_data = 0, limits = [128,1,1,1,1,1,1,1], strides = [8,0,0,0,0,0,0,0], slice_enable_bitmap = [0xffffffffffffffff,0xffffffffffffffff,0xffffffffffffffff,0xffffffffffffffff] }
    %3 = furiosa.task.sfr.sub_data_path {route = 0x08}
    furiosa.task.mtosfr %1 {sfr_address = 0xff0100}
    furiosa.task.mtosfr %2 {sfr_address = 0xff0198}
    furiosa.task.mtosfr %3 {sfr_address = 0xff0170}
    furiosa.tuc.exec {subunit_bitmap = 0x0c1 : i32, context_id = false, target_context = true}
    furiosa.tuc.wait {dma_tag_id = 0 : i32, type = false, target_context = true}
    %4 = furiosa.task.dma_descriptor {opcode = 0, source_base = 0x0010010000, destination_base = 0xC000020000, source_limits = [4,1,1,1,1,1,1,1], source_strides = [256,0,0,0,0,0,0,0], destination_limits = [4,1,1,1,1,1,1,1], destination_strides = [256,0,0,0,0,0,0,0]}
    furiosa.task.dmaw %4 {dma_tag_id = 1, profile = false, profile_id = 0}
    furiosa.tuc.wait {dma_tag_id = 1 : i32, type = true, target_context = false}
    %5 = tensor.empty() : tensor<256xf32, #furiosa.address<0x20000>>
    return %5 : tensor<256xf32, #furiosa.address<0x20000>>
  }
}

```

```shell
furiosa-mlir-opt example.mlir | furiosa-mlir-translate -furiosa-to-binary
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<furiosa-torch>/target/release:<npu-virtual-platform>/build/renegade \
furiosa-runner furiosa.bin
```
