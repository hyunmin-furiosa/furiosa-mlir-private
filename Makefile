# Tools
CLANG_FORMAT := clang-format-19
CMAKE := cmake
MAKE := make
NINJA := Ninja

# Paths and options
BUILD_DIR ?= $(CURDIR)/build
BUILD_TYPE ?= Release
JOBS := 8
LLVM_BUILD_DIR ?= $(LLVM_BUILD_DIR)
TARGETS := $(shell find . -path ./build -prune -type f -o -iname "*.h" -o -iname "*.cpp" -o -iname "*.td")

# CMake flags
FLAGS := -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
FLAGS += -DCLANG_DIR="$(LLVM_BUILD_DIR)/lib/cmake/clang/"
FLAGS += -DLLD_DIR="$(LLVM_BUILD_DIR)/lib/cmake/lld/"
FLAGS += -DLLVM_DIR="$(LLVM_BUILD_DIR)/lib/cmake/llvm/"
FLAGS += -DMLIR_DIR="$(LLVM_BUILD_DIR)/lib/cmake/mlir/"
FLAGS += -DLLVM_BUILD_DIR="$(LLVM_BUILD_DIR)"
FLAGS += -DLLVM_TARGETS_TO_BUILD=host
# use clang
FLAGS += -DCMAKE_C_COMPILER=clang
FLAGS += -DCMAKE_CXX_COMPILER=clang++
# use ccache to cache build results
FLAGS += -DCMAKE_C_COMPILER_LAUNCHER=ccache
FLAGS += -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
# use LLD to link in seconds, rather than minutes
# if using clang <= 13, replace --ld-path=ld.lld with -fuse-ld=lld
FLAGS += -DCMAKE_EXE_LINKER_FLAGS_INIT="--ld-path=ld.lld"
FLAGS += -DCMAKE_MODULE_LINKER_FLAGS_INIT="--ld-path=ld.lld"
FLAGS += -DCMAKE_SHARED_LINKER_FLAGS_INIT="--ld-path=ld.lld"
FLAGS += -DTORCH_MLIR_ENABLE_STABLEHLO=OFF
FLAGS += -DTORCH_MLIR_ENABLE_REFBACKEND=OFF
FLAGS += -DMLIR_ENABLE_BINDINGS_PYTHON=OFF

furiosa-mlir: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel $(JOBS)

furiosa-mlir-doc: configure
	$(CMAKE) --build $(BUILD_DIR) --target mlir-doc --parallel $(JOBS)
	cp $(BUILD_DIR)/docs/Furiosa/*.md $(CURDIR)/docs

configure:
	mkdir -p $(BUILD_DIR)
	$(CMAKE) . -G $(NINJA) -B $(BUILD_DIR) $(FLAGS)

format: format-fix

format-fix:
	$(MAKE) _$@ -j$(JOBS) --output-sync --no-print-directory

_format-fix: $(addsuffix .format-fix, $(TARGETS))

%.format-fix: %
	$(CLANG_FORMAT) $< -i

clean:
	rm -rf $(BUILD_DIR)