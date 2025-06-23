#pragma once

#include "device_runtime.h"

#include "furiosa-mlir/ExecutionEngine/ExecutionEngine.h"
#include "furiosa-mlir/ExecutionEngine/TensorDescriptor.h"
#include "furiosa-mlir/Target/C/FuriosaTaskToC.h"

namespace mlir::furiosa {

class ExecutionEngine;

#define CEIL(a, b) (((a + b - 1) / b) * b)

static constexpr auto MIN_BYTE_ARRAY_SIZE = 256;
static constexpr auto DRAM_ACCESS_WIDTH = 256;
using byte_array_t = SmallVector<std::uint8_t, MIN_BYTE_ARRAY_SIZE>;
using pe_program_t = SmallVector<device_runtime::Stmt *>;
using hal_program_t = SmallVector<device_runtime::Program *>;
using device_t = device_runtime::Device *;
using execution_t = device_runtime::Execution *;
using address_size_t = std::tuple<std::int64_t, std::int64_t>;

LogicalResult executeFunction(ExecutionEngine &engine, StringRef function_name,
                              std::int64_t num_args, std::int64_t num_inputs,
                              void **args);

} // namespace mlir::furiosa
