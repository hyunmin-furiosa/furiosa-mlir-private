#pragma once

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::furiosa {

static constexpr std::uint64_t SFR_BROADCAST = 0xff0000;
static constexpr std::uint64_t CONTEXT_ID_OFFSET = 0x800;
static constexpr std::uint64_t MAIN_FETCH_SIZE = 0x20;
static constexpr std::uint64_t SUB_FETCH_SIZE = 0x8;
static constexpr std::uint64_t TENSOR_REGISTER_FILE_ROW_SIZE = 0x20;
static constexpr std::uint64_t SUB_FETCH_WORDS_PER_PACKET = 4;
static constexpr std::uint64_t DRAM_BASE = 0xc000000000;
static constexpr std::uint64_t SRAM_BASE = 0x10000000;

struct DataPathUnitRoute {
  static constexpr std::uint64_t DotProductEngine = 1 << 0;
  static constexpr std::uint64_t VectorEngine = 1 << 1;
  static constexpr std::uint64_t TransposeEngine = 1 << 2;
  static constexpr std::uint64_t Commit = 1 << 3;
  static constexpr std::uint64_t TensorRegisterFile = 1 << 4;
  static constexpr std::uint64_t VectorRegisterFile = 1 << 5;
  static constexpr std::uint64_t RegisterConfigUnit = 1 << 6;
};

struct SubUnit {
  static constexpr std::uint64_t DataMemorySlice = 1 << 0;
  static constexpr std::uint64_t FetchUnit = 1 << 1;
  static constexpr std::uint64_t DotProductEngine = 1 << 2;
  static constexpr std::uint64_t VectorEngine = 1 << 3;
  static constexpr std::uint64_t TransposeEngine = 1 << 4;
  static constexpr std::uint64_t CommitUnit = 1 << 5;
  static constexpr std::uint64_t SubFetchUnit = 1 << 6;
  static constexpr std::uint64_t SubCommitUnit = 1 << 7;
  static constexpr std::uint64_t TensorRegisterFile = 1 << 8;
  static constexpr std::uint64_t VectorRegisterFile = 1 << 9;
  static constexpr std::uint64_t RegisterConfig = 1 << 10;
};

FailureOr<std::int64_t> getAddress(Value value) {
  auto defining_op = value.getDefiningOp();
  furiosa::AllocOp alloc_op;
  if (llvm::isa<furiosa::AllocOp>(defining_op)) {
    alloc_op = llvm::dyn_cast_or_null<furiosa::AllocOp>(defining_op);
  } else if (llvm::isa<linalg::ContractOp>(defining_op)) {
    auto contract_op = llvm::dyn_cast_or_null<linalg::ContractOp>(defining_op);
    assert(contract_op.getOutputs().size() == 1 &&
           "contract op should have exactly one output");
    auto contract_output_op = contract_op.getOutputs()[0].getDefiningOp();
    if (llvm::isa<furiosa::AllocOp>(contract_output_op)) {
      alloc_op = llvm::dyn_cast_or_null<furiosa::AllocOp>(contract_output_op);
    }
  }
  if (alloc_op && alloc_op->hasAttr("address")) {
    return alloc_op->getAttrOfType<IntegerAttr>("address").getInt();
  } else {
    return failure();
  }
}

FailureOr<MemoryType> getMemoryType(Value value) {
  auto tensor_type = llvm::dyn_cast_or_null<RankedTensorType>(value.getType());
  if (!tensor_type)
    return failure();

  if (auto encoding = tensor_type.getEncoding()) {
    return llvm::dyn_cast_or_null<MemoryTypeAttr>(encoding).getValue();
  } else {
    return MemoryType::dram; // Default to DRAM if no encoding is specified
  }
}

} // namespace mlir::furiosa
