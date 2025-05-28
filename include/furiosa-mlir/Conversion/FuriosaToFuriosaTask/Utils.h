#pragma once

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::furiosa {

static constexpr std::uint64_t SFR_BROADCAST = 0xff0000;
static constexpr std::uint64_t CONTEXT_ID_OFFSET = 0x800;
static constexpr std::uint64_t MAIN_FETCH_SIZE = 0x20;
static constexpr std::uint64_t SUB_FETCH_SIZE = 0x8;
static constexpr std::uint64_t TENSOR_REGISTER_FILE_ROW_SIZE = 0x20;

struct DataPathUnitRoute {
  static constexpr std::uint64_t DotProductEngine = 0;
  static constexpr std::uint64_t VectorEngine = 1;
  static constexpr std::uint64_t TransposeEngine = 2;
  static constexpr std::uint64_t Commit = 3;
  static constexpr std::uint64_t TensorRegisterFile = 4;
  static constexpr std::uint64_t VectorRegisterFile = 5;
  static constexpr std::uint64_t RegisterConfigUnit = 6;
};

struct SubUnit {
  static constexpr std::uint64_t DataMemorySlice = 0;
  static constexpr std::uint64_t FetchUnit = 1;
  static constexpr std::uint64_t DotProductEngine = 2;
  static constexpr std::uint64_t VectorEngine = 3;
  static constexpr std::uint64_t TransposeEngine = 4;
  static constexpr std::uint64_t CommitUnit = 5;
  static constexpr std::uint64_t SubFetchUnit = 6;
  static constexpr std::uint64_t SubCommitUnit = 7;
  static constexpr std::uint64_t TensorRegisterFile = 8;
  static constexpr std::uint64_t VectorRegisterFile = 9;
  static constexpr std::uint64_t RegisterConfig = 10;
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

} // namespace mlir::furiosa
