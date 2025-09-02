#include "furiosa-mlir/Dialect/Furiosa/Transforms/LoadTensorRegisterFile.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/Utils.h"
#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

namespace mlir {
class Pass;

namespace furiosa {
#define GEN_PASS_DEF_FURIOSALOADTENSORREGISTERFILEPASS
#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.h.inc"

using namespace mlir;
namespace {

struct ContractOpLowering : public OpRewritePattern<linalg::ContractOp> {
public:
  ContractOpLowering(MLIRContext *context)
      : OpRewritePattern<linalg::ContractOp>(context) {}

  LogicalResult matchAndRewrite(linalg::ContractOp op,
                                PatternRewriter &rewriter) const final;
};

struct LoadTensorRegisterFile
    : public impl::FuriosaLoadTensorRegisterFilePassBase<
          LoadTensorRegisterFile> {
  using Base::Base;

public:
  void runOnOperation() final;
};

} // namespace

LogicalResult
ContractOpLowering::matchAndRewrite(linalg::ContractOp op,
                                    PatternRewriter &rewriter) const {
  assert(op.getInputs().size() == 2);
  assert(op.getOutputs().size() == 1);

  auto trf_attr = furiosa::MemoryTypeAttr::get(rewriter.getContext(),
                                               furiosa::MemoryType::trf);
  static constexpr auto TRF_INPUT_INDEX = 0;
  auto trf_input = op.getInputs()[TRF_INPUT_INDEX];
  auto trf_type = llvm::cast<RankedTensorType>(trf_input.getType())
                      .cloneWithEncoding(trf_attr);
  if (*getMemoryType(trf_input) == furiosa::MemoryType::trf) {
    return failure();
  }

  auto sram_alloc_op =
      llvm::dyn_cast_or_null<furiosa::AllocOp>(trf_input.getDefiningOp());
  auto trf_alloc_op =
      rewriter.create<furiosa::AllocOp>(op.getLoc(), trf_type, IntegerAttr());
  rewriter.create<furiosa::LoadTrfOp>(op.getLoc(), sram_alloc_op.getResult(),
                                      trf_alloc_op.getResult());
  op.setOperand(TRF_INPUT_INDEX, trf_alloc_op);

  return success();
}

void LoadTensorRegisterFile::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<ContractOpLowering>(patterns.getContext());

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

} // namespace furiosa
} // namespace mlir
