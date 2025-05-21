#include "furiosa-mlir/Dialect/Furiosa/Transforms/PromoteSlicePartitionLoop.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"
#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

namespace mlir {
class Pass;

namespace furiosa {
#define GEN_PASS_DEF_FURIOSAPROMOTESLICEPARTITIONLOOPPASS
#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.h.inc"

using namespace mlir;
namespace {

struct SlicePartitionLoopPromotion : public OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForallOp op,
                                PatternRewriter &rewriter) const final;
};

struct PromoteSlicePartitionLoop
    : public impl::FuriosaPromoteSlicePartitionLoopPassBase<
          PromoteSlicePartitionLoop> {
  using Base::Base;

public:
  void runOnOperation() final;
};

} // namespace

LogicalResult
SlicePartitionLoopPromotion::matchAndRewrite(scf::ForallOp op,
                                             PatternRewriter &rewriter) const {
  rewriter.replaceAllUsesWith(op.getResults(), op.getOperands());
  rewriter.eraseOp(op.getTerminator());
  auto arguments =
      SmallVector<Value>(op.getRank()); // induction variables are removed
  for (auto operand : op.getOperands()) {
    arguments.push_back(operand);
  }
  rewriter.inlineBlockBefore(op.getBody(), op.getOperation(), arguments);

  return success();
}

void PromoteSlicePartitionLoop::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<SlicePartitionLoopPromotion>(patterns.getContext());

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

} // namespace furiosa
} // namespace mlir
