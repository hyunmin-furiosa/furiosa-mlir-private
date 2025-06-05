#include "furiosa-mlir/Dialect/Furiosa/Transforms/Deallocation.h"
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
#define GEN_PASS_DEF_FURIOSADEALLOCATIONPASS
#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.h.inc"

using namespace mlir;
namespace {

struct AllocOpDeallocation : public OpRewritePattern<furiosa::AllocOp> {
  using OpRewritePattern<furiosa::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(furiosa::AllocOp alloc_op,
                                PatternRewriter &rewriter) const final;
};

struct Deallocation : public impl::FuriosaDeallocationPassBase<Deallocation> {
  using Base::Base;

public:
  void runOnOperation() final;
};

} // namespace

LogicalResult
AllocOpDeallocation::matchAndRewrite(furiosa::AllocOp alloc_op,
                                     PatternRewriter &rewriter) const {
  // If alloc already has a dealloc, ignore it
  for (auto user : alloc_op->getUsers()) {
    if (llvm::isa<furiosa::DeallocOp>(user)) {
      return rewriter.notifyMatchFailure(alloc_op, "alloc already has dealloc");
    }
  }
  rewriter.setInsertionPoint(alloc_op->getBlock()->getTerminator());
  rewriter.create<furiosa::DeallocOp>(alloc_op.getLoc(), alloc_op.getResult());

  return success();
}

void Deallocation::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<AllocOpDeallocation>(patterns.getContext());

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

} // namespace furiosa
} // namespace mlir
