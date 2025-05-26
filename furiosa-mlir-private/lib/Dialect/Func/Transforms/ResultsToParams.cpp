#include "furiosa-mlir/Dialect/Func/Transforms/ResultsToParams.h"
#include "furiosa-mlir/Dialect/Func/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
class Pass;

namespace furiosa {
#define GEN_PASS_DEF_FUNCRESULTSTOPARAMSPASS
#include "furiosa-mlir/Dialect/Func/Transforms/Passes.h.inc"

using namespace mlir;
namespace {

struct FuncOpTransformation : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const final;
};

struct ResultsToParams
    : public impl::FuncResultsToParamsPassBase<ResultsToParams> {
  using Base::Base;

public:
  void runOnOperation() final;
};

} // namespace

LogicalResult
FuncOpTransformation::matchAndRewrite(func::FuncOp func_op,
                                      PatternRewriter &rewriter) const {
  // TODO

  return success();
}

void ResultsToParams::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<FuncOpTransformation>(patterns.getContext());

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

} // namespace furiosa
} // namespace mlir
