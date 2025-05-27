#include "furiosa-mlir/Conversion/FuriosaToFuriosaTask/FuriosaToFuriosaTask.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskDialect.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class Pass;

namespace furiosa {
#define GEN_PASS_DEF_CONVERTFURIOSATOFURIOSATASKPASS
#include "furiosa-mlir/Conversion/Passes.h.inc"

using namespace mlir;
namespace {

struct DmaOpLowering : public OpRewritePattern<furiosa::DmaOp> {
public:
  DmaOpLowering(MLIRContext *context)
      : OpRewritePattern<furiosa::DmaOp>(context) {}

  LogicalResult matchAndRewrite(furiosa::DmaOp op,
                                PatternRewriter &rewriter) const final;
};

struct ConvertFuriosaToFuriosaTask
    : public impl::ConvertFuriosaToFuriosaTaskPassBase<
          ConvertFuriosaToFuriosaTask> {
  using Base::Base;

public:
  void runOnOperation() final;
};

} // namespace

LogicalResult DmaOpLowering::matchAndRewrite(furiosa::DmaOp alloc_op,
                                             PatternRewriter &rewriter) const {
  alloc_op.dump();
  return success();
}

void ConvertFuriosaToFuriosaTask::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<furiosa::task::TaskDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<DmaOpLowering>(patterns.getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

} // namespace furiosa
} // namespace mlir
