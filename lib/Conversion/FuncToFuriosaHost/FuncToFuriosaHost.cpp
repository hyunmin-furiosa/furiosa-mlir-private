#include "furiosa-mlir/Conversion/FuncToFuriosaHost/FuncToFuriosaHost.h"
#include "furiosa-mlir/Dialect/Host/IR/HostDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class Pass;

namespace furiosa {
#define GEN_PASS_DEF_CONVERTFUNCTOFURIOSAHOSTPASS
#include "furiosa-mlir/Conversion/Passes.h.inc"

using namespace mlir;
namespace {

template <typename Op>
struct FuncOpToFuriosaHostOp : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};

struct ConvertFuncToFuriosaHost
    : public impl::ConvertFuncToFuriosaHostPassBase<ConvertFuncToFuriosaHost> {
  using Base::Base;

public:
  void runOnOperation() final;
};

} // namespace

template <typename Op>
LogicalResult
FuncOpToFuriosaHostOp<Op>::matchAndRewrite(Op op,
                                           PatternRewriter &rewriter) const {
  return success();
}

void ConvertFuncToFuriosaHost::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<FuncOpToFuriosaHostOp<func::CallOp>>(patterns.getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

} // namespace furiosa
} // namespace mlir
