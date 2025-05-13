#include "furiosa-mlir/Dialect/Linalg/Transforms/GeneralizeToContractOps.h"
#include "furiosa-mlir/Dialect/Linalg/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
class Pass;

namespace furiosa {
#define GEN_PASS_DEF_LINALGGENERALIZETOCONTRACTOPSPASS
#include "furiosa-mlir/Dialect/Linalg/Transforms/Passes.h.inc"

using namespace mlir;
namespace {

struct GeneralizationToContractOps
    : public OpInterfaceRewritePattern<linalg::ContractionOpInterface> {
  using OpInterfaceRewritePattern<
      linalg::ContractionOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::ContractionOpInterface op,
                                PatternRewriter &rewriter) const final;
};

struct GeneralizeToContractOps
    : public impl::LinalgGeneralizeToContractOpsPassBase<
          GeneralizeToContractOps> {
  using Base::Base;

public:
  void runOnOperation() final;
};

} // namespace

LogicalResult
GeneralizationToContractOps::matchAndRewrite(linalg::ContractionOpInterface op,
                                             PatternRewriter &rewriter) const {
  if (llvm::isa<linalg::ContractOp>(op))
    return rewriter.notifyMatchFailure(op, "already a contract op");

  auto linalgOp = llvm::cast<linalg::LinalgOp>(op.getOperation());
  rewriter.replaceOpWithNewOp<linalg::ContractOp>(
      linalgOp,
      ValueRange{linalgOp.getDpsInputs()[0], linalgOp.getDpsInputs()[1]},
      ValueRange{linalgOp.getDpsInits()[0]}, linalgOp.getIndexingMaps());

  return success();
}

void GeneralizeToContractOps::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<GeneralizationToContractOps>(patterns.getContext());

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

} // namespace furiosa
} // namespace mlir
