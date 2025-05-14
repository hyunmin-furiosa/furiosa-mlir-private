#include "furiosa-mlir/Conversion/LinalgToFuriosa/LinalgToFuriosa.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class Pass;

namespace furiosa {
#define GEN_PASS_DEF_CONVERTLINALGTOFURIOSAPASS
#include "furiosa-mlir/Conversion/Passes.h.inc"

using namespace mlir;
namespace {

struct ForallOpLowering : public OpRewritePattern<scf::ForallOp> {
public:
  ForallOpLowering(MLIRContext *context)
      : OpRewritePattern<scf::ForallOp>(context) {}

  LogicalResult matchAndRewrite(scf::ForallOp op,
                                PatternRewriter &rewriter) const final;
};

struct ConvertLinalgToFuriosa
    : public impl::ConvertLinalgToFuriosaPassBase<ConvertLinalgToFuriosa> {
  using Base::Base;

public:
  void runOnOperation() final;
};

} // namespace

LogicalResult
ForallOpLowering::matchAndRewrite(scf::ForallOp op,
                                  PatternRewriter &rewriter) const {
  auto mapping = op.getMapping();
  if (!mapping || (mapping->size() != 1) ||
      !llvm::isa<furiosa::MappingAttr>((*mapping)[0])) {
    return rewriter.notifyMatchFailure(op,
                                       "op does not have a mapping attribute");
  }

  WalkResult status = op.walk([](Operation *op) {
    WalkResult status =
        llvm::TypeSwitch<Operation *, WalkResult>(op)
            .Case<tensor::ExtractSliceOp>([&](auto op) {
              op->dump();
              return WalkResult::advance();
            })
            .Case<tensor::ParallelInsertSliceOp>([&](auto op) {
              op->dump();
              return WalkResult::advance();
            })
            .Case<linalg::ContractOp>([&](auto op) {
              op->dump();
              return WalkResult::advance();
            })
            .Case<scf::InParallelOp>([&](auto op) {
              op->dump();
              return WalkResult::advance();
            })
            .Case<scf::ForallOp, arith::MulFOp, arith::AddFOp, linalg::YieldOp>(
                [&](auto op) { return WalkResult::skip(); })
            .Default([&](Operation *) {
              return op->emitOpError("unable to convert this op");
            });

    return status;
  });

  if (status.wasInterrupted()) {
    return failure();
  }

  return success();
}

void ConvertLinalgToFuriosa::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<furiosa::FuriosaDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<ForallOpLowering>(patterns.getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

} // namespace furiosa
} // namespace mlir
