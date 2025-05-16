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

  LogicalResult replaceExtractSliceOp(tensor::ExtractSliceOp op,
                                      PatternRewriter &rewriter) const;
  LogicalResult replaceContractOp(linalg::ContractOp op,
                                  PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(scf::ForallOp op,
                                PatternRewriter &rewriter) const final;
};

struct EmptyOpLowering : public OpRewritePattern<tensor::EmptyOp> {
public:
  EmptyOpLowering(MLIRContext *context)
      : OpRewritePattern<tensor::EmptyOp>(context) {}

  LogicalResult matchAndRewrite(tensor::EmptyOp op,
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
ForallOpLowering::replaceExtractSliceOp(tensor::ExtractSliceOp op,
                                        PatternRewriter &rewriter) const {
  auto sram_attr = furiosa::MemoryTypeAttr::get(rewriter.getContext(),
                                                furiosa::MemoryType::sram);
  auto type = llvm::cast<RankedTensorType>(op.getType());
  auto dma_op = rewriter.create<furiosa::DmaOp>(
      op.getLoc(), type.cloneWithEncoding(sram_attr), op.getSource());
  rewriter.moveOpBefore(dma_op, op);
  op.replaceAllUsesWith(dma_op.getResult());
  rewriter.eraseOp(op);

  return success();
}

LogicalResult
ForallOpLowering::replaceContractOp(linalg::ContractOp op,
                                    PatternRewriter &rewriter) const {
  assert(op.getInputs().size() == 2);
  assert(op.getOutputs().size() == 1);

  auto sram_attr = furiosa::MemoryTypeAttr::get(rewriter.getContext(),
                                                furiosa::MemoryType::sram);
  auto trf_attr = furiosa::MemoryTypeAttr::get(rewriter.getContext(),
                                               furiosa::MemoryType::trf);

  auto in1_type = llvm::cast<RankedTensorType>(op.getInputs()[1].getType());
  auto in1_op =
      llvm::dyn_cast_or_null<furiosa::DmaOp>(op.getInputs()[1].getDefiningOp());
  auto trf_dma_op = rewriter.create<furiosa::DmaOp>(
      in1_op.getLoc(), in1_type.cloneWithEncoding(trf_attr),
      in1_op.getResult());
  rewriter.moveOpAfter(trf_dma_op, in1_op);
  in1_op.getResult().replaceAllUsesExcept(trf_dma_op.getResult(), trf_dma_op);

  auto type = llvm::cast<RankedTensorType>(op.getOutputs()[0].getType());
  rewriter.modifyOpInPlace(op, [&]() {
    op.getResult(0).setType(type.cloneWithEncoding(sram_attr));
  });

  assert(op->hasOneUse());
  for (auto user : op->getUsers()) {
    auto result_op =
        llvm::dyn_cast_or_null<tensor::ParallelInsertSliceOp>(user);
    auto dma_op = rewriter.create<furiosa::DmaOp>(
        result_op.getLoc(), mlir::Type(), op.getResult(0), result_op.getDest());
    rewriter.moveOpAfter(dma_op, op);
    rewriter.eraseOp(result_op);
  }

  return success();
}

LogicalResult
ForallOpLowering::matchAndRewrite(scf::ForallOp op,
                                  PatternRewriter &rewriter) const {
  auto mapping = op.getMapping();
  if (!mapping || (mapping->size() != 1) ||
      !llvm::isa<furiosa::MappingAttr>((*mapping)[0])) {
    return rewriter.notifyMatchFailure(op,
                                       "op does not have a mapping attribute");
  }

  WalkResult status = op.walk([&](tensor::ExtractSliceOp op) {
    if (failed(replaceExtractSliceOp(op, rewriter))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (status.wasInterrupted()) {
    return failure();
  }

  status = op.walk([&](linalg::ContractOp op) {
    if (failed(replaceContractOp(op, rewriter))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (status.wasInterrupted()) {
    return failure();
  }

  return success();
}

LogicalResult
EmptyOpLowering::matchAndRewrite(tensor::EmptyOp op,
                                 PatternRewriter &rewriter) const {
  auto type = llvm::cast<RankedTensorType>(op.getType());
  auto size = type.getNumElements();
  if (!type.getEncoding()) {
    auto dram_attr = furiosa::MemoryTypeAttr::get(rewriter.getContext(),
                                                  furiosa::MemoryType::dram);
    auto allocOp = rewriter.create<furiosa::AllocOp>(
        op.getLoc(), type.cloneWithEncoding(dram_attr),
        furiosa::MemoryType::dram, size);
    op.replaceAllUsesWith(allocOp.getOperation());
  }

  return success();
}

void ConvertLinalgToFuriosa::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                         furiosa::FuriosaDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<ForallOpLowering>(patterns.getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

} // namespace furiosa
} // namespace mlir
