#include "furiosa-mlir/Conversion/LinalgToFuriosa/LinalgToFuriosa.h"
#include "furiosa-mlir/Conversion/LinalgToFuriosa/Utils.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

  // Convert operators to furiosa dialect
  LogicalResult replaceExtractSliceOp(tensor::ExtractSliceOp op,
                                      ValueMapper value_mapper,
                                      PatternRewriter &rewriter) const;
  LogicalResult replaceParallelInsertSliceOp(tensor::ParallelInsertSliceOp op,
                                             ValueMapper value_mapper,
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
                                        ValueMapper value_mapper,
                                        PatternRewriter &rewriter) const {
  assert(op.hasUnitStride());

  auto sram_attr = furiosa::TensorAttr::get(
      rewriter.getContext(), furiosa::MemoryType::sram, Attribute());
  auto [source_indexer, result_indexer] =
      getIndexers(op.getSourceType(), op.getResultType(), op.getMixedOffsets(),
                  value_mapper);
  auto source_limits = rewriter.getI64ArrayAttr(source_indexer.first);
  auto source_strides = rewriter.getI64ArrayAttr(source_indexer.second);
  auto destination_limits = rewriter.getI64ArrayAttr(result_indexer.first);
  auto destination_strides = rewriter.getI64ArrayAttr(result_indexer.second);
  auto alloc_op = rewriter.replaceOpWithNewOp<furiosa::AllocOp>(
      op, op.getResultType().cloneWithEncoding(sram_attr), IntegerAttr());
  rewriter.create<furiosa::DmaOp>(
      op.getLoc(), op.getSource(), alloc_op.getResult(), source_limits,
      source_strides, destination_limits, destination_strides);
  return success();
}

LogicalResult ForallOpLowering::replaceParallelInsertSliceOp(
    tensor::ParallelInsertSliceOp op, ValueMapper value_mapper,
    PatternRewriter &rewriter) const {
  assert(op.hasUnitStride());

  auto [destination_indexer, source_indexer] = getIndexers(
      op.getDestType(), op.getSourceType(), op.getMixedOffsets(), value_mapper);
  auto source_limits = rewriter.getI64ArrayAttr(source_indexer.first);
  auto source_strides = rewriter.getI64ArrayAttr(source_indexer.second);
  auto destination_limits = rewriter.getI64ArrayAttr(destination_indexer.first);
  auto destination_strides =
      rewriter.getI64ArrayAttr(destination_indexer.second);
  rewriter.replaceOpWithNewOp<furiosa::DmaOp>(
      op, op.getSource(), op.getDest(), source_limits, source_strides,
      destination_limits, destination_strides);
  return success();
}

LogicalResult
ForallOpLowering::matchAndRewrite(scf::ForallOp op,
                                  PatternRewriter &rewriter) const {
  auto mapping = op.getMapping();
  if (!mapping || (mapping->size() < 1) ||
      !llvm::isa<furiosa::MappingAttr>((*mapping)[0])) {
    return rewriter.notifyMatchFailure(op,
                                       "op does not have a mapping attribute");
  }

  // Store the induction variables and its ranges
  assert(op.isNormalized());
  ValueMapper value_mapper;
  auto rank = op.getRank();
  auto induction_vars = op.getInductionVars();
  auto upper_bounds = op.getStaticUpperBound();
  for (auto dim = 0; dim < rank; dim++) {
    auto induction_var = induction_vars[dim];
    auto upper_bound = upper_bounds[dim];
    if (!value_mapper.count(induction_var)) {
      value_mapper.insert_or_assign(induction_var, upper_bound);
    }
  }

  // Mark all operands and results of contract op as SRAM
  auto sram_attr = furiosa::TensorAttr::get(
      rewriter.getContext(), furiosa::MemoryType::sram, Attribute());
  WalkResult status = op.walk([&](linalg::ContractOp contract_op) {
    rewriter.modifyOpInPlace(contract_op, [&]() {
      for (auto operand : contract_op.getOperands()) {
        auto type = llvm::cast<RankedTensorType>(operand.getType());
        operand.setType(type.cloneWithEncoding(sram_attr));
      }
      for (auto result : contract_op.getResults()) {
        auto type = llvm::cast<RankedTensorType>(result.getType());
        result.setType(type.cloneWithEncoding(sram_attr));
      }
    });
    return WalkResult::advance();
  });
  if (status.wasInterrupted()) {
    return failure();
  }

  // Convert operators to furiosa dialect
  status = op.walk([&](tensor::ExtractSliceOp extract_op) {
    rewriter.setInsertionPoint(extract_op);
    if (failed(replaceExtractSliceOp(extract_op, value_mapper, rewriter))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (status.wasInterrupted()) {
    return failure();
  }

  auto terminator = op.getTerminator();
  status = terminator.walk([&](tensor::ParallelInsertSliceOp insert_op) {
    rewriter.setInsertionPoint(terminator);
    if (failed(
            replaceParallelInsertSliceOp(insert_op, value_mapper, rewriter))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (status.wasInterrupted()) {
    return failure();
  }

  status = op.walk([&](affine::AffineApplyOp apply_op) {
    rewriter.eraseOp(apply_op);
    return WalkResult::advance();
  });

  return success();
}

LogicalResult
EmptyOpLowering::matchAndRewrite(tensor::EmptyOp op,
                                 PatternRewriter &rewriter) const {
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<furiosa::AllocOp>(op, op.getType(),
                                                IntegerAttr());

  return success();
}

void ConvertLinalgToFuriosa::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<affine::AffineDialect, arith::ArithDialect,
                         func::FuncDialect, linalg::LinalgDialect,
                         tensor::TensorDialect, furiosa::FuriosaDialect>();
  target.addIllegalOp<tensor::EmptyOp>();

  RewritePatternSet patterns(&getContext());
  patterns.add<EmptyOpLowering>(patterns.getContext());
  patterns.add<ForallOpLowering>(patterns.getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

} // namespace furiosa
} // namespace mlir
