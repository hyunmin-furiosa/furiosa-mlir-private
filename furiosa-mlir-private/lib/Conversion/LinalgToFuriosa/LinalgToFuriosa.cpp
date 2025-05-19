#include "furiosa-mlir/Conversion/LinalgToFuriosa/LinalgToFuriosa.h"
#include "furiosa-mlir/Conversion/LinalgToFuriosa/Utils.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

  // Mark the furiosa memory type of the tensors
  LogicalResult markExtractSliceOp(tensor::ExtractSliceOp op,
                                   PatternRewriter &rewriter) const;
  LogicalResult markContractOp(linalg::ContractOp op,
                               PatternRewriter &rewriter) const;

  // Convert operators to furiosa dialect
  LogicalResult replaceExtractSliceOp(tensor::ExtractSliceOp op,
                                      ValueMapper value_mapper,
                                      PatternRewriter &rewriter) const;
  FailureOr<Operation *> replaceContractOp(linalg::ContractOp op,
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
ForallOpLowering::markExtractSliceOp(tensor::ExtractSliceOp op,
                                     PatternRewriter &rewriter) const {
  auto sram_attr = furiosa::MemoryTypeAttr::get(rewriter.getContext(),
                                                furiosa::MemoryType::sram);
  auto type = llvm::cast<RankedTensorType>(op.getType());
  rewriter.modifyOpInPlace(
      op, [&]() { op.getResult().setType(type.cloneWithEncoding(sram_attr)); });

  return success();
}

LogicalResult
ForallOpLowering::markContractOp(linalg::ContractOp op,
                                 PatternRewriter &rewriter) const {
  assert(op.getInputs().size() == 2);
  assert(op.getOutputs().size() == 1);

  auto sram_attr = furiosa::MemoryTypeAttr::get(rewriter.getContext(),
                                                furiosa::MemoryType::sram);
  auto trf_attr = furiosa::MemoryTypeAttr::get(rewriter.getContext(),
                                               furiosa::MemoryType::trf);

  auto in1_type = llvm::cast<RankedTensorType>(op.getInputs()[1].getType());
  auto in1_op = llvm::dyn_cast_or_null<tensor::ExtractSliceOp>(
      op.getInputs()[1].getDefiningOp());
  rewriter.modifyOpInPlace(in1_op, [&]() {
    in1_op.getResult().setType(in1_type.cloneWithEncoding(trf_attr));
  });

  auto type = llvm::cast<RankedTensorType>(op.getOutputs()[0].getType());
  rewriter.modifyOpInPlace(op, [&]() {
    op.getResult(0).setType(type.cloneWithEncoding(sram_attr));
  });

  return success();
}

LogicalResult
ForallOpLowering::replaceExtractSliceOp(tensor::ExtractSliceOp op,
                                        ValueMapper value_mapper,
                                        PatternRewriter &rewriter) const {
  auto dram_attr = furiosa::MemoryTypeAttr::get(rewriter.getContext(),
                                                furiosa::MemoryType::dram);
  auto sram_attr = furiosa::MemoryTypeAttr::get(rewriter.getContext(),
                                                furiosa::MemoryType::sram);
  auto trf_attr = furiosa::MemoryTypeAttr::get(rewriter.getContext(),
                                               furiosa::MemoryType::trf);

  auto source_type = op.getSourceType();
  auto source_memory_type = furiosa::MemoryType::dram;
  if (auto encoding = source_type.getEncoding()) {
    source_memory_type =
        llvm::dyn_cast_or_null<furiosa::MemoryTypeAttr>(encoding).getValue();
  }

  auto result_type = op.getResultType();
  auto result_memory_type = furiosa::MemoryType::dram;
  if (auto encoding = result_type.getEncoding()) {
    result_memory_type =
        llvm::dyn_cast_or_null<furiosa::MemoryTypeAttr>(encoding).getValue();
  }

  if (source_memory_type == furiosa::MemoryType::dram) {
    if (result_memory_type == furiosa::MemoryType::dram) {
      return failure();
    } else if (result_memory_type == furiosa::MemoryType::sram) {
      assert(op.hasUnitStride());
      auto [source_indexer, result_indexer] = getIndexers(
          source_type, result_type, op.getMixedOffsets(), value_mapper);
      auto source_limits = rewriter.getI64ArrayAttr(source_indexer.first);
      auto source_strides = rewriter.getI64ArrayAttr(source_indexer.second);
      auto destination_limits = rewriter.getI64ArrayAttr(result_indexer.first);
      auto destination_strides =
          rewriter.getI64ArrayAttr(result_indexer.second);
      auto dmaOp = rewriter.create<furiosa::DmaOp>(
          op.getLoc(), result_type.cloneWithEncoding(sram_attr), op.getSource(),
          Value(), source_limits, source_strides, destination_limits,
          destination_strides);
      rewriter.moveOpBefore(dmaOp, op);
      op.replaceAllUsesWith(dmaOp.getResult());
      rewriter.eraseOp(op);
      return success();
    } else if (result_memory_type == furiosa::MemoryType::trf) {
      assert(op.hasUnitStride());
      auto [source_indexer, result_indexer] = getIndexers(
          source_type, result_type, op.getMixedOffsets(), value_mapper);
      auto source_limits = rewriter.getI64ArrayAttr(source_indexer.first);
      auto source_strides = rewriter.getI64ArrayAttr(source_indexer.second);
      auto destination_limits = rewriter.getI64ArrayAttr(result_indexer.first);
      auto destination_strides =
          rewriter.getI64ArrayAttr(result_indexer.second);
      auto dmaOp = rewriter.create<furiosa::DmaOp>(
          op.getLoc(), result_type.cloneWithEncoding(trf_attr), op.getSource(),
          Value(), source_limits, source_strides, destination_limits,
          destination_strides);
      rewriter.moveOpBefore(dmaOp, op);
      op.replaceAllUsesWith(dmaOp.getResult());
      rewriter.eraseOp(op);
      return success();
    } else if (result_memory_type == furiosa::MemoryType::vrf) {
      return failure();
    }
  } else if (source_memory_type == furiosa::MemoryType::sram) {
    if (result_memory_type == furiosa::MemoryType::dram) {
      return failure();
    } else if (result_memory_type == furiosa::MemoryType::sram) {
      return failure();
    } else if (result_memory_type == furiosa::MemoryType::trf) {
      return failure();
    } else if (result_memory_type == furiosa::MemoryType::vrf) {
      return failure();
    }
  } else if (source_memory_type == furiosa::MemoryType::trf) {
    return failure();
  } else if (source_memory_type == furiosa::MemoryType::vrf) {
    return failure();
  }

  return failure();
}

FailureOr<Operation *>
ForallOpLowering::replaceContractOp(linalg::ContractOp op,
                                    PatternRewriter &rewriter) const {
  return failure();
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
  ValueMapper value_mapper;
  auto rank = op.getRank();
  auto induction_vars = op.getInductionVars();
  auto lower_bounds = op.getStaticLowerBound();
  auto upper_bounds = op.getStaticUpperBound();
  auto steps = op.getStaticStep();
  for (auto dim = 0; dim < rank; dim++) {
    auto induction_var = induction_vars[dim];
    auto lower_bound = lower_bounds[dim];
    auto upper_bound = upper_bounds[dim];
    auto step = steps[dim];
    if (!value_mapper.count(induction_var)) {
      value_mapper.insert_or_assign(
          induction_var, std::make_tuple(lower_bound, upper_bound, step));
    }
  }

  // Mark the furiosa memory type of the tensors
  WalkResult status = op.walk([&](tensor::ExtractSliceOp op) {
    if (failed(markExtractSliceOp(op, rewriter))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (status.wasInterrupted()) {
    return failure();
  }

  status = op.walk([&](linalg::ContractOp op) {
    if (failed(markContractOp(op, rewriter))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (status.wasInterrupted()) {
    return failure();
  }

  // Convert operators to furiosa dialect
  status = op.walk([&](tensor::ExtractSliceOp op) {
    if (failed(replaceExtractSliceOp(op, value_mapper, rewriter))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (status.wasInterrupted()) {
    return failure();
  }

  // status = op.walk([&](linalg::ContractOp op) {
  //   auto new_op = replaceContractOp(op, rewriter);
  //   if (failed(new_op)) {
  //     return WalkResult::interrupt();
  //   } else {
  //     rewriter.moveOpBefore(*new_op, op);
  //   }
  //   return WalkResult::advance();
  // });
  // if (status.wasInterrupted()) {
  //   return failure();
  // }
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
                         tensor::TensorDialect, furiosa::FuriosaDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<ForallOpLowering>(patterns.getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

} // namespace furiosa
} // namespace mlir
