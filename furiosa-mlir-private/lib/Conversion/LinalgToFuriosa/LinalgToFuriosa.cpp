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

  // Mark the furiosa memory type of the tensors
  LogicalResult markExtractSliceOp(tensor::ExtractSliceOp op,
                                   PatternRewriter &rewriter) const;
  LogicalResult markContractOp(linalg::ContractOp op,
                               PatternRewriter &rewriter) const;

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
      auto alloc_op = rewriter.replaceOpWithNewOp<furiosa::AllocOp>(
          op, result_type.cloneWithEncoding(sram_attr), IntegerAttr());
      rewriter.create<furiosa::DmaOp>(
          op.getLoc(), op.getSource(), alloc_op.getResult(), source_limits,
          source_strides, destination_limits, destination_strides);
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
      auto sram_alloc_op = rewriter.create<furiosa::AllocOp>(
          op.getLoc(), result_type.cloneWithEncoding(sram_attr), IntegerAttr());
      rewriter.create<furiosa::DmaOp>(
          op.getLoc(), op.getSource(), sram_alloc_op.getResult(), source_limits,
          source_strides, destination_limits, destination_strides);
      auto trf_alloc_op = rewriter.replaceOpWithNewOp<furiosa::AllocOp>(
          op, result_type.cloneWithEncoding(trf_attr), IntegerAttr());
      rewriter.create<furiosa::LoadTrfOp>(
          op.getLoc(), sram_alloc_op.getResult(), trf_alloc_op.getResult());
      return success();
    } else if (result_memory_type == furiosa::MemoryType::vrf) {
      return failure();
    }
  } else if (source_memory_type == furiosa::MemoryType::sram) {
    return failure();
  } else if (source_memory_type == furiosa::MemoryType::trf) {
    return failure();
  } else if (source_memory_type == furiosa::MemoryType::vrf) {
    return failure();
  }

  return failure();
}

LogicalResult ForallOpLowering::replaceParallelInsertSliceOp(
    tensor::ParallelInsertSliceOp op, ValueMapper value_mapper,
    PatternRewriter &rewriter) const {
  auto source_type = op.getSourceType();
  auto source_memory_type = furiosa::MemoryType::dram;
  if (auto encoding = source_type.getEncoding()) {
    source_memory_type =
        llvm::dyn_cast_or_null<furiosa::MemoryTypeAttr>(encoding).getValue();
  }
  auto dest_type = op.getDestType();
  auto dest_memory_type = furiosa::MemoryType::dram;
  if (auto encoding = dest_type.getEncoding()) {
    dest_memory_type =
        llvm::dyn_cast_or_null<furiosa::MemoryTypeAttr>(encoding).getValue();
  }

  if (source_memory_type == furiosa::MemoryType::dram) {
    return failure();
  } else if (source_memory_type == furiosa::MemoryType::sram) {
    if (dest_memory_type == furiosa::MemoryType::dram) {
      assert(op.hasUnitStride());
      auto [destination_indexer, source_indexer] = getIndexers(
          dest_type, source_type, op.getMixedOffsets(), value_mapper);
      auto source_limits = rewriter.getI64ArrayAttr(source_indexer.first);
      auto source_strides = rewriter.getI64ArrayAttr(source_indexer.second);
      auto destination_limits =
          rewriter.getI64ArrayAttr(destination_indexer.first);
      auto destination_strides =
          rewriter.getI64ArrayAttr(destination_indexer.second);
      rewriter.replaceOpWithNewOp<furiosa::DmaOp>(
          op, op.getSource(), op.getDest(), source_limits, source_strides,
          destination_limits, destination_strides);
      return success();
    } else if (dest_memory_type == furiosa::MemoryType::sram) {
      return failure();
    } else if (dest_memory_type == furiosa::MemoryType::trf) {
      return failure();
    } else if (dest_memory_type == furiosa::MemoryType::vrf) {
      return failure();
    }
  } else if (source_memory_type == furiosa::MemoryType::trf) {
    return failure();
  } else if (source_memory_type == furiosa::MemoryType::vrf) {
    return failure();
  }

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

  // Mark the furiosa memory type of the tensors
  WalkResult status = op.walk([&](tensor::ExtractSliceOp extract_op) {
    if (failed(markExtractSliceOp(extract_op, rewriter))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (status.wasInterrupted()) {
    return failure();
  }

  status = op.walk([&](linalg::ContractOp contract_op) {
    if (failed(markContractOp(contract_op, rewriter))) {
      return WalkResult::interrupt();
    }
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
