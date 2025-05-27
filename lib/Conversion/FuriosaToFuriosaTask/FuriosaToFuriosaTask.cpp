#include "furiosa-mlir/Conversion/FuriosaToFuriosaTask/FuriosaToFuriosaTask.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskDialect.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

  LogicalResult matchAndRewrite(furiosa::DmaOp dma_op,
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

LogicalResult DmaOpLowering::matchAndRewrite(furiosa::DmaOp dma_op,
                                             PatternRewriter &rewriter) const {
  dma_op.dump();

  // If source or destination is from function argument, use the Value directly.
  // Otherwise, we need to extract the address from the AllocOp.
  auto source = Value();
  auto destination = Value();
  auto source_base = IntegerAttr();
  auto destination_base = IntegerAttr();

  if (llvm::isa<BlockArgument>(dma_op.getSource())) {
    source = dma_op.getSource();
  } else {
    auto source_op = dma_op.getSource().getDefiningOp();
    furiosa::AllocOp alloc_op;
    if (llvm::isa<furiosa::AllocOp>(source_op)) {
      alloc_op = llvm::dyn_cast_or_null<furiosa::AllocOp>(source_op);
    } else if (llvm::isa<linalg::ContractOp>(source_op)) {
      auto contract_op = llvm::dyn_cast_or_null<linalg::ContractOp>(source_op);
      assert(contract_op.getOutputs().size() == 1 &&
             "contract op should have exactly one output");
      auto contract_output_op = contract_op.getOutputs()[0].getDefiningOp();
      if (llvm::isa<furiosa::AllocOp>(contract_output_op)) {
        alloc_op = llvm::dyn_cast_or_null<furiosa::AllocOp>(contract_output_op);
      }
    }
    if (alloc_op) {
      source_base = alloc_op->getAttrOfType<IntegerAttr>("address");
    } else {
      return rewriter.notifyMatchFailure(
          dma_op, "source must be a BlockArgument or AllocOp or ContractOp");
    }
  }

  if (llvm::isa<BlockArgument>(dma_op.getDestination())) {
    destination = dma_op.getDestination();
  } else {
    auto destination_op = dma_op.getDestination().getDefiningOp();
    furiosa::AllocOp alloc_op;
    if (llvm::isa<furiosa::AllocOp>(destination_op)) {
      alloc_op = llvm::dyn_cast_or_null<furiosa::AllocOp>(destination_op);
    } else if (llvm::isa<linalg::ContractOp>(destination_op)) {
      auto contract_op =
          llvm::dyn_cast_or_null<linalg::ContractOp>(destination_op);
      assert(contract_op.getOutputs().size() == 1 &&
             "contract op should have exactly one output");
      auto contract_output_op = contract_op.getOutputs()[0].getDefiningOp();
      if (llvm::isa<furiosa::AllocOp>(contract_output_op)) {
        alloc_op = llvm::dyn_cast_or_null<furiosa::AllocOp>(contract_output_op);
      }
    }
    if (alloc_op) {
      destination_base = alloc_op->getAttrOfType<IntegerAttr>("address");
    } else {
      return rewriter.notifyMatchFailure(
          dma_op,
          "destination must be a BlockArgument or AllocOp or ContractOp");
    }
  }

  rewriter.setInsertionPoint(dma_op);

  std::uint64_t opcode = 0;
  std::uint64_t indirect = 0;
  auto source_limits = dma_op.getSourceLimits();
  auto source_strides = dma_op.getSourceStrides();
  auto destination_limits = dma_op.getDestinationLimits();
  auto destination_strides = dma_op.getDestinationStrides();
  auto descriptor_op = rewriter.create<furiosa::task::DmaDescriptorOp>(
      dma_op.getLoc(), source, destination, opcode, indirect, source_base,
      destination_base, source_limits, source_strides, destination_limits,
      destination_strides);

  std::uint64_t dma_tag_id = 0;
  bool profile = 0;
  std::uint64_t profile_id = 0;
  auto dynamic_dmaw_op = rewriter.create<furiosa::task::DynamicDmawOp>(
      dma_op.getLoc(), descriptor_op, ValueRange(), dma_tag_id, profile,
      profile_id);

  bool type = true;            // 0 for exec, 1 for DMA
  bool target_context = false; // ignored for DMA
  rewriter.create<furiosa::task::tuc::WaitOp>(dma_op.getLoc(),
                                              ValueRange({dynamic_dmaw_op}),
                                              dma_tag_id, type, target_context);

  rewriter.eraseOp(dma_op);

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
