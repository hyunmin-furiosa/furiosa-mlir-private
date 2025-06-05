#include "furiosa-mlir/Dialect/Furiosa/Transforms/AllocateAddress.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"
#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.h"
#include "furiosa-mlir/Dialect/Furiosa/Transforms/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

namespace mlir {
class Pass;

namespace furiosa {
#define GEN_PASS_DEF_FURIOSAALLOCATEADDRESSPASS
#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.h.inc"

using namespace mlir;
namespace {

struct AddressAllocation : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp func_op,
                                PatternRewriter &rewriter) const final;
};

struct AllocateAddress
    : public impl::FuriosaAllocateAddressPassBase<AllocateAddress> {
  using Base::Base;

public:
  void runOnOperation() final;
};

} // namespace

LogicalResult
AddressAllocation::matchAndRewrite(func::FuncOp func_op,
                                   PatternRewriter &rewriter) const {
  if (func_op->hasAttr("address_allocated")) {
    // If the function already has address allocation, skip it.
    return rewriter.notifyMatchFailure(
        func_op, "function already has address allocation");
  }

  mlir::furiosa::MemoryAllocator allocator;

  func_op.walk([&](Operation *op) {
    LogicalResult status =
        llvm::TypeSwitch<Operation *, LogicalResult>(op)
            .Case<furiosa::AllocOp>([&](auto op) {
              auto type = op.getType();
              auto tensor_type = llvm::cast<RankedTensorType>(type);
              auto size = tensor_type.getNumElements() *
                          tensor_type.getElementTypeBitWidth() / CHAR_BIT;
              auto memory_type = furiosa::MemoryType::dram;
              if (auto encoding = tensor_type.getEncoding()) {
                memory_type =
                    llvm::dyn_cast_or_null<furiosa::MemoryTypeAttr>(encoding)
                        .getValue();
              }
              auto address = allocator.allocate(size, memory_type);
              auto address_attr = rewriter.getI64IntegerAttr(address);
              rewriter.modifyOpInPlace(
                  op, [&]() { op->setAttr("address", address_attr); });

              return success();
            })
            .Case<furiosa::DeallocOp>([&](auto op) {
              auto alloc_op = llvm::dyn_cast<furiosa::AllocOp>(
                  op.getBuffer().getDefiningOp());
              auto address =
                  alloc_op->template getAttrOfType<IntegerAttr>("address")
                      .getInt();
              auto type = alloc_op.getType();
              auto tensor_type = llvm::cast<RankedTensorType>(type);
              auto memory_type = furiosa::MemoryType::dram;
              if (auto encoding = tensor_type.getEncoding()) {
                memory_type =
                    llvm::dyn_cast_or_null<furiosa::MemoryTypeAttr>(encoding)
                        .getValue();
              }
              allocator.deallocate(address, memory_type);

              return success();
            })
            .Case<linalg::ContractOp>([&](auto op) {
              auto context_id = 0;
              auto context_id_attr = rewriter.getBoolAttr(context_id);
              rewriter.modifyOpInPlace(
                  op, [&]() { op->setAttr("context_id", context_id_attr); });

              return success();
            })
            .Default([&](Operation *op) { return success(); });

    if (failed(status)) {
      llvm::report_fatal_error(
          llvm::Twine("failed to match and rewrite operation"));
    }
  });

  // Mark the function as having address allocation.
  func_op->setAttr("address_allocated", rewriter.getUnitAttr());

  return success();
};

void AllocateAddress::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<AddressAllocation>(patterns.getContext());

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

} // namespace furiosa
} // namespace mlir
