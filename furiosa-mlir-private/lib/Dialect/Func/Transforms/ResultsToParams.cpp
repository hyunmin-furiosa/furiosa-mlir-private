#include "furiosa-mlir/Dialect/Func/Transforms/ResultsToParams.h"
#include "furiosa-mlir/Dialect/Func/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
class Pass;

namespace furiosa {
#define GEN_PASS_DEF_FUNCRESULTSTOPARAMSPASS
#include "furiosa-mlir/Dialect/Func/Transforms/Passes.h.inc"

using namespace mlir;
namespace {

struct FuncOpTransformation : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const final;
};

struct ResultsToParams
    : public impl::FuncResultsToParamsPassBase<ResultsToParams> {
  using Base::Base;

public:
  void runOnOperation() final;
};

} // namespace

LogicalResult
FuncOpTransformation::matchAndRewrite(func::FuncOp func_op,
                                      PatternRewriter &rewriter) const {
  // If the function has no results, nothing to do.
  auto num_results = func_op.getNumResults();
  if (num_results == 0)
    return rewriter.notifyMatchFailure(func_op, "no results to convert");

  // Change function signature
  auto new_argument_types = SmallVector<Type>(
      func_op.getArgumentTypes().begin(), func_op.getArgumentTypes().end());
  new_argument_types.append(func_op.getResultTypes().begin(),
                            func_op.getResultTypes().end());
  rewriter.modifyOpInPlace(func_op, [&] {
    for (auto result_type : func_op.getResultTypes()) {
      auto result =
          func_op.insertArgument(func_op.getNumArguments(), result_type,
                                 DictionaryAttr(), func_op.getLoc());
      if (failed(result)) {
        llvm::report_fatal_error("failed to insert argument");
      }
      result = func_op.eraseResult(0);
      if (failed(result)) {
        llvm::report_fatal_error("failed to erase result");
      }
    }
  });

  // Change internal returns
  SmallVector<Operation *> ops_to_erase;
  auto new_arguments = func_op.getArguments().take_back(num_results);
  func_op.walk([&](func::ReturnOp return_op) {
    assert(return_op.getNumOperands() == num_results);
    for (auto operand : return_op.getOperands()) {
      ops_to_erase.push_back(operand.getDefiningOp());
    }
    rewriter.replaceAllUsesWith(return_op.getOperands(), new_arguments);
    rewriter.modifyOpInPlace(return_op, [&] {
      return_op.getOperation()->eraseOperands(0, num_results);
    });
  });
  assert(ops_to_erase.size() == num_results &&
         "number of ops to erase should match number of results");

  // Change function calls
  auto uses = *SymbolTable::getSymbolUses(func_op,
                                          func_op->getParentOfType<ModuleOp>());
  for (auto use : uses) {
    if (auto call_op = llvm::dyn_cast_or_null<func::CallOp>(use.getUser())) {
      rewriter.setInsertionPoint(call_op);
      SmallVector<Value> new_values;
      for (auto op : ops_to_erase) {
        auto alloc_op = rewriter.clone(*op);
        alloc_op->setAttr("result", rewriter.getUnitAttr());
        new_values.push_back(alloc_op->getResult(0));
      }
      for (auto operand : call_op.getOperands()) {
        if (auto defining_op = operand.getDefiningOp()) {
          defining_op->setAttr("argument", rewriter.getUnitAttr());
        }
      }
      auto new_operands = SmallVector<Value>(call_op.getOperands());
      new_operands.append(new_values.begin(), new_values.end());
      rewriter.create<func::CallOp>(call_op.getLoc(), call_op.getCallee(),
                                    TypeRange(), new_operands);
      rewriter.replaceAllUsesWith(call_op.getResults(), new_values);
      rewriter.eraseOp(call_op);
    }
  }

  for (auto op : ops_to_erase) {
    rewriter.eraseOp(op);
  }

  return success();
}

void ResultsToParams::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<FuncOpTransformation>(patterns.getContext());

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

} // namespace furiosa
} // namespace mlir
