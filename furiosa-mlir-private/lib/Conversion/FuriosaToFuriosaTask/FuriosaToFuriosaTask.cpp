#include "furiosa-mlir/Conversion/FuriosaToFuriosaTask/FuriosaToFuriosaTask.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskDialect.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

struct ConvertFuriosaToFuriosaTask
    : public impl::ConvertFuriosaToFuriosaTaskPassBase<
          ConvertFuriosaToFuriosaTask> {
  using Base::Base;

public:
  void runOnOperation() final;
};

} // namespace

void ConvertFuriosaToFuriosaTask::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<furiosa::task::TaskDialect>();

  RewritePatternSet patterns(&getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

} // namespace furiosa
} // namespace mlir
