#include "furiosa-mlir/Conversion/FuncToFuriosaHost/FuncToFuriosaHost.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaTypes.h"
#include "furiosa-mlir/Dialect/Host/IR/HostDialect.h"
#include "furiosa-mlir/Dialect/Host/IR/HostOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class Pass;

namespace furiosa {
#define GEN_PASS_DEF_CONVERTFUNCTOFURIOSAHOSTPASS
#include "furiosa-mlir/Conversion/Passes.h.inc"

using namespace mlir;
namespace {

struct CallOpLowering : public OpRewritePattern<func::CallOp> {
public:
  CallOpLowering(MLIRContext *context)
      : OpRewritePattern<func::CallOp>(context) {}

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter &rewriter) const final;
};

struct ConvertFuncToFuriosaHost
    : public impl::ConvertFuncToFuriosaHostPassBase<ConvertFuncToFuriosaHost> {
  using Base::Base;

public:
  void runOnOperation() final;
};

} // namespace

LogicalResult CallOpLowering::matchAndRewrite(func::CallOp op,
                                              PatternRewriter &rewriter) const {
  auto callee = op.getCalleeAttr();
  auto function_op = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(op->getParentOfType<ModuleOp>(), callee));
  if (!function_op || function_op.empty()) {
    llvm::report_fatal_error(llvm::Twine("function not found"));
    return failure();
  }
  if (!function_op->hasAttr("target")) {
    llvm::report_fatal_error(
        llvm::Twine("this function does not have target attribute"));
  }

  auto func_alloc_op =
      rewriter.create<furiosa::host::FuncAllocOp>(op.getLoc(), callee);
  rewriter.moveOpBefore(func_alloc_op, op);

  auto pe_binary_dram_address_attr =
      op->getAttrOfType<IntegerAttr>("dram_address");
  auto pe_binary_spm_address_attr =
      op->getAttrOfType<IntegerAttr>("spm_address");
  auto pe_program_load_inst_op =
      rewriter.create<furiosa::host::PeProgramLoadInstOp>(
          op.getLoc(), pe_binary_dram_address_attr, pe_binary_spm_address_attr,
          func_alloc_op);
  rewriter.moveOpBefore(pe_program_load_inst_op, op);

  auto pe_program_launch_op = rewriter.create<furiosa::host::PeProgramLaunchOp>(
      op.getLoc(), pe_binary_spm_address_attr);
  rewriter.moveOpBefore(pe_program_launch_op, op);

  auto pe_binary_write_op = rewriter.create<furiosa::host::HalProgramWriteAtOp>(
      op.getLoc(), pe_binary_dram_address_attr, func_alloc_op);
  rewriter.moveOpBefore(pe_binary_write_op, op);

  SmallVector<Value> operand_write_ops;
  SmallVector<Value> result_read_ops;
  for (auto operand : op.getOperands()) {
    auto defining_op = operand.getDefiningOp();
    auto tensor_type = llvm::cast<RankedTensorType>(operand.getType());
    auto dram_address_attr =
        defining_op->getAttrOfType<IntegerAttr>("dram_address");
    auto size = tensor_type.getNumElements() *
                tensor_type.getElementTypeBitWidth() / CHAR_BIT;
    auto size_attr = rewriter.getI64IntegerAttr(size);
    auto data_attr = rewriter.getI64ArrayAttr({0});
    auto alloc_op = rewriter.create<furiosa::host::AllocOp>(
        op.getLoc(), size_attr, data_attr);
    rewriter.moveOpBefore(alloc_op, op);
    if (defining_op->hasAttr("operand")) {
      auto write_op = rewriter.create<furiosa::host::HalProgramWriteAtOp>(
          op.getLoc(), dram_address_attr, alloc_op);
      operand_write_ops.push_back(write_op);
      rewriter.moveOpBefore(write_op, op);
    } else if (defining_op->hasAttr("result")) {
      auto read_op = rewriter.create<furiosa::host::HalProgramReadAtOp>(
          op.getLoc(), dram_address_attr, alloc_op);
      result_read_ops.push_back(read_op);
      rewriter.moveOpBefore(read_op, op);
    } else {
      llvm::report_fatal_error(
          llvm::Twine("arguments to kernel function need to have either "
                      "operand or result attribute"));
    }
  }

  auto pe_program_seq_op = rewriter.create<furiosa::host::PeProgramSeqOp>(
      op.getLoc(), ValueRange({pe_program_load_inst_op, pe_program_launch_op}));
  rewriter.moveOpBefore(pe_program_seq_op, op);

  auto hal_program_execute_op =
      rewriter.create<furiosa::host::HalProgramExecuteOp>(op.getLoc(),
                                                          pe_program_seq_op);
  rewriter.moveOpBefore(hal_program_execute_op, op);

  SmallVector<Value> hal_programs;
  hal_programs.push_back(pe_binary_write_op);
  hal_programs.insert(hal_programs.end(), operand_write_ops.begin(),
                      operand_write_ops.end());
  hal_programs.push_back(hal_program_execute_op);
  hal_programs.insert(hal_programs.end(), result_read_ops.begin(),
                      result_read_ops.end());
  auto hal_program_seq_op = rewriter.create<furiosa::host::HalProgramSeqOp>(
      op.getLoc(), ValueRange(hal_programs));
  rewriter.moveOpBefore(hal_program_seq_op, op);

  auto target_attr = op->getAttrOfType<furiosa::TargetAttr>("target");
  auto device_new_op =
      rewriter.create<furiosa::host::DeviceNewOp>(op.getLoc(), target_attr);
  rewriter.moveOpBefore(device_new_op, op);

  auto device_execute_op = rewriter.create<furiosa::host::DeviceExecuteOp>(
      op.getLoc(), device_new_op, hal_program_seq_op);
  rewriter.moveOpBefore(device_execute_op, op);

  for (auto operand : op.getOperands()) {
    rewriter.eraseOp(operand.getDefiningOp());
  }
  rewriter.eraseOp(op);

  return success();
}

void ConvertFuncToFuriosaHost::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect, furiosa::host::HostDialect>();
  target.addIllegalOp<func::CallOp>();

  RewritePatternSet patterns(&getContext());
  patterns.add<CallOpLowering>(patterns.getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

} // namespace furiosa
} // namespace mlir
