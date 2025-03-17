#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"

using namespace mlir;

namespace mlir::furiosa {

/// Emitter that uses dialect specific emitters to emit Arm C code.
struct ArmCEmitter {
  explicit ArmCEmitter(raw_ostream &os);

  /// Emits operation 'op' with/without training semicolon or returns failure.
  ///
  /// For operations that should never be followed by a semicolon, like ForOp,
  /// the `trailingSemicolon` argument is ignored and a semicolon is not
  /// emitted.
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; };

private:
  /// Output stream to emit to.
  raw_indented_ostream os;
};

static LogicalResult printOperation(ArmCEmitter &emitter, furiosa::ExecutionOp executionOp) {
  raw_ostream &os = emitter.ostream();
  os << "furiosa::exec();\n";
  return success();
}

static LogicalResult printOperation(ArmCEmitter &emitter, furiosa::WaitOp waitOp) {
  raw_ostream &os = emitter.ostream();
  os << "furiosa::wait();\n";
  return success();
}
static LogicalResult printFunctionBody(ArmCEmitter &emitter,
                                       Operation *functionOp,
                                       Region::BlockListType &blocks) {
  raw_indented_ostream &os = emitter.ostream();
  os.indent();
  for (Block &block : blocks) {
    for (Operation &op : block.getOperations()) {
      if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
        return failure();
    }
  }
  os.unindent();

  return success();
}

static LogicalResult printOperation(ArmCEmitter &emitter,
                                    func::FuncOp functionOp) {
  raw_ostream &os = emitter.ostream();
  Operation *operation = functionOp.getOperation();
  os << functionOp.getName() << "() {\n";
  if (failed(printFunctionBody(emitter, operation, functionOp.getBlocks())))
    return failure();
  os << "}\n";

  return success();
}

static LogicalResult printOperation(ArmCEmitter &emitter,
                                    func::ReturnOp returnOp) {
  raw_ostream &os = emitter.ostream();
  os << "return\n";
  return success();
}

static LogicalResult printOperation(ArmCEmitter &emitter, ModuleOp moduleOp) {
  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
      return failure();
  }
  return success();
}

ArmCEmitter::ArmCEmitter(raw_ostream &os)
    : os(os) {}

LogicalResult ArmCEmitter::emitOperation(Operation &op, bool trailingSemicolon) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
          // Func ops.
          .Case<func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<furiosa::ExecutionOp, furiosa::WaitOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Default([&](Operation *) {
            return op.emitOpError("unable to find printer for op");
          });
  
  if (failed(status))
    return failure();

  return success();
}

LogicalResult translateFuriosaToBinary(
    Operation *op, llvm::raw_ostream &os) {
  ArmCEmitter emitter(os);
  return emitter.emitOperation(*op, /*trailingSemicolon=*/false);
}

} // namespace mlir::furiosa
