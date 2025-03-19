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
#include "furiosa-mlir/Dialect/Furiosa/IR/Utils.h"

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

static LogicalResult printCommand(ArmCEmitter &emitter, std::uint32_t command) {
  raw_indented_ostream &os = emitter.ostream();
  os << "TUC_COMMAND_QUEUE_ENTRY[tail] = 0x";
  os.write_hex(command);
  os << ";\n";
  os << "tail = (tail + 1) % TUC_COMMAND_QUEUE_SIZE;\n";
  os << "TUC_COMMAND_QUEUE_TAIL = tail";
  return success();
}

static LogicalResult printOperation(ArmCEmitter &emitter,
                                    furiosa::ExecutionOp executionOp) {
  std::uint32_t command = *getCommand(*executionOp.getOperation());
  return printCommand(emitter, command);
  return success();
}

static LogicalResult printOperation(ArmCEmitter &emitter,
                                    furiosa::WaitOp waitOp) {
  std::uint32_t command = *getCommand(*waitOp.getOperation());
  return printCommand(emitter, command);
}

static LogicalResult printFunctionBody(ArmCEmitter &emitter,
                                       Operation *functionOp,
                                       Region::BlockListType &blocks) {
  raw_indented_ostream &os = emitter.ostream();
  os.indent();

  // Initialize kernel
  os << "uint32_t tail = *TUC_COMMAND_QUEUE_TAIL;\n";
  os << "\n";

  // Emit the body of the function.
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
  raw_indented_ostream &os = emitter.ostream();

  // Define constants
  os << "#define TUC_BASE UINT64_C(0x000C000000)\n";
  os << "#define TUC_COMMAND_QUEUE_HEAD ((volatile uint64_t *)(TUC_BASE + "
        "0x020))\n";
  os << "#define TUC_COMMAND_QUEUE_TAIL ((volatile uint64_t *)(TUC_BASE + "
        "0x028))\n";
  os << "#define TUC_COMMAND_QUEUE_ENTRY ((volatile uint32_t *)(TUC_BASE + "
        "0x100))\n";
  os << "#define TUC_GENERAL_REGISTERS ((volatile uint64_t *)(TUC_BASE + "
        "0x200))\n";
  os << "#define TUC_COMMAND_QUEUE_SIZE 64\n";
  os << "#define TUC_REGISTER_COUNT 64\n";
  os << "#define TRAMPOLINE_EXIT (0 << 8)\n";
  os << "\n";

  // Define function
  os << "void " << functionOp.getName() << "() {\n";
  Operation *operation = functionOp.getOperation();
  if (failed(printFunctionBody(emitter, operation, functionOp.getBlocks())))
    return failure();
  os << "}\n";

  return success();
}

static LogicalResult printOperation(ArmCEmitter &emitter,
                                    func::ReturnOp returnOp) {
  raw_indented_ostream &os = emitter.ostream();
  os << "\n";
  os << "trampoline(TRAMPOLINE_EXIT, 0, 0);";
  return success();
}

static LogicalResult printOperation(ArmCEmitter &emitter, ModuleOp moduleOp) {
  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
      return failure();
  }
  return success();
}

ArmCEmitter::ArmCEmitter(raw_ostream &os) : os(os) {}

LogicalResult ArmCEmitter::emitOperation(Operation &op,
                                         bool trailingSemicolon) {
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

  os << (trailingSemicolon ? ";\n" : "\n");

  return success();
}

LogicalResult translateFuriosaToBinary(Operation *op, llvm::raw_ostream &os) {
  ArmCEmitter emitter(os);
  return emitter.emitOperation(*op, /*trailingSemicolon=*/false);
}

} // namespace mlir::furiosa
