#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

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
  os << "*TUC_COMMAND_QUEUE_TAIL = tail";
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

static LogicalResult printKernelFunction(func::FuncOp functionOp) {
  int fd;
  llvm::Twine filepath_c = functionOp.getSymName() + ".c";
  llvm::Twine filepath_o = functionOp.getSymName() + ".o";
  if (std::error_code error = llvm::sys::fs::openFileForWrite(filepath_c, fd)) {
    llvm::report_fatal_error(llvm::Twine("Failed to open file: ") +
                             error.message());
    return failure();
  }
  {
    // C file needs to be closed to be compiled properly
    llvm::raw_fd_ostream os(fd, /*shouldClose=*/true);

    // Define constants
    os << "#include <stdint.h>\n";
    os << "\n";
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
    os << "uint64_t (*trampoline)(uint64_t, uint64_t, uint64_t);\n";
    os << "\n";

    // Define function
    os << "void " << functionOp.getName() << "() {\n";
    Operation *operation = functionOp.getOperation();
    ArmCEmitter armCEmitter(os);
    if (failed(
            printFunctionBody(armCEmitter, operation, functionOp.getBlocks())))
      return failure();
    os << "}\n";

    if (os.has_error())
      llvm::report_fatal_error(llvm::Twine("Error emitting the IR to file '") +
                               filepath_c);
  }

  // Compile the C code
  std::string command = "aarch64-none-elf-gcc ";
  command += "-r ";
  command += "-fno-builtin ";
  command += "-fno-zero-initialized-in-bss ";
  static constexpr std::uint32_t MAX_STACK_USAGE = 1020 * 1024;
  command += "-Werror=stack-usage=" + std::to_string(MAX_STACK_USAGE) + " ";
  command += "-nostdlib ";
  command += "-fwrapv ";
  command += "-static ";
  command += "-Wl,-n ";
  command += "-xc ";
  command += "-Werror ";
  command += "-fno-omit-frame-pointer ";
  command += "-O3 ";
  command += "-std=c11 ";
  command += filepath_c.str() + " ";
  command += "-o " + filepath_o.str() + " ";
  system(command.c_str());

  // Read compiled binary
  // auto buffer =
  //     llvm::MemoryBuffer::getFile(filepath_o.str())->get()->getBuffer();

  return success();
}

static LogicalResult printOperation(ArmCEmitter &emitter,
                                    func::FuncOp functionOp) {
  if (functionOp.getSymName() == "kernel") {
    return printKernelFunction(functionOp);
  }
  return success();
}

static LogicalResult printOperation(ArmCEmitter &emitter,
                                    func::ReturnOp returnOp) {
  raw_indented_ostream &os = emitter.ostream();
  os << "\n";
  os << "trampoline(TRAMPOLINE_EXIT, 0, 0)";
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
  LogicalResult status =
      emitter.emitOperation(*op, /*trailingSemicolon=*/false);
  if (failed(status))
    return failure();

  return status;
}

} // namespace mlir::furiosa
