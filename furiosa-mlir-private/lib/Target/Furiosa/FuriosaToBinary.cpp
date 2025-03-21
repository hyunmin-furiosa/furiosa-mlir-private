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
  os << "*TUC_COMMAND_QUEUE_TAIL = tail;\n";
  os << "while (*TUC_COMMAND_QUEUE_HEAD != tail) {}";
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
  llvm::Twine symName = functionOp.getSymName();
  llvm::Twine filepath_c = symName + ".c";
  llvm::Twine filepath_o = symName + ".o";
  llvm::Twine filepath_bin = symName + ".bin";
  if (std::error_code error = llvm::sys::fs::openFileForWrite(filepath_c, fd)) {
    llvm::report_fatal_error(llvm::Twine("Failed to open file: ") +
                             error.message());
    return failure();
  }
  {
    // C file needs to be closed to be compiled properly
    llvm::raw_fd_ostream fd_os(fd, /*shouldClose=*/true);
    ArmCEmitter armCEmitter(fd_os);
    raw_indented_ostream &os = armCEmitter.ostream();

    // Define constants
    os << R"""(#include <assert.h>
#include <stdalign.h>)""";
#include <stdint.h>
    os << "\n";

    os << R"""(
#define TRAMPOLINE_EXIT (0 << 8)
#define TRAMPOLINE_RECV_MESSAGE (1 << 8)
#define TRAMPOLINE_WAIT_FOR_TUC (2 << 8)

#define NUM_ITERATOR_INDEXERS 8

#define TUC_BASE UINT64_C(0x000C000000)

#define TUC_COMMAND_QUEUE_HEAD ((volatile uint64_t *)(TUC_BASE + 0x020))
#define TUC_COMMAND_QUEUE_TAIL ((volatile uint64_t *)(TUC_BASE + 0x028))
#define TUC_COMMAND_QUEUE_ENTRY ((volatile uint32_t *)(TUC_BASE + 0x100))
#define TUC_GENERAL_REGISTERS ((volatile uint64_t *)(TUC_BASE + 0x200))

#define TUC_COMMAND_QUEUE_SIZE 64
#define TUC_REGISTER_COUNT 64

#define TUC_COMMAND_WAIT 17
#define TUC_COMMAND_INTERRUPT 18
#define TUC_COMMAND_DMA 19
#define TUC_COMMAND_DMA_W 22

/* How often interrupts are enqueued. */
#define TUC_INTERRUPT_PERIOD 16

/* How many descriptors can be in one tensor dma request (= maximal number of fusioned PE) */
#define TDMA_DESC_PER_REQUEST 4

#define SHARED_FIELDS_ADDR 0xEB000

#define SHARED_AREA_SIZE (1 << 12)

#define NUM_MAX_CLUSTERS 8)""";
    os << "\n";

    // dma descriptor
    os << R"""(
/**
 * Tensor DMA Descriptor
 */
struct dma_desc_t
{
    /**
     * [1:0] : opcode (0: TensorLoop, 1: SourceIndirect, 2: DestinationIndirect)
     */
    alignas(256) uint64_t metadata;

    /**
     * [2:0] : dimension
     * [8:8] : indirect entry type
     *   (0: indirect address, 1: indirect list address)
     */
    uint64_t indirect_access;

    uint64_t source_base;
    uint64_t destination_base;

    uint16_t source_limits[NUM_ITERATOR_INDEXERS];
    int32_t source_strides[NUM_ITERATOR_INDEXERS];

    uint16_t destination_limits[NUM_ITERATOR_INDEXERS];
    int32_t destination_strides[NUM_ITERATOR_INDEXERS];

    int32_t indirect_indices[32];
};

typedef uint64_t tuc_dma_desc_bitmap_t;

typedef uint64_t tuc_register_bitmap_t;

typedef uint32_t tuc_timestamp_t;

struct __attribute((packed)) commit_info_t
{
    uint64_t address;
    uint32_t size;
};

/**
 * A set of TUC resources (registers and DMA descriptors), which are reserved as operands of
 * scheduled commands. They will be freed (i.e. commands get completed) after the queue head reaches
 * `interrupt_index`.
 *
 * You should enqueue `interrupt` command at `interrupt_index` to wait for blocks being released.
 */
struct tuc_block_t
{
    uint32_t interrupt_index;
    tuc_register_bitmap_t used_registers;
    tuc_dma_desc_bitmap_t used_descs;
    tuc_timestamp_t current_timestamp;

    struct commit_info_t commits[TUC_INTERRUPT_PERIOD];
    uint32_t num_commits;
};

/**
 * Shared data structure between PERT and task, and between tasks (libpe)
 */
struct shared_field_t
{
    // shared fields between tasks
    alignas(256) volatile uint8_t shared_area[SHARED_AREA_SIZE];
    struct dma_desc_t dma_desc_arena[TUC_COMMAND_QUEUE_SIZE][TDMA_DESC_PER_REQUEST];
    struct tuc_block_t blocks[TUC_COMMAND_QUEUE_SIZE];
    struct commit_info_t commits_to_be_sent[TUC_INTERRUPT_PERIOD];
    uint64_t cluster_timestamps[NUM_MAX_CLUSTERS];
    uint64_t dummy[211];
    // shared fields between task and PERT
    void (*msg_send)(uint64_t, uint64_t, uint64_t);
    uint64_t tuc_profile_level;
    // `panic_message_ptr`, `panic_lr` is considered unstable implementation detail
    const char *panic_message_ptr;
    uint64_t panic_lr;
    __attribute__((noreturn)) void (*panic)(const char *);
    uint64_t (*trampoline)(uint64_t, uint64_t, uint64_t);
    struct span_handle_t (*span_enter)(const char *, uint64_t);
    void (*span_leave)(struct span_handle_t);
    uint64_t npu_id;                       // F_FFD8
    uint64_t npu_vid;                      // F_FFE0
    uint64_t pe_id;                        // F_FFE8
    void (*commit_fn)(uint64_t, uint32_t); // F_FFF0
    void (*pert_isr)(uint32_t);            // F_FFF8
};

/* Shared data structure */
static volatile struct shared_field_t *const shared = (struct shared_field_t *)SHARED_FIELDS_ADDR;)""";
    os << "\n";
    os << "\n";

    // Define function
    os << "void " << functionOp.getName() << "() {\n";
    Operation *operation = functionOp.getOperation();
    if (failed(
            printFunctionBody(armCEmitter, operation, functionOp.getBlocks())))
      return failure();
    os << "}\n";

    if (fd_os.has_error())
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

  // Convert elf into machine code
  command = "aarch64-none-elf-objcopy ";
  command += "-O binary ";
  command += filepath_o.str() + " ";
  command += filepath_bin.str() + " ";
  system(command.c_str());

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
  os << "shared->trampoline(TRAMPOLINE_EXIT, 0, 0)";
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
