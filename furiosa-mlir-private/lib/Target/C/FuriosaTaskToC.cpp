#include <stack>

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskOps.h"
#include "furiosa-mlir/Dialect/Task/IR/Utils.h"
#include "furiosa-mlir/Target/C/FuriosaTaskToC.h"
#include "furiosa-mlir/Target/C/Utils.h"

using namespace mlir;
using namespace mlir::furiosa::task;

namespace mlir::furiosa {

/// Emitter for outer code
struct FuriosaEmitter {
  explicit FuriosaEmitter(raw_ostream &os);

  /// Emits operation 'op' or returns failure.
  LogicalResult emitOperation(Operation &op);

  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; };

private:
  /// Output stream to emit to.
  raw_indented_ostream os;
};

/// Emitter that uses dialect specific emitters to emit Arm C code.
struct ArmCEmitter {
  explicit ArmCEmitter(raw_ostream &os);

  /// Emits operation 'op' or returns failure.
  LogicalResult emitOperation(Operation &op);

  /// Emits a declaration of a variable with the given type and name.
  LogicalResult emitVariableDeclaration(Location loc, Type type,
                                        StringRef name);

  /// Emits type 'type' or returns failure.
  LogicalResult emitType(Location loc, Type type);

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value val) {
    if (!valueMapper.count(val)) {
      valueMapper.insert(val, llvm::formatv("v{0}", valueInScopeCount.top()++));
    }
    return *valueMapper.begin(val);
  }

  /// Return the existing or a new name for a function argument Value.
  void createArgumentName(Value val) {
    if (!valueMapper.count(val)) {
      valueMapper.insert(
          val, llvm::formatv("addrs[{0}]", valueInScopeCount.top()++));
    }
  }

  /// RAII helper function to manage entering/exiting C++ scopes.
  struct Scope {
    Scope(ArmCEmitter &emitter)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.labelInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;
    ArmCEmitter &emitter;
  };

  bool hasValueInScope(Value val) { return valueMapper.count(val); }

  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; };

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;

  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Map from value to name of C++ variable that contain the name.
  ValueMapper valueMapper;

  /// Map from block to name of C++ label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;
};

static LogicalResult printTensorUnitCommand(ArmCEmitter &emitter,
                                            Operation *op) {
  raw_indented_ostream &os = emitter.ostream();
  auto [command, registers] = *getCommand(*op);

  for (auto [command_reg_idx, reg] : llvm::enumerate(registers)) {
    std::uint32_t general_reg_idx = command_reg_idx;
    os << "TUC_GENERAL_REGISTERS[" << general_reg_idx
       << "] = " << llvm::format_hex(reg.value, 0) << ";\n";
    command.setReg(command_reg_idx, general_reg_idx);
  }
  os << "TUC_COMMAND_QUEUE_ENTRY[tail] = " << llvm::format_hex(command.value, 0)
     << ";\n";
  os << "tail = (tail + 1) % TUC_COMMAND_QUEUE_SIZE;\n";
  os << "*TUC_COMMAND_QUEUE_TAIL = tail;\n";
  os << "while (*TUC_COMMAND_QUEUE_HEAD != tail) {};\n";
  return success();
}

static LogicalResult printStaticSfr(ArmCEmitter &emitter, Operation *op) {
  raw_indented_ostream &os = emitter.ostream();
  auto [sfr_address, sfr_vector] = *getStaticSfr(*op);

  os << "{\n";
  os.indent();
  os << "static const uint64_t _sfr[] = { ";
  llvm::ListSeparator LS;
  for (auto it = sfr_vector.begin(); it != sfr_vector.end(); ++it) {
    os << LS << llvm::format_hex(*it, 0);
  }
  os << " };\n";
  os << "memcpy((void *)" << llvm::format_hex(sfr_address, 0)
     << ", &_sfr, sizeof(_sfr));\n";
  os << "flush_cache((void *)" << llvm::format_hex(sfr_address, 0)
     << ", sizeof(_sfr));\n";
  os.unindent();
  os << "}\n";

  return success();
}

static LogicalResult printSfr(ArmCEmitter &emitter, Operation *op) {
  raw_indented_ostream &os = emitter.ostream();
  auto sfr_vector = *getSfr(*op);

  OpResult result = op->getResult(0);
  os << "static const uint64_t " << emitter.getOrCreateName(result)
     << "[] = { ";
  llvm::ListSeparator LS;
  for (auto it = sfr_vector.begin(); it != sfr_vector.end(); ++it) {
    os << LS << llvm::format_hex(*it, 0);
  }
  os << " };\n";

  return success();
}

static LogicalResult printStaticDmaDescriptor(ArmCEmitter &emitter,
                                              StaticDmaDescriptorOp op) {
  raw_indented_ostream &os = emitter.ostream();
  auto descriptor = *getDmaDescriptor(op);

  os << "{\n";
  os.indent();
  os << "static const struct dma_desc_t _desc = { ";
  os << descriptor.opcode << ", ";
  os << descriptor.indirect.value << ", ";
  os << llvm::format_hex(descriptor.source_base, 0) << ", ";
  os << llvm::format_hex(descriptor.destination_base, 0) << ", ";
  os << "{ ";
  llvm::ListSeparator LS;
  for (auto i = 0u; i < DIMS; i++) {
    os << LS << descriptor.source_limits[i];
  }
  os << " }, ";
  os << "{ ";
  LS = llvm::ListSeparator();
  for (auto i = 0u; i < DIMS; i++) {
    os << LS << descriptor.source_strides[i];
  }
  os << " }, ";
  os << "{ ";
  LS = llvm::ListSeparator();
  for (auto i = 0u; i < DIMS; i++) {
    os << LS << descriptor.destination_limits[i];
  }
  os << " }, ";
  os << "{ ";
  LS = llvm::ListSeparator();
  for (auto i = 0u; i < DIMS; i++) {
    os << LS << descriptor.destination_strides[i];
  }
  os << " } ";
  os << "};\n";
  os << "memcpy((void *)" << llvm::format_hex(op.getDescAddr(), 0)
     << ", &_desc, sizeof(struct dma_desc_t));\n";
  os << "flush_cache((void *)" << llvm::format_hex(op.getDescAddr(), 0)
     << ", sizeof(struct dma_desc_t));\n";
  os.unindent();
  os << "}\n";

  return success();
}

static LogicalResult printDmaDescriptor(ArmCEmitter &emitter,
                                        DmaDescriptorOp op) {
  raw_indented_ostream &os = emitter.ostream();
  auto descriptor = *getDmaDescriptor(op);

  OpResult result = op->getResult(0);
  os << "struct dma_desc_t " << emitter.getOrCreateName(result) << " = { ";
  os << descriptor.opcode << ", ";
  os << descriptor.indirect.value << ", ";

  if (auto source = op.getSource()) {
    os << "DRAM_BASE | " << emitter.getOrCreateName(source) << ", ";
  } else {
    os << llvm::format_hex(descriptor.source_base, 0) << ", ";
  }
  if (auto destination = op.getDestination()) {
    os << "DRAM_BASE | " << emitter.getOrCreateName(destination) << ", ";
  } else {
    os << llvm::format_hex(descriptor.destination_base, 0) << ", ";
  }
  os << "{ ";
  llvm::ListSeparator LS;
  for (auto i = 0u; i < DIMS; i++) {
    os << LS << descriptor.source_limits[i];
  }
  os << " }, ";
  os << "{ ";
  LS = llvm::ListSeparator();
  for (auto i = 0u; i < DIMS; i++) {
    os << LS << descriptor.source_strides[i];
  }
  os << " }, ";
  os << "{ ";
  LS = llvm::ListSeparator();
  for (auto i = 0u; i < DIMS; i++) {
    os << LS << descriptor.destination_limits[i];
  }
  os << " }, ";
  os << "{ ";
  LS = llvm::ListSeparator();
  for (auto i = 0u; i < DIMS; i++) {
    os << LS << descriptor.destination_strides[i];
  }
  os << " } ";
  os << "};\n";
  os << "flush_cache((void *)&" << emitter.getOrCreateName(result)
     << ", sizeof(struct dma_desc_t));\n";

  return success();
};

static LogicalResult printDynamicMtosfr(ArmCEmitter &emitter,
                                        DynamicMtosfrOp op) {
  raw_indented_ostream &os = emitter.ostream();
  auto [command, registers] = *getCommand(*op.getOperation());

  Value operand = op->getOperand(0);
  auto operandName = emitter.getOrCreateName(operand);
  os << "TUC_GENERAL_REGISTERS[0] = " << llvm::format_hex(registers[0].value, 0)
     << " | ((uint64_t)" << operandName << " & 0xffffff)"
     << " | (((sizeof(" << operandName << ") / sizeof(" << operandName
     << "[0])) & 0xff) << 24)"
     << ";\n";
  command.setReg(0, 0);
  os << "TUC_COMMAND_QUEUE_ENTRY[tail] = " << llvm::format_hex(command.value, 0)
     << ";\n";
  os << "tail = (tail + 1) % TUC_COMMAND_QUEUE_SIZE;\n";
  os << "*TUC_COMMAND_QUEUE_TAIL = tail;\n";
  os << "while (*TUC_COMMAND_QUEUE_HEAD != tail) {};\n";
  os << "\n";
  return success();
}

static LogicalResult printDynamicDmaw(ArmCEmitter &emitter, DynamicDmawOp op) {
  raw_indented_ostream &os = emitter.ostream();
  auto [command, registers] = *getCommand(*op.getOperation());

  static constexpr std::uint64_t remotePeBase = 0x80'0000'0000;
  static constexpr std::uint64_t peSize = 0x2000'0000; // 512MB
  std::uint64_t remotePeOffset = 0;
  if (auto functionOp = op->getParentOfType<func::FuncOp>()) {
    auto targetAttr = functionOp->getAttrOfType<TargetAttr>("target");
    auto clusterPeBegin = targetAttr.getPeBegin() % 4; // within cluster
    remotePeOffset = remotePeBase + clusterPeBegin * peSize;
  }

  Value operand = op->getOperand(0);
  auto operandName = emitter.getOrCreateName(operand);
  os << "TUC_GENERAL_REGISTERS[0] = " << llvm::format_hex(registers[0].value, 0)
     << " | ((uint64_t) &" << operandName << " & 0xffffffffff);\n";
  os << "TUC_GENERAL_REGISTERS[1] = " << llvm::format_hex(registers[1].value, 0)
     << " | ((uint64_t) &" << operandName << " & 0xffffffffff) | "
     << llvm::format_hex(remotePeOffset, 0) << ";\n"; // remote pe0
  os << "TUC_GENERAL_REGISTERS[2] = " << llvm::format_hex(registers[2].value, 0)
     << " | ((uint64_t) &" << operandName << " & 0xffffffffff) | "
     << llvm::format_hex(remotePeOffset, 0) << ";\n"; // remote pe0
  os << "TUC_GENERAL_REGISTERS[3] = " << llvm::format_hex(registers[3].value, 0)
     << " | ((uint64_t) &" << operandName << " & 0xffffffffff) | "
     << llvm::format_hex(remotePeOffset, 0) << ";\n"; // remote pe0
  command.setReg(0, 0);
  command.setReg(1, 1);
  command.setReg(2, 2);
  command.setReg(3, 3);
  os << "TUC_COMMAND_QUEUE_ENTRY[tail] = " << llvm::format_hex(command.value, 0)
     << ";\n";
  os << "tail = (tail + 1) % TUC_COMMAND_QUEUE_SIZE;\n";
  os << "*TUC_COMMAND_QUEUE_TAIL = tail;\n";
  os << "while (*TUC_COMMAND_QUEUE_HEAD != tail) {};\n";
  os << "\n";
  return success();
}

static LogicalResult printFunctionArgs(ArmCEmitter &emitter,
                                       Operation *functionOp,
                                       Region::BlockArgListType arguments) {
  raw_indented_ostream &os = emitter.ostream();
  os << "uint64_t *addrs, uint32_t len_addrs, uint16_t profile_base_uid";

  // register real function arguments under addrs
  // these are always the first values in the scope (addrs[0], addrs[1], ...)
  for (auto arg : arguments) {
    emitter.createArgumentName(arg);
  }
  return success();
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
      if (failed(emitter.emitOperation(op)))
        return failure();
    }
  }
  os.unindent();

  return success();
}

static LogicalResult printKernelFunction(ArmCEmitter &emitter,
                                         func::FuncOp functionOp) {
  raw_indented_ostream &os = emitter.ostream();

  // Necessary defines and includes
  os << R"""(#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>

#include "builtin.c"
#include "printf.c"

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

#define NUM_MAX_CLUSTERS 8

#define SRAM_BASE UINT64_C(0x0010000000)
#define DRAM_BASE UINT64_C(0xC000000000)
#define TUC_SFR_BASE UINT64_C(0x000E000000)
#define TDMA_BASE UINT64_C(0x0000C00000)
#define REMOTE_PE_DATA_BASE UINT64_C(0x8000000000)
#define REMOTE_ENTRY_SIZE UINT64_C(0x100000000)
#define REMOTE_ENTRY_SIZE_PER_PE UINT64_C(0x20000000)

#define NUM_SLICES 64
#define SLICE_SPACE_SIZE UINT64_C(0x400000) /* 4MB */

#define TDMA_DESC_QUEUE_SUB_HEAD ((volatile uint64_t *)(TDMA_BASE + 0x040))
#define TDMA_DESC_QUEUE_SUB_TAIL ((volatile uint64_t *)(TDMA_BASE + 0x048))
#define TDMA_DESC_QUEUE_COMP_HEAD ((volatile uint64_t *)(TDMA_BASE + 0x050))
#define TDMA_DESC_QUEUE_COMP_TAIL ((volatile uint64_t *)(TDMA_BASE + 0x058))
#define TDMA_DESC_QUEUE_ENTRY ((volatile uint64_t *)(TDMA_BASE + 0x100))
#define TDMA_DESC_QUEUE_SIZE 32

#define TDMA_WORD_SIZE 32

/**
 * Flushes data cache to scratchpad memory.
 *
 * Given memory range, begin..begin + size must be accessed under volatile
 * qualifier to ensure stores to the range are written to the physical memory
 * (either cache or main data memory).
 *
 * This function only ensures modifications on cache will be applied to the data
 * memory.
 */
static inline void flush_cache(volatile void *begin, size_t size)
{
  /**
   * Arm Cortex-A55 has L1 data cache which has 64-byte cache line length.
   * Refer to:
   * https://developer.arm.com/documentation/100442/0100/functional-description/level-1-memory-system/about-the-l1-memory-system
   */
  const size_t cache_line_size = 64;

  uintptr_t begin_aligned = (uintptr_t)begin & ~(cache_line_size - 1);

  /* inclusive */
  uintptr_t end_aligned =
      ((uintptr_t)begin + size - 1) & ~(cache_line_size - 1);

  /* Before flushing, ensure all previous stores are committed to cache. */
  __asm__ __volatile__("dsb sy");

  for (uintptr_t address = begin_aligned; address <= end_aligned;
       address += cache_line_size)
  {
    __asm__ __volatile__("dc cvac, %0" : : "r"(address));
  }

  /* Ensure that flushing cache completes. */
  __asm__ __volatile__("dsb sy");
}

/**
 * Invalidates data cache.
 *
 * Given memory range, begin..begin + size must be accessed under volatile
 * qualifier to ensure loads from the range are read from the physical memory
 * (either cache or main data memory).
 *
 * This function only ensures modifications on data memory will be applited to
 * cache.
 */
static inline void invalidate_cache(volatile void *begin, size_t size)
{
  /**
   * Arm Cortex-A55 has L1 data cache which has 64-byte cache line length.
   * Refer to:
   * https://developer.arm.com/documentation/100442/0100/functional-description/level-1-memory-system/about-the-l1-memory-system
   */
  const size_t cache_line_size = 64;

  uintptr_t begin_aligned = (uintptr_t)begin & ~(cache_line_size - 1);

  /* inclusive */
  uintptr_t end_aligned =
      ((uintptr_t)begin + size - 1) & ~(cache_line_size - 1);

  /* Ensure that data invalidation of cache starts after all memory accesses are
   * finished. */
  __asm__ __volatile__("dsb sy");

  for (uintptr_t address = begin_aligned; address <= end_aligned;
       address += cache_line_size)
  {
    __asm__ __volatile__("dc ivac, %0" : : "r"(address));
  }

  /* Ensure that invalidation of cache completes. */
  __asm__ __volatile__("dsb sy");
}

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
static volatile struct shared_field_t *const shared = (struct shared_field_t *)SHARED_FIELDS_ADDR;
)""";
  os << "\n";

  // Define function
  ArmCEmitter::Scope scope(emitter);
  os << "__attribute__ ((section (\".text.main\"))) void "
     << functionOp.getName() << "(";
  Operation *operation = functionOp.getOperation();
  if (failed(printFunctionArgs(emitter, operation, functionOp.getArguments())))
    return failure();
  os << ") {\n";
  if (failed(printFunctionBody(emitter, operation, functionOp.getBlocks())))
    return failure();
  os << "\n";
  os << "}\n";

  return success();
}

static LogicalResult printOperation(FuriosaEmitter &emitter,
                                    func::CallOp callOp) {
  return success();
}

static LogicalResult printOperation(FuriosaEmitter &emitter,
                                    func::FuncOp functionOp) {
  raw_indented_ostream &os = emitter.ostream();
  ArmCEmitter armCEmitter(os);
  if (auto targetAttr = functionOp->getAttrOfType<TargetAttr>("target")) {
    return printKernelFunction(armCEmitter, functionOp);
  } else {
    for (Block &block : functionOp.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        if (failed(emitter.emitOperation(op)))
          return failure();
      }
    }
  }
  return success();
}

static LogicalResult printOperation(FuriosaEmitter &emitter,
                                    func::ReturnOp returnOp) {
  return success();
}

static LogicalResult printOperation(ArmCEmitter &emitter,
                                    func::ReturnOp returnOp) {
  raw_indented_ostream &os = emitter.ostream();
  os << "shared->trampoline(TRAMPOLINE_EXIT, 0, 0);";
  return success();
}

static LogicalResult printOperation(FuriosaEmitter &emitter,
                                    ModuleOp moduleOp) {
  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op)))
      return failure();
    // os << "\n";
  }
  return success();
}

FuriosaEmitter::FuriosaEmitter(raw_ostream &os) : os(os) {}

LogicalResult FuriosaEmitter::emitOperation(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
          // Func ops.
          .Case<func::CallOp, func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<tensor::EmptyOp>([&](auto op) { return success(); })
          .Default([&](Operation *) { return success(); });

  if (failed(status))
    return failure();

  return success();
}

ArmCEmitter::ArmCEmitter(raw_ostream &os) : os(os) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
}

LogicalResult ArmCEmitter::emitOperation(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Func ops.
          .Case<func::ReturnOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<tuc::ItosfrOp, tuc::RtosfrOp, tuc::RtosfriOp, tuc::MtosfrOp,
                tuc::StosfrOp, tuc::SfrtosOp, tuc::StallOp, tuc::ItosOp,
                tuc::ItosiOp, tuc::StosOp, tuc::StotabOp, tuc::StotrfOp,
                tuc::StovrfOp, tuc::ExecutionOp, tuc::WaitOp, tuc::WaitiOp,
                tuc::InterruptOp, tuc::DmaOp, tuc::Dma1Op, tuc::DmawOp,
                tuc::ProfileOp, tuc::ProfileiOp, tuc::PrflushOp>([&](auto op) {
            return printTensorUnitCommand(*this, op.getOperation());
          })
          .Case<
              sfr::StaticSfrDotProductEngineOp, sfr::StaticSfrMainCommitUnitOp,
              sfr::StaticSfrMainDataPathUnitOp, sfr::StaticSfrMainFetchUnitOp,
              sfr::StaticSfrRegisterConfigUnitOp, sfr::StaticSfrSubCommitUnitOp,
              sfr::StaticSfrSubDataPathUnitOp, sfr::StaticSfrSubFetchUnitOp,
              sfr::StaticSfrTensorRegisterFileOp,
              sfr::StaticSfrTransposeEngineOp,
              sfr::StaticSfrVectorArithmeticUnitOp,
              sfr::StaticSfrVectorReduceUnitOp,
              sfr::StaticSfrVectorRegisterFileOp,
              sfr::StaticSfrVectorRouteUnitOp>(
              [&](auto op) { return printStaticSfr(*this, op.getOperation()); })
          .Case<sfr::SfrDotProductEngineOp, sfr::SfrMainCommitUnitOp,
                sfr::SfrMainDataPathUnitOp, sfr::SfrMainFetchUnitOp,
                sfr::SfrRegisterConfigUnitOp, sfr::SfrSubCommitUnitOp,
                sfr::SfrSubDataPathUnitOp, sfr::SfrSubFetchUnitOp,
                sfr::SfrTensorRegisterFileOp, sfr::SfrTransposeEngineOp,
                sfr::SfrVectorArithmeticUnitOp, sfr::SfrVectorReduceUnitOp,
                sfr::SfrVectorRegisterFileOp, sfr::SfrVectorRouteUnitOp>(
              [&](auto op) { return printSfr(*this, op.getOperation()); })
          .Case<StaticDmaDescriptorOp>(
              [&](auto op) { return printStaticDmaDescriptor(*this, op); })
          .Case<DmaDescriptorOp>(
              [&](auto op) { return printDmaDescriptor(*this, op); })
          .Case<DynamicMtosfrOp>(
              [&](auto op) { return printDynamicMtosfr(*this, op); })
          .Case<DynamicDmawOp>(
              [&](auto op) { return printDynamicDmaw(*this, op); })
          .Default([&](Operation *) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status))
    return failure();

  return success();
}

LogicalResult ArmCEmitter::emitVariableDeclaration(Location loc, Type type,
                                                   StringRef name) {
  if (failed(emitType(loc, type)))
    return failure();
  os << " " << name;
  return success();
}

LogicalResult ArmCEmitter::emitType(Location loc, Type type) {
  os << "uint64_t";
  return success();
}

FailureOr<binary_t> translateKernelFunctionToBinary(func::FuncOp functionOp) {
  int fd;
  llvm::Twine symName = functionOp.getSymName();

  SmallVector<char> filepath_c;
  llvm::sys::fs::createTemporaryFile(symName, "c", fd, filepath_c);
  if (std::error_code error = llvm::sys::fs::openFileForWrite(filepath_c, fd)) {
    llvm::report_fatal_error(llvm::Twine("Failed to open file: ") +
                             error.message());
    return failure();
  }
  {
    llvm::raw_fd_ostream fd_os(fd, /*shouldClose=*/true);
    ArmCEmitter emitter(fd_os);
    if (auto targetAttr = functionOp->getAttrOfType<TargetAttr>("target")) {
      if (failed(printKernelFunction(emitter, functionOp)))
        llvm::report_fatal_error(
            llvm::Twine("kernel function translation failed"));
    } else {
      llvm::report_fatal_error(
          llvm::Twine("this function does not have target"));
    }

    if (fd_os.has_error())
      llvm::report_fatal_error(llvm::Twine("Error emitting the IR to file '") +
                               filepath_c);
  }

  SmallVector<char> filepath_o;
  llvm::sys::fs::createTemporaryFile(symName, "o", fd, filepath_o);
  if (failed(convertArmCToObject(filepath_c, filepath_o))) {
    return failure();
  }

  SmallVector<char> filepath_o_linked;
  llvm::sys::fs::createTemporaryFile(symName, "o", fd, filepath_o_linked);
  if (failed(linkObject(filepath_o, filepath_o_linked))) {
    return failure();
  }

  SmallVector<char> filepath_bin;
  llvm::sys::fs::createTemporaryFile(symName, "bin", fd, filepath_bin);
  if (failed(convertObjectToBinary(filepath_o_linked, filepath_bin))) {
    return failure();
  }

  auto status = llvm::MemoryBuffer::getFile(filepath_bin);
  if (!status) {
    llvm::report_fatal_error(llvm::Twine("Failed to open file: ") +
                             status.getError().message());
  }
  llvm::BinaryByteStream stream(status->get()->getBuffer(),
                                llvm::endianness::native);

  binary_t binary = stream.str();

  return binary;
}

LogicalResult translateFuriosaToArmC(Operation *op, llvm::raw_ostream &os) {
  FuriosaEmitter emitter(os);
  LogicalResult status = emitter.emitOperation(*op);
  if (failed(status))
    return failure();

  return status;
}

} // namespace mlir::furiosa
