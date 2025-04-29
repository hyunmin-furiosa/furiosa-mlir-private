#include <stack>

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaTaskOps.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaTucOps.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaTypes.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/Utils.h"
#include "furiosa-mlir/Target/Furiosa/Binary.h"
#include "furiosa-mlir/Target/Furiosa/Utils.h"

using namespace mlir;

namespace mlir::furiosa {

/// Emitter that uses dialect specific emitters to emit Arm C code.
struct ArmCEmitter {
  explicit ArmCEmitter(raw_ostream &os);

  /// Emits operation 'op' or returns failure.
  LogicalResult emitOperation(Operation &op);

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value val) {
    if (!valueMapper.count(val)) {
      valueMapper.insert(val, llvm::formatv("v{0}", ++valueInScopeCount.top()));
    }
    return *valueMapper.begin(val);
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

static LogicalResult printFuriosaCommand(ArmCEmitter &emitter, Operation *op) {
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
  for (auto it = sfr_vector.begin(); it != sfr_vector.end(); ++it) {
    os << llvm::format_hex(*it, 0);
    if (it != sfr_vector.end() - 1)
      os << ", ";
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
  for (auto it = sfr_vector.begin(); it != sfr_vector.end(); ++it) {
    os << llvm::format_hex(*it, 0);
    if (it != sfr_vector.end() - 1)
      os << ", ";
  }
  os << " };\n";

  return success();
}

static LogicalResult
printStaticDmaDescriptor(ArmCEmitter &emitter,
                         furiosa::task::StaticDmaDescriptorOp op) {
  raw_indented_ostream &os = emitter.ostream();
  auto descriptor = *getDmaDescriptor(op);

  os << "{\n";
  os.indent();
  os << "static const struct dma_desc_t _desc = { ";
  os << descriptor.opcode << ", ";
  os << 0 << ", ";
  os << llvm::format_hex(descriptor.source_base, 0) << ", ";
  os << llvm::format_hex(descriptor.destination_base, 0) << ", ";
  os << "{ ";
  for (auto i = 0u; i < DIMS; i++) {
    os << descriptor.source_limits[i];
    if (i != DIMS - 1)
      os << ", ";
  }
  os << " }, ";
  os << "{ ";
  for (auto i = 0u; i < DIMS; i++) {
    os << descriptor.source_strides[i];
    if (i != DIMS - 1)
      os << ", ";
  }
  os << " }, ";
  os << "{ ";
  for (auto i = 0u; i < DIMS; i++) {
    os << descriptor.destination_limits[i];
    if (i != DIMS - 1)
      os << ", ";
  }
  os << " }, ";
  os << "{ ";
  for (auto i = 0u; i < DIMS; i++) {
    os << descriptor.destination_strides[i];
    if (i != DIMS - 1)
      os << ", ";
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
                                        furiosa::task::DmaDescriptorOp op) {
  raw_indented_ostream &os = emitter.ostream();
  auto descriptor = *getDmaDescriptor(op);

  OpResult result = op->getResult(0);
  os << "static const struct dma_desc_t " << emitter.getOrCreateName(result)
     << " = { ";
  os << descriptor.opcode << ", ";
  os << 0 << ", ";
  os << llvm::format_hex(descriptor.source_base, 0) << ", ";
  os << llvm::format_hex(descriptor.destination_base, 0) << ", ";
  os << "{ ";
  for (auto i = 0u; i < DIMS; i++) {
    os << descriptor.source_limits[i];
    if (i != DIMS - 1)
      os << ", ";
  }
  os << " }, ";
  os << "{ ";
  for (auto i = 0u; i < DIMS; i++) {
    os << descriptor.source_strides[i];
    if (i != DIMS - 1)
      os << ", ";
  }
  os << " }, ";
  os << "{ ";
  for (auto i = 0u; i < DIMS; i++) {
    os << descriptor.destination_limits[i];
    if (i != DIMS - 1)
      os << ", ";
  }
  os << " }, ";
  os << "{ ";
  for (auto i = 0u; i < DIMS; i++) {
    os << descriptor.destination_strides[i];
    if (i != DIMS - 1)
      os << ", ";
  }
  os << " } ";
  os << "};\n";

  return success();
};

static LogicalResult printDynamicMtosfr(ArmCEmitter &emitter,
                                        task::MtosfrOp op) {
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

static LogicalResult printDynamicDmaw(ArmCEmitter &emitter, task::DmawOp op) {
  raw_indented_ostream &os = emitter.ostream();
  auto [command, registers] = *getCommand(*op.getOperation());

  static constexpr std::uint64_t remotePeBase = 0x80'0000'0000;
  static constexpr std::uint64_t peSize = 0x2000'0000; // 512MB
  std::uint64_t remotePeOffset = 0;
  if (auto functionOp = op->getParentOfType<func::FuncOp>()) {
    auto targetAttr = functionOp->getAttrOfType<furiosa::TargetAttr>("target");
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

static LogicalResult printKernelFunction(func::FuncOp functionOp) {
  int fd;
  llvm::Twine symName = functionOp.getSymName();
  llvm::Twine filepath_c = symName + ".c";
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
    ArmCEmitter::Scope scope(armCEmitter);

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
    os << "__attribute__ ((section (\".text.main\"))) void "
       << functionOp.getName() << "(";
    Operation *operation = functionOp.getOperation();
    if (failed(printFunctionArgs(armCEmitter, operation,
                                 functionOp.getArguments())))
      return failure();
    os << ") {\n";
    if (failed(
            printFunctionBody(armCEmitter, operation, functionOp.getBlocks())))
      return failure();
    os << "\n";
    os << "}\n";

    if (fd_os.has_error())
      llvm::report_fatal_error(llvm::Twine("Error emitting the IR to file '") +
                               filepath_c);
  }

  FuriosaBinary furiosaBinary{};

  auto targetAttr = functionOp->getAttrOfType<furiosa::TargetAttr>("target");
  furiosaBinary.metadata.npu = targetAttr.getNpu();
  furiosaBinary.metadata.peBegin = targetAttr.getPeBegin();
  furiosaBinary.metadata.peEnd = targetAttr.getPeEnd();

  auto addressAttr = functionOp->getAttrOfType<furiosa::AddressAttr>("address");
  furiosaBinary.metadata.binaryAddress = addressAttr.getAddress();

  std::string filepath_o = *convertArmCToObject(filepath_c);
  std::string filepath_link = *linkObject(filepath_o);
  furiosaBinary.binary = *convertObjectToBinary(filepath_link);
  furiosaBinary.metadata.binarySize = furiosaBinary.binary.size();

  for (auto arg : functionOp.getArgumentTypes()) {
    if (auto tensorType = llvm::cast<RankedTensorType>(arg)) {
      if (auto addressAttr =
              llvm::cast<furiosa::AddressAttr>(tensorType.getEncoding())) {
        std::uint64_t address = addressAttr.getAddress();
        std::uint64_t size = tensorType.getNumElements() *
                             tensorType.getElementTypeBitWidth() / CHAR_BIT;
        furiosaBinary.arguments.push_back(std::make_pair(address, size));
      }
    }
  }
  furiosaBinary.metadata.argumentSize = furiosaBinary.arguments.size();

  for (auto res : functionOp.getResultTypes()) {
    if (auto tensorType = llvm::cast<RankedTensorType>(res)) {
      if (auto addressAttr =
              llvm::cast<furiosa::AddressAttr>(tensorType.getEncoding())) {
        std::uint64_t address = addressAttr.getAddress();
        std::uint64_t size = tensorType.getNumElements() *
                             tensorType.getElementTypeBitWidth() / CHAR_BIT;
        furiosaBinary.results.push_back(std::make_pair(address, size));
      }
    }
  }
  furiosaBinary.metadata.resultSize = furiosaBinary.results.size();

  if (failed(writeFuriosaBinary("furiosa.bin", furiosaBinary))) {
    return failure();
  }

  return success();
}

static LogicalResult printOperation(ArmCEmitter &emitter,
                                    func::FuncOp functionOp) {
  if (auto targetAttr =
          functionOp->getAttrOfType<furiosa::TargetAttr>("target")) {
    return printKernelFunction(functionOp);
  }
  return success();
}

static LogicalResult printOperation(ArmCEmitter &emitter,
                                    func::ReturnOp returnOp) {
  raw_indented_ostream &os = emitter.ostream();
  os << "shared->trampoline(TRAMPOLINE_EXIT, 0, 0);";
  return success();
}

static LogicalResult printOperation(ArmCEmitter &emitter, ModuleOp moduleOp) {
  // raw_indented_ostream &os = emitter.ostream();
  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op)))
      return failure();
    // os << "\n";
  }
  return success();
}

ArmCEmitter::ArmCEmitter(raw_ostream &os) : os(os) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
}

LogicalResult ArmCEmitter::emitOperation(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
          // Func ops.
          .Case<func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<tensor::EmptyOp>([&](auto op) { return success(); })
          .Case<furiosa::tuc::ItosfrOp, furiosa::tuc::RtosfrOp,
                furiosa::tuc::RtosfriOp, furiosa::tuc::MtosfrOp,
                furiosa::tuc::StosfrOp, furiosa::tuc::SfrtosOp,
                furiosa::tuc::StallOp, furiosa::tuc::ItosOp,
                furiosa::tuc::ItosiOp, furiosa::tuc::StosOp,
                furiosa::tuc::StotabOp, furiosa::tuc::StotrfOp,
                furiosa::tuc::StovrfOp, furiosa::tuc::ExecutionOp,
                furiosa::tuc::WaitOp, furiosa::tuc::WaitiOp,
                furiosa::tuc::InterruptOp, furiosa::tuc::DmaOp,
                furiosa::tuc::Dma1Op, furiosa::tuc::DmawOp,
                furiosa::tuc::ProfileOp, furiosa::tuc::ProfileiOp,
                furiosa::tuc::PrflushOp>([&](auto op) {
            return printFuriosaCommand(*this, op.getOperation());
          })
          .Case<furiosa::task::StaticSfrDotProductEngineOp,
                furiosa::task::StaticSfrMainCommitUnitOp,
                furiosa::task::StaticSfrMainDataPathUnitOp,
                furiosa::task::StaticSfrMainFetchUnitOp,
                furiosa::task::StaticSfrRegisterConfigUnitOp,
                furiosa::task::StaticSfrSubCommitUnitOp,
                furiosa::task::StaticSfrSubDataPathUnitOp,
                furiosa::task::StaticSfrSubFetchUnitOp,
                furiosa::task::StaticSfrTensorRegisterFileOp,
                furiosa::task::StaticSfrTransposeEngineOp,
                furiosa::task::StaticSfrVectorArithmeticUnitOp,
                furiosa::task::StaticSfrVectorReduceUnitOp,
                furiosa::task::StaticSfrVectorRegisterFileOp,
                furiosa::task::StaticSfrVectorRouteUnitOp>(
              [&](auto op) { return printStaticSfr(*this, op.getOperation()); })
          .Case<furiosa::task::SfrDotProductEngineOp,
                furiosa::task::SfrMainCommitUnitOp,
                furiosa::task::SfrMainDataPathUnitOp,
                furiosa::task::SfrMainFetchUnitOp,
                furiosa::task::SfrRegisterConfigUnitOp,
                furiosa::task::SfrSubCommitUnitOp,
                furiosa::task::SfrSubDataPathUnitOp,
                furiosa::task::SfrSubFetchUnitOp,
                furiosa::task::SfrTensorRegisterFileOp,
                furiosa::task::SfrTransposeEngineOp,
                furiosa::task::SfrVectorArithmeticUnitOp,
                furiosa::task::SfrVectorReduceUnitOp,
                furiosa::task::SfrVectorRegisterFileOp,
                furiosa::task::SfrVectorRouteUnitOp>(
              [&](auto op) { return printSfr(*this, op.getOperation()); })
          .Case<furiosa::task::StaticDmaDescriptorOp>(
              [&](auto op) { return printStaticDmaDescriptor(*this, op); })
          .Case<furiosa::task::DmaDescriptorOp>(
              [&](auto op) { return printDmaDescriptor(*this, op); })
          .Case<furiosa::task::MtosfrOp>(
              [&](auto op) { return printDynamicMtosfr(*this, op); })
          .Case<furiosa::task::DmawOp>(
              [&](auto op) { return printDynamicDmaw(*this, op); })
          .Default([&](Operation *) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status))
    return failure();

  return success();
}

LogicalResult translateFuriosaToBinary(Operation *op, llvm::raw_ostream &os) {
  ArmCEmitter emitter(os);
  LogicalResult status = emitter.emitOperation(*op);
  if (failed(status))
    return failure();

  return status;
}

} // namespace mlir::furiosa
