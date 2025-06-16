#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "furiosa-mlir/Dialect/Host/IR/HostOps.h"
#include "furiosa-mlir/ExecutionEngine/RenegadeRuntime.h"
#include "furiosa-mlir/Target/C/FuriosaTaskToC.h"

using namespace mlir;

namespace mlir::furiosa {

class Printer {
public:
  static void print(bool value) {
    llvm::outs() << (value ? "true" : "false") << "\n";
  }
  static void print(byte_array_t &buffer) {
    llvm::outs() << "[";
    llvm::ListSeparator LS;
    for (auto byte : buffer) {
      llvm::outs() << LS << llvm::format("0x%02x", byte);
    }
    llvm::outs() << "]\n";
  }
};

LogicalResult executeOperation(ExecutionContext &context, func::ReturnOp op) {
  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::AllocOp op) {
  byte_array_t data_buffer;
  auto size = op.getSize();
  if (auto data = op->getAttrOfType<ArrayAttr>("data")) {
    if (data.empty()) {
      // randomize input when data is empty
      // seed is fixed for testing
      data_buffer.resize(size);
      std::generate(data_buffer.begin(), data_buffer.end(), [&]() {
        return static_cast<std::uint8_t>(context.getRandomNumber() % 256);
      });
    } else {
      for (auto d : data) {
        data_buffer.push_back(dyn_cast_or_null<IntegerAttr>(d).getInt());
      }
      auto input_data_size = data_buffer.size();
      data_buffer.reserve(CEIL(size, input_data_size));
      for (auto i = 0u; i < CEIL(size, input_data_size); i += input_data_size) {
        std::copy_n(data_buffer.begin(), input_data_size,
                    std::back_inserter(data_buffer));
      }
    }
  }
  data_buffer.resize(size);
  data_buffer.resize(CEIL(data_buffer.size(), DRAM_ACCESS_WIDTH));
  context.createValue(op->getResult(0), llvm::Any(data_buffer));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::FuncAllocOp op) {
  auto name = op.getFunction();
  auto function = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(context.getModule(), name));
  if (!function || function.empty()) {
    llvm::report_fatal_error(llvm::Twine("entry point not found"));
    return failure();
  }
  auto binary = translateKernelFunctionToBinary(function);
  if (failed(binary)) {
    llvm::report_fatal_error(llvm::Twine("failed to translate kernel"));
    return failure();
  }
  byte_array_t data_buffer;
  for (auto d : *binary) {
    data_buffer.push_back(reinterpret_cast<std::uint8_t &>(d));
  }
  data_buffer.resize(CEIL(data_buffer.size(), DRAM_ACCESS_WIDTH));
  context.createValue(op->getResult(0), llvm::Any(data_buffer));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::CompareOp op) {
  auto &buffer0 =
      llvm::any_cast<byte_array_t &>(context.getValue(op.getBuffer0()));
  auto &buffer1 =
      llvm::any_cast<byte_array_t &>(context.getValue(op.getBuffer1()));
  context.createValue(op->getResult(0), llvm::Any(buffer0 == buffer1));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::PrintOp op) {
  auto buffer = op.getBuffer();
  auto &buffer_any = context.getValue(buffer);
  if (llvm::any_cast<bool>(&buffer_any)) {
    auto &buffer_data = llvm::any_cast<bool &>(buffer_any);
    Printer::print(buffer_data);
  } else if (llvm::any_cast<byte_array_t>(&buffer_any)) {
    auto &buffer_data = llvm::any_cast<byte_array_t &>(buffer_any);
    Printer::print(buffer_data);
  } else {
    llvm::report_fatal_error(llvm::Twine("operand type cannot be printed"));
    return failure();
  }

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::PeProgramLoadInstOp op) {
  auto dram_address = op.getDramAddress();
  auto spm_address = op.getSpmAddress();
  auto &buffer =
      llvm::any_cast<byte_array_t &>(context.getValue(op.getBinary()));
  pe_program_t programs;
  programs.push_back(device_runtime::pe_program_load_inst(
      dram_address, spm_address, buffer.size()));
  context.createValue(op->getResult(0), llvm::Any(programs));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::PeProgramLaunchOp op) {
  auto spm_address = op.getSpmAddress();
  pe_program_t programs;
  std::vector<std::uint64_t> operands;
  for (auto operand : op.getOperands()) {
    operands.push_back(dyn_cast_or_null<IntegerAttr>(operand).getInt());
  }
  programs.push_back(device_runtime::pe_program_launch(
      spm_address, 0, 0, 0, operands.data(), operands.size()));
  context.createValue(op->getResult(0), llvm::Any(programs));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::PeProgramSeqOp op) {
  auto pe_programs = op.getPePrograms();
  pe_program_t program_list;
  for (auto pe_program : pe_programs) {
    auto &program =
        llvm::any_cast<pe_program_t &>(context.getValue(pe_program));
    program_list.insert(program_list.end(), program.begin(), program.end());
  }
  pe_program_t merged_programs;
  merged_programs.push_back(
      device_runtime::pe_program_seq(program_list.data(), program_list.size()));
  context.createValue(op->getResult(0), llvm::Any(merged_programs));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::HalProgramWriteAtOp op) {
  auto dram_address = op.getDramAddress();
  auto &buffer =
      llvm::any_cast<byte_array_t &>(context.getValue(op.getBuffer()));
  hal_program_t programs;
  programs.push_back(device_runtime::hal_program_write_at(
      reinterpret_cast<std::uint64_t>(buffer.data()), dram_address,
      buffer.size()));
  context.createValue(op->getResult(0), llvm::Any(programs));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::HalProgramReadAtOp op) {
  auto dram_address = op.getDramAddress();
  auto &buffer =
      llvm::any_cast<byte_array_t &>(context.getValue(op.getBuffer()));
  hal_program_t programs;
  programs.push_back(device_runtime::hal_program_read_at(
      dram_address, reinterpret_cast<std::uint64_t>(buffer.data()),
      buffer.size()));
  context.createValue(op->getResult(0), llvm::Any(programs));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::HalProgramExecuteOp op) {
  auto &pe_program =
      llvm::any_cast<pe_program_t &>(context.getValue(op.getPeProgram()));
  hal_program_t programs;
  assert(pe_program.size() == 1);
  programs.push_back(device_runtime::hal_program_execute(pe_program[0]));
  context.createValue(op->getResult(0), llvm::Any(programs));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::HalProgramSeqOp op) {
  auto hal_programs = op.getHalPrograms();
  hal_program_t program_list;
  for (auto hal_program : hal_programs) {
    auto &program =
        llvm::any_cast<hal_program_t &>(context.getValue(hal_program));
    program_list.insert(program_list.end(), program.begin(), program.end());
  }
  hal_program_t merged_programs;
  merged_programs.push_back(device_runtime::hal_program_seq(
      program_list.data(), program_list.size()));
  context.createValue(op->getResult(0), llvm::Any(merged_programs));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::DeviceNewOp op) {
  auto target = op.getTarget();
  auto npu = target.getNpu();
  auto pe_begin = target.getPeBegin();
  auto pe_end = target.getPeEnd();
  device_t device = device_runtime::device_new(npu, pe_begin, pe_end);
  context.createValue(op->getResult(0), llvm::Any(device));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::DeviceExecuteOp op) {
  auto &hal_program =
      llvm::any_cast<hal_program_t &>(context.getValue(op.getHalProgram()));
  auto &device = llvm::any_cast<device_t &>(context.getValue(op.getDevice()));
  assert(hal_program.size() == 1);
  auto execution = device_runtime::device_execute(device, hal_program[0]);
  context.createValue(op->getResult(0), llvm::Any(execution));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::DeviceExecutionWaitOp op) {
  auto &execution =
      llvm::any_cast<execution_t &>(context.getValue(op.getExecution()));
  if (!device_runtime::device_execution_wait(execution)) {
    llvm::report_fatal_error(llvm::Twine("device execution wait failed"));
    return failure();
  }

  return success();
}

LogicalResult executeOperation(ExecutionContext &context, Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          .Case<func::ReturnOp>(
              [&](auto op) { return executeOperation(context, op); })
          .Case<furiosa::host::AllocOp, furiosa::host::FuncAllocOp,
                furiosa::host::CompareOp, furiosa::host::PrintOp,
                furiosa::host::PeProgramLoadInstOp,
                furiosa::host::PeProgramLaunchOp, furiosa::host::PeProgramSeqOp,
                furiosa::host::HalProgramWriteAtOp,
                furiosa::host::HalProgramReadAtOp,
                furiosa::host::HalProgramExecuteOp,
                furiosa::host::HalProgramSeqOp, furiosa::host::DeviceNewOp,
                furiosa::host::DeviceExecuteOp,
                furiosa::host::DeviceExecutionWaitOp>(
              [&](auto op) { return executeOperation(context, op); })
          .Default([&](Operation *) {
            return op.emitOpError("unable to find executor for op");
          });

  if (failed(status))
    return failure();

  return success();
}

LogicalResult executeFunction(Operation *module, StringRef entry_point,
                              StringRef entry_point_type) {
  ExecutionContext context = ExecutionContext(module);

  auto main_function = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(module, entry_point));
  if (!main_function || main_function.empty()) {
    llvm::report_fatal_error(llvm::Twine("entry point not found"));
    return failure();
  }

  // Emit the body of the function.
  for (Block &block : main_function.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      if (failed(executeOperation(context, op)))
        return failure();
    }
  }

  return success();
}

} // namespace mlir::furiosa
