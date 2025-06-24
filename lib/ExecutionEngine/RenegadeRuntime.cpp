#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "furiosa-mlir/Conversion/Passes.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"
#include "furiosa-mlir/Dialect/Furiosa/Transforms/Passes.h"
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

LogicalResult executeOperation(ExecutionEngine &engine, func::ReturnOp op) {
  return success();
}

LogicalResult executeOperation(ExecutionEngine &engine,
                               furiosa::host::AllocOp op) {
  byte_array_t data_buffer;
  auto size = op.getSize();
  if (auto data_ptr = op->getAttrOfType<IntegerAttr>("data_ptr")) {
    // use data allocated outside of the execution engine
    auto address_size = address_size_t(data_ptr.getInt(), size);
    engine.createValue(op->getResult(0), llvm::Any(address_size));
  } else {
    // allocate data inside the execution engine
    if (auto data = op->getAttrOfType<ArrayAttr>("data")) {
      // data is provided
      if (data.empty()) {
        // randomize input when data is empty
        // seed is fixed for testing
        data_buffer.resize(size);
        std::generate(data_buffer.begin(), data_buffer.end(), [&]() {
          return static_cast<std::uint8_t>(engine.getRandomNumber() % 256);
        });
      } else {
        for (auto d : data) {
          data_buffer.push_back(dyn_cast_or_null<IntegerAttr>(d).getInt());
        }
        auto input_data_size = data_buffer.size();
        data_buffer.reserve(CEIL(size, input_data_size));
        for (auto i = 0u; i < CEIL(size, input_data_size);
             i += input_data_size) {
          std::copy_n(data_buffer.begin(), input_data_size,
                      std::back_inserter(data_buffer));
        }
      }
    }
    data_buffer.resize(size);
    data_buffer.resize(CEIL(data_buffer.size(), DRAM_ACCESS_WIDTH));
    engine.createValue(op->getResult(0), llvm::Any(data_buffer));
  }

  return success();
}

LogicalResult executeOperation(ExecutionEngine &engine,
                               furiosa::host::FuncAllocOp op) {
  auto name = op.getFunction();
  auto function = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(engine.getModule(), name));
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
  engine.createValue(op->getResult(0), llvm::Any(data_buffer));

  return success();
}

LogicalResult executeOperation(ExecutionEngine &engine,
                               furiosa::host::CompareOp op) {
  auto &buffer0 =
      llvm::any_cast<byte_array_t &>(engine.getValue(op.getBuffer0()));
  auto &buffer1 =
      llvm::any_cast<byte_array_t &>(engine.getValue(op.getBuffer1()));
  engine.createValue(op->getResult(0), llvm::Any(buffer0 == buffer1));

  return success();
}

LogicalResult executeOperation(ExecutionEngine &engine,
                               furiosa::host::PrintOp op) {
  auto buffer = op.getBuffer();
  auto &buffer_any = engine.getValue(buffer);
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

LogicalResult executeOperation(ExecutionEngine &engine,
                               furiosa::host::PeProgramLoadInstOp op) {
  auto dram_address = op.getDramAddress();
  auto spm_address = op.getSpmAddress();
  auto &buffer =
      llvm::any_cast<byte_array_t &>(engine.getValue(op.getBinary()));
  pe_program_t programs;
  programs.push_back(device_runtime::pe_program_load_inst(
      dram_address, spm_address, buffer.size()));
  engine.createValue(op->getResult(0), llvm::Any(programs));

  return success();
}

LogicalResult executeOperation(ExecutionEngine &engine,
                               furiosa::host::PeProgramLaunchOp op) {
  auto spm_address = op.getSpmAddress();
  pe_program_t programs;
  std::vector<std::uint64_t> operands;
  for (auto operand : op.getOperands()) {
    operands.push_back(dyn_cast_or_null<IntegerAttr>(operand).getInt());
  }
  programs.push_back(device_runtime::pe_program_launch(
      spm_address, 0, 0, 0, operands.data(), operands.size()));
  engine.createValue(op->getResult(0), llvm::Any(programs));

  return success();
}

LogicalResult executeOperation(ExecutionEngine &engine,
                               furiosa::host::PeProgramSeqOp op) {
  auto pe_programs = op.getPePrograms();
  pe_program_t program_list;
  for (auto pe_program : pe_programs) {
    auto &program = llvm::any_cast<pe_program_t &>(engine.getValue(pe_program));
    program_list.insert(program_list.end(), program.begin(), program.end());
  }
  pe_program_t merged_programs;
  merged_programs.push_back(
      device_runtime::pe_program_seq(program_list.data(), program_list.size()));
  engine.createValue(op->getResult(0), llvm::Any(merged_programs));

  return success();
}

address_size_t getAddressSize(llvm::Any &buffer) {
  if (llvm::any_cast<address_size_t>(&buffer)) {
    return llvm::any_cast<address_size_t &>(buffer);
  } else if (llvm::any_cast<byte_array_t>(&buffer)) {
    auto &buffer_data = llvm::any_cast<byte_array_t &>(buffer);
    return std::make_tuple(reinterpret_cast<std::uint64_t>(buffer_data.data()),
                           buffer_data.size());
  }
  llvm::report_fatal_error(
      llvm::Twine("cannot get address and size from buffer"));
  return std::make_tuple(0, 0);
}

LogicalResult executeOperation(ExecutionEngine &engine,
                               furiosa::host::HalProgramWriteAtOp op) {
  auto dram_address = op.getDramAddress();
  auto [buffer_address, buffer_size] =
      getAddressSize(engine.getValue(op.getBuffer()));
  hal_program_t programs;
  programs.push_back(device_runtime::hal_program_write_at(
      buffer_address, dram_address, buffer_size));
  engine.createValue(op->getResult(0), llvm::Any(programs));

  return success();
}

LogicalResult executeOperation(ExecutionEngine &engine,
                               furiosa::host::HalProgramReadAtOp op) {
  auto dram_address = op.getDramAddress();
  auto [buffer_address, buffer_size] =
      getAddressSize(engine.getValue(op.getBuffer()));
  hal_program_t programs;
  programs.push_back(device_runtime::hal_program_read_at(
      dram_address, buffer_address, buffer_size));
  engine.createValue(op->getResult(0), llvm::Any(programs));

  return success();
}

LogicalResult executeOperation(ExecutionEngine &engine,
                               furiosa::host::HalProgramExecuteOp op) {
  auto &pe_program =
      llvm::any_cast<pe_program_t &>(engine.getValue(op.getPeProgram()));
  hal_program_t programs;
  assert(pe_program.size() == 1);
  programs.push_back(device_runtime::hal_program_execute(pe_program[0]));
  engine.createValue(op->getResult(0), llvm::Any(programs));

  return success();
}

LogicalResult executeOperation(ExecutionEngine &engine,
                               furiosa::host::HalProgramSeqOp op) {
  auto hal_programs = op.getHalPrograms();
  hal_program_t program_list;
  for (auto hal_program : hal_programs) {
    auto &program =
        llvm::any_cast<hal_program_t &>(engine.getValue(hal_program));
    program_list.insert(program_list.end(), program.begin(), program.end());
  }
  hal_program_t merged_programs;
  merged_programs.push_back(device_runtime::hal_program_seq(
      program_list.data(), program_list.size()));
  engine.createValue(op->getResult(0), llvm::Any(merged_programs));

  return success();
}

LogicalResult executeOperation(ExecutionEngine &engine,
                               furiosa::host::DeviceNewOp op) {
  auto target = op.getTarget();
  auto npu = target.getNpu();
  auto pe_begin = target.getPeBegin();
  auto pe_end = target.getPeEnd();
  device_t device = device_runtime::device_new(npu, pe_begin, pe_end);
  engine.createValue(op->getResult(0), llvm::Any(device));

  return success();
}

LogicalResult executeOperation(ExecutionEngine &engine,
                               furiosa::host::DeviceExecuteOp op) {
  auto &hal_program =
      llvm::any_cast<hal_program_t &>(engine.getValue(op.getHalProgram()));
  auto &device = llvm::any_cast<device_t &>(engine.getValue(op.getDevice()));
  assert(hal_program.size() == 1);
  auto execution = device_runtime::device_execute(device, hal_program[0]);
  engine.createValue(op->getResult(0), llvm::Any(execution));

  return success();
}

LogicalResult executeOperation(ExecutionEngine &engine,
                               furiosa::host::DeviceExecutionWaitOp op) {
  auto &execution =
      llvm::any_cast<execution_t &>(engine.getValue(op.getExecution()));
  if (!device_runtime::device_execution_wait(execution)) {
    llvm::report_fatal_error(llvm::Twine("device execution wait failed"));
    return failure();
  }

  return success();
}

LogicalResult executeOperation(ExecutionEngine &engine, Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          .Case<func::ReturnOp>(
              [&](auto op) { return executeOperation(engine, op); })
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
              [&](auto op) { return executeOperation(engine, op); })
          .Default([&](Operation *) {
            return op.emitOpError("unable to find executor for op");
          });

  if (failed(status))
    return failure();

  return success();
}

LogicalResult executeMainFunction(ExecutionEngine engine,
                                  func::FuncOp function_op, void **args) {

  // Emit the body of the function.
  for (Block &block : function_op.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      if (failed(executeOperation(engine, op)))
        return failure();
    }
  }

  return success();
}

LogicalResult executeKernelFunction(ExecutionEngine &engine,
                                    func::FuncOp function_op,
                                    std::int64_t num_args,
                                    std::int64_t num_inputs, void **args) {
  // Create a main function that calls the kernel function.
  auto module = engine.getModule();
  auto context = module->getContext();
  auto builder = OpBuilder(context);
  builder.setInsertionPointToEnd(module.getBody());
  auto main_function = builder.create<func::FuncOp>(
      module->getLoc(), "main",
      builder.getFunctionType(mlir::TypeRange(), mlir::TypeRange()));
  auto entry_block = main_function.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);
  builder.create<func::ReturnOp>(module->getLoc());
  builder.setInsertionPointToStart(entry_block);

  // Create a call to the kernel function with the provided arguments.
  auto types = SmallVector<Type>();
  auto argument_types = function_op.getArgumentTypes();
  auto result_types = function_op.getResultTypes();
  types.append(argument_types.begin(), argument_types.end());
  types.append(result_types.begin(), result_types.end());
  SmallVector<Value> arg_values;
  for (auto index_value : llvm::enumerate(types)) {
    auto &arg_type = index_value.value();
    auto index = index_value.index();
    auto input_argument = args[index];
    auto tensor_type = llvm::dyn_cast_or_null<RankedTensorType>(arg_type);
    auto pointer = getTensorDataPointer(tensor_type, input_argument);
    auto arg_value = builder.create<furiosa::AllocOp>(
        module->getLoc(), tensor_type, IntegerAttr());
    arg_value->setAttr("data_ptr", builder.getI64IntegerAttr(pointer));
    if (index < static_cast<std::size_t>(num_inputs)) {
      arg_value->setAttr("argument", builder.getUnitAttr());
    } else {
      arg_value->setAttr("result", builder.getUnitAttr());
    }
    arg_values.push_back(arg_value);
  }
  auto call_op =
      builder.create<func::CallOp>(module->getLoc(), function_op, arg_values);
  call_op->setAttr("target", engine.getTarget());

  // Apply appropriate passes for converting to host dialect
  PassManager pm(context);
  pm.addPass(furiosa::createFuriosaDeallocationPass());
  pm.addPass(bufferization::createOptimizeAllocationLivenessPass());
  pm.addPass(furiosa::createFuriosaAllocateAddressPass());
  pm.addPass(furiosa::createConvertFuncToFuriosaHostPass());
  auto status = pm.run(main_function);
  if (failed(status)) {
    llvm::report_fatal_error(llvm::Twine("failed to run pass manager"));
    return failure();
  }

  status = executeMainFunction(engine, main_function, args);
  if (failed(status)) {
    llvm::report_fatal_error(llvm::Twine("failed to execute main function"));
    return failure();
  }

  return success();
}

LogicalResult executeFunction(ExecutionEngine &engine, StringRef function_name,
                              std::int64_t num_args, std::int64_t num_inputs,
                              void **args) {
  auto module = engine.getModule();
  auto function_op = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(module, function_name));
  if (!function_op || function_op.empty()) {
    llvm::report_fatal_error(llvm::Twine("entry point not found"));
    return failure();
  }
  auto num_arguments = function_op.getNumArguments();
  auto num_results = function_op.getNumResults();
  auto total = num_arguments + num_results;
  assert(num_args == total &&
         "number of arguments does not match the function signature");

  if (function_op->hasAttr("target")) {
    return executeKernelFunction(engine, function_op, num_args, num_inputs,
                                 args);
  } else {
    assert(num_arguments == 0 && num_inputs == 0 &&
           "number of arguments must be zero for non-target functions");
    return executeMainFunction(engine, function_op, args);
  }
}

} // namespace mlir::furiosa
