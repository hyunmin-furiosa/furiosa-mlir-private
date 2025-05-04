#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "furiosa-mlir/Dialect/Host/IR/HostOps.h"
#include "furiosa-mlir/ExecutionEngine/DeviceRuntime.h"
#include "furiosa-mlir/Target/Furiosa/FuriosaToBinary.h"

void launchKernel(mlir::furiosa::FuriosaBinary furiosaBinary) {
  // generate pe program
  std::vector<furiosa_torch::PeProgram *> pe_programs;
  pe_programs.push_back(furiosa_torch::pe_program_load_inst(
      furiosaBinary.metadata.binaryAddress, 0x0, furiosaBinary.binary.size()));
  pe_programs.push_back(furiosa_torch::pe_program_launch(0x0, nullptr));
  auto pe_program =
      furiosa_torch::pe_program_seq(pe_programs.data(), pe_programs.size());

  // generate hal program for kernel
  std::vector<furiosa_torch::HalProgram *> hal_programs;
  hal_programs.push_back(furiosa_torch::hal_program_write_at(
      reinterpret_cast<const std::uint64_t>(furiosaBinary.binary.data()),
      furiosaBinary.metadata.binaryAddress, furiosaBinary.binary.size()));

  // write arguments
  for (auto i = 0u; i < furiosaBinary.metadata.numArguments; i++) {
    auto &[address, size, data] = furiosaBinary.tensors[i];
    hal_programs.push_back(furiosa_torch::hal_program_write_at(
        reinterpret_cast<const std::uint64_t>(data.data()), address, size));
  }

  hal_programs.push_back(furiosa_torch::hal_program_execute(pe_program));

  // read results
  llvm::SmallVector<llvm::SmallVector<std::uint8_t>> resultsBuffer;
  resultsBuffer.reserve(furiosaBinary.metadata.numResults);
  for (auto i = 0u; i < furiosaBinary.metadata.numResults; i++) {
    auto &[address, size, data] =
        furiosaBinary.tensors[furiosaBinary.metadata.numArguments + i];
    resultsBuffer.push_back(llvm::SmallVector<std::uint8_t>(size, 0));
    hal_programs.push_back(furiosa_torch::hal_program_read_at(
        address,
        reinterpret_cast<const std::uint64_t>(resultsBuffer.back().data()),
        size));
  }

  // initialize device and execute hal program
  auto hal_program =
      furiosa_torch::hal_program_seq(hal_programs.data(), hal_programs.size());
  auto device = furiosa_torch::device_new(furiosaBinary.metadata.npu,
                                          furiosaBinary.metadata.peBegin,
                                          furiosaBinary.metadata.peEnd);
  furiosa_torch::device_execute(device, hal_program);

  for (auto i = 0u; i < furiosaBinary.metadata.numResults; i++) {
    auto &[address, size, expected_data] =
        furiosaBinary.tensors[furiosaBinary.metadata.numArguments + i];
    auto &actual_data = resultsBuffer[i];
    if (expected_data != actual_data) {
      llvm::dbgs() << "Results do not match!\n";
      llvm::dbgs() << "Expected: ";
      for (auto byte : expected_data) {
        llvm::dbgs() << llvm::format_hex(byte, 2) << " ";
      }
      llvm::dbgs() << "\n";
      llvm::dbgs() << "Actual: ";
      for (auto byte : actual_data) {
        llvm::dbgs() << llvm::format_hex(byte, 2) << " ";
      }
      llvm::dbgs() << "\n";
    } else {
      llvm::dbgs() << "Results match!\n";
    }
  }
}

using namespace mlir;

namespace mlir::furiosa {

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::AllocOp op) {
  auto size = op.getSize();
  auto data = op.getData();
  byte_array_t data_buffer;
  for (auto d : *data) {
    data_buffer.push_back(dyn_cast_or_null<IntegerAttr>(d).getInt());
  }
  auto input_data_size = data_buffer.size();
  data_buffer.reserve(CEIL(size, input_data_size));
  for (auto i = 0u; i < CEIL(size, input_data_size); i += input_data_size) {
    std::copy_n(data_buffer.begin(), input_data_size,
                std::back_inserter(data_buffer));
  }
  data_buffer.resize(size);
  data_buffer.resize(CEIL(data_buffer.size(), 256));
  context.createValue(op->getResult(0),
                      std::make_any<byte_array_t>(data_buffer));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::FuncAllocOp op) {
  auto name = op.getFunction();
  auto function = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(context.module, name));
  if (!function || function.empty()) {
    llvm::report_fatal_error(llvm::Twine("entry point not found"));
    return failure();
  }
  auto binary = translateKernelToBinary(function);
  if (failed(binary)) {
    llvm::report_fatal_error(llvm::Twine("failed to translate kernel"));
    return failure();
  }
  byte_array_t data_buffer;
  for (auto d : *binary) {
    data_buffer.push_back(reinterpret_cast<std::uint8_t &>(d));
  }
  data_buffer.resize(CEIL(data_buffer.size(), 256));
  context.createValue(op->getResult(0),
                      std::make_any<byte_array_t>(data_buffer));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::PeProgramLoadInstOp op) {
  auto dramAddress = op.getDramAddress();
  auto spmAddress = op.getSpmAddress();
  auto buffer = std::any_cast<byte_array_t>(context.getValue(op.getBinary()));
  pe_program_t programs;
  programs.push_back(furiosa_torch::pe_program_load_inst(
      dramAddress, spmAddress, buffer.size()));
  context.createValue(op->getResult(0), std::make_any<pe_program_t>(programs));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::PeProgramLaunchOp op) {
  auto spmAddress = op.getSpmAddress();
  pe_program_t programs;
  programs.push_back(furiosa_torch::pe_program_launch(spmAddress, nullptr));
  context.createValue(op->getResult(0), std::make_any<pe_program_t>(programs));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::PeProgramSeqOp op) {
  auto pePrograms = op.getPePrograms();
  pe_program_t programList;
  for (auto peProgram : pePrograms) {
    auto program = std::any_cast<pe_program_t>(context.getValue(peProgram));
    programList.insert(programList.end(), program.begin(), program.end());
  }
  pe_program_t mergedPrograms;
  mergedPrograms.push_back(
      furiosa_torch::pe_program_seq(programList.data(), programList.size()));
  context.createValue(op->getResult(0),
                      std::make_any<pe_program_t>(mergedPrograms));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::HalProgramWriteAtOp op) {
  auto dramAddress = op.getDramAddress();
  auto &buffer =
      std::any_cast<byte_array_t &>(context.getValue(op.getBuffer()));
  hal_program_t programs;
  programs.push_back(furiosa_torch::hal_program_write_at(
      reinterpret_cast<std::uint64_t>(buffer.data()), dramAddress,
      buffer.size()));
  context.createValue(op->getResult(0), std::make_any<hal_program_t>(programs));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::HalProgramReadAtOp op) {
  auto dramAddress = op.getDramAddress();
  auto &buffer =
      std::any_cast<byte_array_t &>(context.getValue(op.getBuffer()));
  hal_program_t programs;
  programs.push_back(furiosa_torch::hal_program_read_at(
      dramAddress, reinterpret_cast<std::uint64_t>(buffer.data()),
      buffer.size()));
  context.createValue(op->getResult(0), std::make_any<hal_program_t>(programs));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::HalProgramExecuteOp op) {
  auto peProgram =
      std::any_cast<pe_program_t>(context.getValue(op.getPeProgram()));
  hal_program_t programs;
  assert(peProgram.size() == 1);
  programs.push_back(furiosa_torch::hal_program_execute(peProgram[0]));
  context.createValue(op->getResult(0), std::make_any<hal_program_t>(programs));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::HalProgramSeqOp op) {
  auto halPrograms = op.getHalPrograms();
  hal_program_t programList;
  for (auto halProgram : halPrograms) {
    auto program = std::any_cast<hal_program_t>(context.getValue(halProgram));
    programList.insert(programList.end(), program.begin(), program.end());
  }
  hal_program_t mergedPrograms;
  mergedPrograms.push_back(
      furiosa_torch::hal_program_seq(programList.data(), programList.size()));
  context.createValue(op->getResult(0),
                      std::make_any<hal_program_t>(mergedPrograms));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::DeviceNewOp op) {
  auto target = op.getTarget();
  auto npu = target.getNpu();
  auto peBegin = target.getPeBegin();
  auto peEnd = target.getPeEnd();
  device_t device = furiosa_torch::device_new(npu, peBegin, peEnd);
  context.createValue(op->getResult(0), std::make_any<device_t>(device));

  return success();
}

LogicalResult executeOperation(ExecutionContext &context,
                               furiosa::host::DeviceExecuteOp op) {
  auto halProgram =
      std::any_cast<hal_program_t>(context.getValue(op.getHalProgram()));
  auto device = std::any_cast<device_t>(context.getValue(op.getDevice()));
  assert(halProgram.size() == 1);
  furiosa_torch::device_execute(device, halProgram[0]);

  return success();
}

LogicalResult executeOperation(ExecutionContext &context, Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          .Case<func::ReturnOp>([&](auto op) { return success(); })
          .Case<furiosa::host::AllocOp, furiosa::host::FuncAllocOp,
                furiosa::host::PeProgramLoadInstOp,
                furiosa::host::PeProgramLaunchOp, furiosa::host::PeProgramSeqOp,
                furiosa::host::HalProgramWriteAtOp,
                furiosa::host::HalProgramReadAtOp,
                furiosa::host::HalProgramExecuteOp,
                furiosa::host::HalProgramSeqOp, furiosa::host::DeviceNewOp,
                furiosa::host::DeviceExecuteOp>(
              [&](auto op) { return executeOperation(context, op); })
          .Default([&](Operation *) {
            return op.emitOpError("unable to find executor for op");
          });

  if (failed(status))
    return failure();

  return success();
}

LogicalResult executeFunction(Operation *module, StringRef entryPoint,
                              StringRef entryPointType) {
  ExecutionContext context;
  context.module = module;

  auto mainFunction = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(module, entryPoint));
  if (!mainFunction || mainFunction.empty()) {
    llvm::report_fatal_error(llvm::Twine("entry point not found"));
    return failure();
  }

  // Emit the body of the function.
  for (Block &block : mainFunction.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      if (failed(executeOperation(context, op)))
        return failure();
    }
  }

  return success();
}

} // namespace mlir::furiosa
