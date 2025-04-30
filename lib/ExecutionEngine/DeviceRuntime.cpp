#include "furiosa_torch.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaHostOps.h"
#include "furiosa-mlir/ExecutionEngine/DeviceRuntime.h"

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

LogicalResult executeOperation(furiosa::host::DeviceExecuteOp op) {
  op.dump();
  return success();
}

LogicalResult executeOperation(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          .Case<func::ReturnOp>([&](auto op) { return success(); })
          .Case<furiosa::host::DeviceExecuteOp>(
              [&](auto op) { return executeOperation(op); })
          .Default([&](Operation *) {
            return op.emitOpError("unable to find executor for op");
          });

  if (failed(status))
    return failure();

  return success();
}

LogicalResult executeFunction(Operation *module, StringRef entryPoint,
                              StringRef entryPointType) {
  auto mainFunction = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(module, entryPoint));
  if (!mainFunction || mainFunction.empty()) {
    llvm::report_fatal_error(llvm::Twine("entry point not found"));
    return failure();
  }

  // Emit the body of the function.
  for (Block &block : mainFunction.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      if (failed(executeOperation(op)))
        return failure();
    }
  }

  return success();
}

} // namespace mlir::furiosa
