#include "furiosa_torch.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"

#include "furiosa-mlir/ExecutionEngine/DeviceRuntime.h"

void launchKernel(mlir::furiosa::FuriosaBinary furiosaBinary) {
  // generate pe program
  std::vector<furiosa_torch::PeProgram *> pe_programs;
  pe_programs.push_back(furiosa_torch::pe_program_load_inst(
      0x0, 0x0, furiosaBinary.binary.size()));
  pe_programs.push_back(furiosa_torch::pe_program_launch(0x0, nullptr));
  auto pe_program =
      furiosa_torch::pe_program_seq(pe_programs.data(), pe_programs.size());

  // generate hal program for kernel
  std::vector<furiosa_torch::HalProgram *> hal_programs;
  hal_programs.push_back(furiosa_torch::hal_program_write_at(
      reinterpret_cast<const std::uint64_t>(furiosaBinary.binary.data()), 0x0,
      furiosaBinary.binary.size()));

  // write arguments
  std::vector<std::vector<std::uint8_t>> argumentsBuffer;
  argumentsBuffer.reserve(furiosaBinary.arguments.size());
  for (auto [address, size] : furiosaBinary.arguments) {
    std::vector<std::uint8_t> argument(size, 1);
    argumentsBuffer.push_back(argument);
    hal_programs.push_back(furiosa_torch::hal_program_write_at(
        reinterpret_cast<const std::uint64_t>(argumentsBuffer.back().data()),
        address, argument.size()));
  }

  hal_programs.push_back(furiosa_torch::hal_program_execute(pe_program));

  // read results
  std::vector<std::vector<std::uint8_t>> resultsBuffer;
  resultsBuffer.reserve(furiosaBinary.results.size());
  for (auto [address, size] : furiosaBinary.results) {
    std::vector<std::uint8_t> result(size, 0);
    resultsBuffer.push_back(result);
    hal_programs.push_back(furiosa_torch::hal_program_read_at(
        address,
        reinterpret_cast<const std::uint64_t>(resultsBuffer.back().data()),
        result.size()));
  }

  // initialize device and execute hal program
  auto hal_program =
      furiosa_torch::hal_program_seq(hal_programs.data(), hal_programs.size());
  auto device = furiosa_torch::device_new(0, 0, 0);
  furiosa_torch::device_execute(device, hal_program);
}
