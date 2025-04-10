#include "furiosa_torch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"

void launchKernel(llvm::StringRef kernel,
                  llvm::ArrayRef<std::uint32_t> argumentSizes,
                  llvm::ArrayRef<std::uint32_t> resultSizes) {
  // generate pe program
  std::vector<furiosa_torch::PeProgram *> pe_programs;
  pe_programs.push_back(
      furiosa_torch::pe_program_load_inst(0x0, 0x0, kernel.size()));
  pe_programs.push_back(furiosa_torch::pe_program_launch(0x0, nullptr));
  auto pe_program =
      furiosa_torch::pe_program_seq(pe_programs.data(), pe_programs.size());

  // generate hal program for kernel
  std::vector<furiosa_torch::HalProgram *> hal_programs;
  hal_programs.push_back(furiosa_torch::hal_program_write_at(
      reinterpret_cast<const std::uint64_t>(kernel.data()), 0x0,
      kernel.size()));

  // write arguments
  std::uint32_t current_address = (kernel.size() + 255) / 256 * 256;
  std::vector<std::vector<std::uint8_t>> arguments;
  for (auto size : argumentSizes) {
    std::vector<std::uint8_t> argument(size, 1);
    arguments.push_back(argument);
  }
  for (auto &argument : arguments) {
    hal_programs.push_back(furiosa_torch::hal_program_write_at(
        reinterpret_cast<const std::uint64_t>(argument.data()), current_address,
        argument.size()));
    current_address = (current_address + argument.size() + 255) / 256 * 256;
  }

  hal_programs.push_back(furiosa_torch::hal_program_execute(pe_program));

  // read results
  std::vector<std::vector<std::uint8_t>> results;
  for (auto size : resultSizes) {
    std::vector<std::uint8_t> result(size, 0);
    results.push_back(result);
  }
  for (auto &result : results) {
    hal_programs.push_back(furiosa_torch::hal_program_read_at(
        current_address, reinterpret_cast<const std::uint64_t>(result.data()),
        result.size()));
    current_address = (current_address + result.size() + 255) / 256 * 256;
  }

  // initialize device and execute hal program
  auto hal_program =
      furiosa_torch::hal_program_seq(hal_programs.data(), hal_programs.size());
  auto device = furiosa_torch::device_new(0, 0, 0);
  furiosa_torch::device_execute(device, hal_program);
}
