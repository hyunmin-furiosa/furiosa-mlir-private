#include "furiosa_torch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"

void launchKernel(llvm::StringRef kernel) {
  // generate pe program
  std::vector<furiosa_torch::PeProgram *> pe_programs;
  pe_programs.push_back(
      furiosa_torch::pe_program_load_inst(0x0, 0x0, kernel.size()));
  pe_programs.push_back(furiosa_torch::pe_program_launch(0x0, nullptr));
  auto pe_program =
      furiosa_torch::pe_program_seq(pe_programs.data(), pe_programs.size());

  // generate hal program
  std::vector<furiosa_torch::HalProgram *> hal_programs;
  hal_programs.push_back(furiosa_torch::hal_program_write_at(
      reinterpret_cast<const std::uint64_t>(kernel.data()), 0x0,
      kernel.size()));
  hal_programs.push_back(furiosa_torch::hal_program_execute(pe_program));
  auto hal_program =
      furiosa_torch::hal_program_seq(hal_programs.data(), hal_programs.size());

  // initialize device and execute hal program
  auto device = furiosa_torch::device_new(0, 0, 0);
  furiosa_torch::device_execute(device, hal_program);
}
