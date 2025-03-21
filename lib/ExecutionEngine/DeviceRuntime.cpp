#include "llvm/Support/MemoryBuffer.h"

extern "C" std::uint32_t get_or_init_single(std::uint8_t chip_id,
                                            std::uint8_t pe_id,
                                            std::uint8_t pe_id_end,
                                            const std::uint8_t *kernel,
                                            std::uint32_t size);

void launchKernel(llvm::StringRef kernel) {
  get_or_init_single(0, 0, 0, reinterpret_cast<const uint8_t *>(kernel.data()),
                     kernel.size());
}
