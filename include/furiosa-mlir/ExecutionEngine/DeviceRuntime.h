#pragma once

void launchKernel(llvm::StringRef kernel,
                  llvm::ArrayRef<std::uint32_t> argumentSizes,
                  llvm::ArrayRef<std::uint32_t> resultSizes);
