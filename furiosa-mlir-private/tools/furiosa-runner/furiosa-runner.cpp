#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "furiosa-mlir/ExecutionEngine/DeviceRuntime.h"

int main(int argc, char **argv) {
  assert(argc == 2);
  llvm::StringRef filename = argv[1];

  auto status = llvm::MemoryBuffer::getFile(filename);
  if (!status) {
    llvm::report_fatal_error(llvm::Twine("Failed to open file: ") +
                             status.getError().message());
  }
  auto buffer = status->get()->getBuffer();
  launchKernel(buffer);

  return 0;
}
