#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  assert(argc == 2);
  llvm::StringRef filename = argv[1];
  // Read compiled binary

  auto status = llvm::MemoryBuffer::getFile(filename);
  if (!status) {
    llvm::report_fatal_error(llvm::Twine("Failed to open file: ") +
                             status.getError().message());
  }
  auto buffer = status->get()->getBuffer();
  llvm::outs() << buffer;

  return 0;
}
