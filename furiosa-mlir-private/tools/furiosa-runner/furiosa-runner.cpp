#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"

#include "furiosa-mlir/ExecutionEngine/DeviceRuntime.h"

int main(int argc, char **argv) {
  assert(argc == 2);
  llvm::StringRef filename = argv[1];

  auto status = llvm::MemoryBuffer::getFile(filename);
  if (!status) {
    llvm::report_fatal_error(llvm::Twine("Failed to open file: ") +
                             status.getError().message());
  }
  llvm::BinaryByteStream stream(status->get()->getBuffer(),
                                llvm::endianness::native);
  llvm::BinaryStreamReader reader(stream);

  std::uint32_t numArguments, numResults;
  llvm::ArrayRef<std::uint32_t> argumentSizes;
  llvm::ArrayRef<std::uint32_t> resultSizes;
  if (reader.readInteger(numArguments)) {
    return -1;
  }
  if (reader.readArray(argumentSizes, numArguments)) {
    return -1;
  }
  if (reader.readInteger(numResults)) {
    return -1;
  }
  if (reader.readArray(resultSizes, numResults)) {
    return -1;
  }
  std::uint32_t codeSize;
  llvm::StringRef binBuffer;
  if (reader.readInteger(codeSize)) {
    return -1;
  }
  if (reader.readFixedString(binBuffer, codeSize)) {
    return -1;
  }

  launchKernel(binBuffer, argumentSizes, resultSizes);

  return 0;
}
