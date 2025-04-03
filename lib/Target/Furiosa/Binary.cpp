#include "mlir/Support/LLVM.h"

#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

#include "furiosa-mlir/Target/Furiosa/Binary.h"

using namespace mlir;

namespace mlir::furiosa {

LogicalResult writeFuriosaBinary(llvm::Twine filepath,
                                 FuriosaBinary furiosaBinary) {
  llvm::AppendingBinaryByteStream stream{};
  llvm::BinaryStreamWriter writer(stream);

  if (writer.writeInteger<std::uint32_t>(furiosaBinary.argumentSizes.size())) {
    return failure();
  }
  if (writer.writeArray(ArrayRef(furiosaBinary.argumentSizes))) {
    return failure();
  }
  if (writer.writeInteger<std::uint32_t>(furiosaBinary.resultSizes.size())) {
    return failure();
  }
  if (writer.writeArray(ArrayRef(furiosaBinary.resultSizes))) {
    return failure();
  }
  if (writer.writeInteger<std::uint32_t>(furiosaBinary.binBuffer.size())) {
    return failure();
  }
  if (writer.writeFixedString(furiosaBinary.binBuffer)) {
    return failure();
  }

  llvm::Expected<std::unique_ptr<llvm::FileOutputBuffer>> fileBuffer =
      llvm::FileOutputBuffer::create(filepath.str(), stream.getLength());
  if (!fileBuffer) {
    llvm::report_fatal_error(
        llvm::Twine(llvm::toString(fileBuffer.takeError())));
    return failure();
  }
  llvm::FileBufferByteStream fileStream(std::move(fileBuffer.get()),
                                        llvm::endianness::native);
  llvm::BinaryStreamWriter fileWriter(fileStream);
  if (fileWriter.writeStreamRef(stream)) {
    return failure();
  }
  if (fileStream.commit()) {
    return failure();
  }

  return success();
}

FailureOr<FuriosaBinary> readFuriosaBinary(llvm::Twine filepath) {
  auto status = llvm::MemoryBuffer::getFile(filepath);
  if (!status) {
    llvm::report_fatal_error(llvm::Twine("Failed to open file: ") +
                             status.getError().message());
  }
  llvm::BinaryByteStream stream(status->get()->getBuffer(),
                                llvm::endianness::native);
  llvm::BinaryStreamReader reader(stream);

  std::uint32_t numArguments, numResults;
  auto argumentSizes = ArrayRef<std::uint32_t>();
  auto resultSizes = ArrayRef<std::uint32_t>();
  auto binBuffer = StringRef();
  if (reader.readInteger(numArguments)) {
    return failure();
  }
  if (reader.readArray(argumentSizes, numArguments)) {
    return failure();
  }
  if (reader.readInteger(numResults)) {
    return failure();
  }
  if (reader.readArray(resultSizes, numResults)) {
    return failure();
  }
  std::uint32_t codeSize;
  if (reader.readInteger(codeSize)) {
    return failure();
  }
  if (reader.readFixedString(binBuffer, codeSize)) {
    return failure();
  }

  FuriosaBinary furiosaBinary = {SmallVector<std::uint32_t>(argumentSizes),
                                 SmallVector<std::uint32_t>(resultSizes),
                                 SmallString<MIN_BINARY_SIZE>(binBuffer)};
  return furiosaBinary;
}

} // namespace mlir::furiosa
