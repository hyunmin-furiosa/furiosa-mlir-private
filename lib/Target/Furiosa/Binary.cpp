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

  if (writer.writeInteger<std::uint32_t>(furiosaBinary.arguments.size())) {
    return failure();
  }
  if (writer.padToAlignment(llvm::Align::Of<address_size_t>().value())) {
    return failure();
  }
  if (writer.writeArray(ArrayRef(furiosaBinary.arguments))) {
    return failure();
  }
  if (writer.writeInteger<std::uint32_t>(furiosaBinary.results.size())) {
    return failure();
  }
  if (writer.padToAlignment(llvm::Align::Of<address_size_t>().value())) {
    return failure();
  }
  if (writer.writeArray(ArrayRef(furiosaBinary.results))) {
    return failure();
  }
  if (writer.writeInteger<std::uint32_t>(furiosaBinary.binary.size())) {
    return failure();
  }
  if (writer.writeFixedString(furiosaBinary.binary)) {
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
  ArrayRef<address_size_t> arguments;
  ArrayRef<address_size_t> results;
  auto binary = StringRef();
  if (reader.readInteger(numArguments)) {
    return failure();
  }
  if (reader.padToAlignment(llvm::Align::Of<address_size_t>().value())) {
    return failure();
  }
  if (reader.readArray(arguments, numArguments)) {
    return failure();
  }
  if (reader.readInteger(numResults)) {
    return failure();
  }
  if (reader.padToAlignment(llvm::Align::Of<address_size_t>().value())) {
    return failure();
  }
  if (reader.readArray(results, numResults)) {
    return failure();
  }
  std::uint32_t codeSize;
  if (reader.readInteger(codeSize)) {
    return failure();
  }
  if (reader.readFixedString(binary, codeSize)) {
    return failure();
  }

  FuriosaBinary furiosaBinary = {SmallVector<address_size_t>(arguments),
                                 SmallVector<address_size_t>(results),
                                 SmallString<MIN_BINARY_SIZE>(binary)};
  return furiosaBinary;
}

} // namespace mlir::furiosa
