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

  if (writer.writeObject(furiosaBinary.metadata)) {
    return failure();
  }
  if (writer.padToAlignment(llvm::Align::Of<address_size_t>().value())) {
    return failure();
  }
  for (auto tensor : furiosaBinary.tensors) {
    if (writer.writeInteger(tensor.address)) {
      return failure();
    }
    if (writer.writeInteger(tensor.size)) {
      return failure();
    }
    if (writer.writeArray(ArrayRef(tensor.data))) {
      return failure();
    }
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

  const FuriosaBinaryMetadata *metadataPtr = nullptr;
  if (reader.readObject(metadataPtr)) {
    return failure();
  }
  FuriosaBinaryMetadata metadata = *metadataPtr;
  if (reader.padToAlignment(llvm::Align::Of<address_size_t>().value())) {
    return failure();
  }

  llvm::SmallVector<tensor_t> tensors;
  auto numTensors = metadata.numArguments + metadata.numResults;
  for (auto i = 0; i < numTensors; ++i) {
    tensor_t tensor;
    if (reader.readInteger(tensor.address)) {
      return failure();
    }
    if (reader.readInteger(tensor.size)) {
      return failure();
    }
    ArrayRef<uint8_t> data;
    if (reader.readArray(data, tensor.size)) {
      return failure();
    }
    tensor.data = SmallVector<std::uint8_t>(data);
    tensors.push_back(tensor);
  }
  auto binary = StringRef();
  if (reader.readFixedString(binary, metadata.binarySize)) {
    return failure();
  }

  FuriosaBinary furiosaBinary = {metadata, SmallVector<tensor_t>(tensors),
                                 SmallString<MIN_BINARY_SIZE>(binary)};
  return furiosaBinary;
}

} // namespace mlir::furiosa
