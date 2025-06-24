#include "furiosa-mlir/ExecutionEngine/TensorDescriptor.h"

namespace mlir::furiosa {

std::int64_t getTensorDataPointer(RankedTensorType type, void *ptr) {
  auto element_type = type.getElementType();
  SmallVector<std::int64_t> input_shape;
  std::int64_t pointer;
  if (auto integer_type = llvm::dyn_cast_or_null<IntegerType>(element_type)) {
    if (integer_type.getWidth() == 8) {
      if (type.getRank() == 1) {
        auto descriptor = TensorDescriptor<std::int8_t, 1>(ptr);
        input_shape = descriptor.getShape();
        pointer = descriptor.getPointer();
      } else if (type.getRank() == 2) {
        auto descriptor = TensorDescriptor<std::int8_t, 2>(ptr);
        input_shape = descriptor.getShape();
        pointer = descriptor.getPointer();
      } else if (type.getRank() == 3) {
        auto descriptor = TensorDescriptor<std::int8_t, 3>(ptr);
        input_shape = descriptor.getShape();
        pointer = descriptor.getPointer();
      } else if (type.getRank() == 4) {
        auto descriptor = TensorDescriptor<std::int8_t, 4>(ptr);
        input_shape = descriptor.getShape();
        pointer = descriptor.getPointer();
      } else if (type.getRank() == 5) {
        auto descriptor = TensorDescriptor<std::int8_t, 5>(ptr);
        input_shape = descriptor.getShape();
        pointer = descriptor.getPointer();
      } else if (type.getRank() == 6) {
        auto descriptor = TensorDescriptor<std::int8_t, 6>(ptr);
        input_shape = descriptor.getShape();
        pointer = descriptor.getPointer();
      } else if (type.getRank() == 7) {
        auto descriptor = TensorDescriptor<std::int8_t, 7>(ptr);
        input_shape = descriptor.getShape();
        pointer = descriptor.getPointer();
      } else if (type.getRank() == 8) {
        auto descriptor = TensorDescriptor<std::int8_t, 8>(ptr);
        input_shape = descriptor.getShape();
        pointer = descriptor.getPointer();
      }
    }
  }
  assert(type.getShape() == ArrayRef(input_shape) &&
         "input shape does not match the tensor type");

  return pointer;
}

} // namespace mlir::furiosa
