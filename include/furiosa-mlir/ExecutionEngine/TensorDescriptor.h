#pragma once

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::furiosa {

template <typename T, int N> struct TensorDescriptor {
  std::int64_t allocated;
  T *aligned;
  std::int64_t offset;
  std::int64_t shape[N];

  TensorDescriptor(void *ptr) {
    auto desc = reinterpret_cast<TensorDescriptor<T, N> *>(ptr);
    *this = *desc;
  }

  llvm::SmallVector<std::int64_t> getShape() const {
    llvm::SmallVector<std::int64_t> shape_vector(shape, shape + N);
    return shape_vector;
  }

  std::int64_t getSize() const {
    auto size = sizeof(T);
    for (auto dim : getShape()) {
      size *= dim;
    }
    return size;
  }

  std::int64_t getPointer() { return allocated; }
};

template <typename T, int N> struct GenerateTensorDescriptor {
  using current = TensorDescriptor<T, N>;
  using next = GenerateTensorDescriptor<T, N - 1>;
};

template <typename T> struct GenerateTensorDescriptor<T, 0> {
  using current = TensorDescriptor<T, 0>;
};

template struct GenerateTensorDescriptor<std::int8_t, 8>;

std::int64_t getTensorDataPointer(RankedTensorType type, void *ptr);

} // namespace mlir::furiosa
