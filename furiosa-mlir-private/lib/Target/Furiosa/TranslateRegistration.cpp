#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaDialect.h"
#include "furiosa-mlir/Target/Furiosa/FuriosaToBinary.h"

using namespace mlir;

namespace mlir::furiosa {

void registerFuriosaToBinary() {
  TranslateFromMLIRRegistration reg(
      "furiosa-to-binary", "translate furiosa to binary",
      [](Operation *op, llvm::raw_ostream &os) -> LogicalResult {
        return translateFuriosaToBinary(op, os);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<mlir::func::FuncDialect,
                        mlir::tensor::TensorDialect,
                        mlir::tosa::TosaDialect,
                        mlir::furiosa::FuriosaDialect>();
        // clang-format on
      });
}

} // namespace mlir::furiosa
