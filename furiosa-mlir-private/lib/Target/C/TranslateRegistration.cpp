#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaDialect.h"
#include "furiosa-mlir/InitAll.h"
#include "furiosa-mlir/Target/C/FuriosaTaskToC.h"

using namespace mlir;

namespace mlir::furiosa {

void registerFuriosaToBinary() {
  TranslateFromMLIRRegistration reg(
      "furiosa-to-arm-c", "translate furiosa to arm c",
      [](Operation *op, llvm::raw_ostream &os) -> LogicalResult {
        return translateFuriosaToArmC(op, os);
      },
      [](DialectRegistry &registry) {
        mlir::furiosa::registerAllDialects(registry);
      });
}

} // namespace mlir::furiosa
