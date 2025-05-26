#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace furiosa {

#define GEN_PASS_DECL
#include "furiosa-mlir/Dialect/Func/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "furiosa-mlir/Dialect/Func/Transforms/Passes.h.inc"

} // namespace furiosa
} // namespace mlir
