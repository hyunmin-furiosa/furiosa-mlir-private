#ifndef FURIOSA_DIALECT_UTILS_H
#define FURIOSA_DIALECT_UTILS_H

#include "furiosa-mlir/Dialect/Furiosa/IR/Commands.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"

namespace mlir::furiosa {

FailureOr<std::uint32_t> getOpcode(Operation &op) {
  return llvm::TypeSwitch<Operation *, FailureOr<std::uint32_t>>(&op)
      .Case<ExecutionOp>([&](auto op) { return 0x10; })
      .Case<WaitOp>([&](auto op) { return 0x11; })
      .Default([&](Operation *) {
        return op.emitOpError(
            "unable to interpret as furiosa dialect operator");
      });
}

FailureOr<std::uint32_t> getCommand(Operation &op) {
  TensorUnitCommand genericCommand = TensorUnitCommand();
  genericCommand.opcode = *getOpcode(op);
  return llvm::TypeSwitch<Operation *, FailureOr<std::uint32_t>>(&op)
      .Case<ExecutionOp>([&](auto op) {
        ExecutionCommand command = ExecutionCommand(genericCommand.value);
        command.target_context = op.getTargetContext();
        command.context_id = op.getContextId();
        command.subunit_bitmap = op.getSubunitBitmap();
        return command.value;
      })
      .Case<WaitOp>([&](auto op) {
        WaitCommand command = WaitCommand(genericCommand.value);
        command.target_context = op.getTargetContext();
        return command.value;
      })
      .Default([&](Operation *) {
        return op.emitOpError(
            "unable to interpret as furiosa dialect operator");
      });
}

} // namespace mlir::furiosa

#endif
