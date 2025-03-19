#ifndef FURIOSA_COMMANDS_H
#define FURIOSA_COMMANDS_H

namespace mlir::furiosa {

struct TensorUnitCommand {
  TensorUnitCommand() : value(0) {}
  TensorUnitCommand(std::uint32_t command) : value(command) {}
  union {
    std::uint32_t value;
    struct {
      std::uint32_t operand : 25;
      std::uint32_t opcode : 7;
    };
    struct {
      std::uint32_t : 1;
      std::uint32_t r3 : 6;
      std::uint32_t r2 : 6;
      std::uint32_t r1 : 6;
      std::uint32_t r0 : 6;
      std::uint32_t opcode : 7;
    } reg;
  };
};

struct ExecutionCommand {
  ExecutionCommand(std::uint32_t command) : value(command) {}
  union {
    std::uint32_t value;
    struct {
      std::uint32_t subunit_bitmap : 11;
      std::uint32_t : 5;
      std::uint32_t context_id : 1;
      std::uint32_t : 7;
      std::uint32_t target_context : 1;
      std::uint32_t opcode : 7;
    };
  };
};

struct WaitCommand {
  WaitCommand(std::uint32_t command) : value(command) {}
  union {
    std::uint32_t value;
    struct {
      std::uint32_t : 24;
      std::uint32_t target_context : 1;
      std::uint32_t opcode : 7;
    };
  };
};

} // namespace mlir::furiosa

#endif
