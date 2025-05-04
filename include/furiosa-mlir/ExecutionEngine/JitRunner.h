#pragma once

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/IR/LLVMContext.h"

namespace mlir::furiosa {

/// JitRunner command line options used by JitRunnerConfig methods
struct JitRunnerOptions {
  /// The name of the main function
  llvm::StringRef mainFuncName;
  /// The type of the main function (as string, from cmd-line)
  llvm::StringRef mainFuncType;
};

/// Configuration to override functionality of the JitRunner
struct JitRunnerConfig {
  /// MLIR transformer applied after parsing the input into MLIR IR and before
  /// passing the MLIR IR to the ExecutionEngine.
  llvm::function_ref<llvm::LogicalResult(mlir::Operation *,
                                         JitRunnerOptions &options)>
      mlirTransformer = nullptr;

  /// A custom function that is passed to ExecutionEngine. It processes MLIR and
  /// creates an LLVM IR module.
  llvm::function_ref<std::unique_ptr<llvm::Module>(Operation *,
                                                   llvm::LLVMContext &)>
      llvmModuleBuilder = nullptr;

  /// A callback to register symbols with ExecutionEngine at runtime.
  llvm::function_ref<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)>
      runtimesymbolMap = nullptr;
};

/// Entry point for all CPU runners. Expects the common argc/argv arguments for
/// standard C++ main functions. The supplied dialect registry is expected to
/// contain any registers that appear in the input IR, they will be loaded
/// on-demand by the parser.
int JitRunnerMain(int argc, char **argv, const DialectRegistry &registry,
                  JitRunnerConfig config = {});

} // namespace mlir::furiosa
