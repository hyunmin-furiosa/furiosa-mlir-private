#include "furiosa-mlir/ExecutionEngine/JitRunner.h"
#include "furiosa-mlir/ExecutionEngine/RenegadeRuntime.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/ParseUtilities.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir::furiosa {

/// This options struct prevents the need for global static initializers, and
/// is only initialized if the JITRunner is invoked.
struct Options {
  llvm::cl::opt<std::string> inputFilename{llvm::cl::Positional,
                                           llvm::cl::desc("<input file>"),
                                           llvm::cl::init("-")};
  llvm::cl::opt<std::string> mainFuncName{
      "e", llvm::cl::desc("The function to be called"),
      llvm::cl::value_desc("<function name>"), llvm::cl::init("main")};
  llvm::cl::opt<std::string> mainFuncType{
      "entry-point-result",
      llvm::cl::desc("Textual description of the function type to be called"),
      llvm::cl::value_desc("f32 | i32 | i64 | void"), llvm::cl::init("f32")};

  /// CLI variables for debugging.
  llvm::cl::opt<bool> hostSupportsJit{"host-supports-jit",
                                      llvm::cl::desc("Report host JIT support"),
                                      llvm::cl::Hidden};

  llvm::cl::opt<bool> noImplicitModule{
      "no-implicit-module",
      llvm::cl::desc(
          "Disable implicit addition of a top-level module op during parsing"),
      llvm::cl::init(false)};
};

static OwningOpRef<Operation *> parseMLIRInput(StringRef inputFilename,
                                               bool insertImplicitModule,
                                               MLIRContext *context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(file), SMLoc());
  OwningOpRef<Operation *> module =
      parseSourceFileForTool(sourceMgr, context, insertImplicitModule);
  if (!module)
    return nullptr;
  if (!module.get()->hasTrait<OpTrait::SymbolTable>()) {
    llvm::errs() << "Error: top-level op must be a symbol table.\n";
    return nullptr;
  }
  return module;
}

/// Entry point for all CPU runners. Expects the common argc/argv arguments for
/// standard C++ main functions.
int JitRunnerMain(int argc, char **argv, const DialectRegistry &registry,
                  JitRunnerConfig config) {
  llvm::ExitOnError exitOnErr;

  // Create the options struct containing the command line options for the
  // runner. This must come before the command line options are parsed.
  Options options;
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Furiosa-MLIR CPU execution driver\n");

  if (options.hostSupportsJit) {
    auto j = llvm::orc::LLJITBuilder().create();
    if (j)
      llvm::outs() << "true\n";
    else {
      llvm::outs() << "false\n";
      exitOnErr(j.takeError());
    }
    return 0;
  }

  MLIRContext context(registry);

  auto m = parseMLIRInput(options.inputFilename, !options.noImplicitModule,
                          &context);
  if (!m) {
    llvm::errs() << "could not parse the input IR\n";
    return 1;
  }

  if (failed(executeFunction(m.get(), options.mainFuncName.getValue(),
                             options.mainFuncType.getValue()))) {
    llvm::errs() << "could not execute the input IR\n";
    return 1;
  }

  return 0;
}

} // namespace mlir::furiosa
