#pragma once

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"

#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::furiosa {

LogicalResult InitializeAArch64() {
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetMC();
  LLVMInitializeAArch64AsmPrinter();
  return success();
}

FailureOr<std::string> convertArmCToObject(llvm::Twine filepath) {
  if (failed(InitializeAArch64())) {
    llvm::report_fatal_error("Failed to initialize AArch64 target");
    return failure();
  }

  // Compile the C code
  llvm::Twine filepath_out = filepath + ".o";
  auto invocation = std::make_shared<clang::CompilerInvocation>();
  const char *args[] = {""}; // -v to make verbose
  clang::CompilerInstance compiler;
  compiler.createDiagnostics(*llvm::vfs::getRealFileSystem());
  clang::CompilerInvocation::CreateFromArgs(*invocation, args,
                                            compiler.getDiagnostics());
  invocation->getTargetOpts().Triple =
      "aarch64-unknown-none-elf"; // -triple aarch64-unknown-none-elf
  invocation->getFrontendOpts().Inputs.clear(); // remove default input '-'
  invocation->getFrontendOpts().Inputs.push_back(
      clang::FrontendInputFile(filepath.str(), clang::Language::C)); // filepath
  invocation->getFrontendOpts().OutputFile =
      filepath_out.str(); // -o filepath_out
  invocation->getHeaderSearchOpts().UseStandardSystemIncludes =
      false; // -nostdsysteminc
  invocation->getHeaderSearchOpts().AddPath(CLANG_CROSS_COMPILE_INCLUDE_DIRS,
                                            clang::frontend::Angled, false,
                                            true);    // -I
  invocation->getCodeGenOpts().OptimizationLevel = 3; // -O3
  compiler.setInvocation(std::move(invocation));
  std::unique_ptr<clang::FrontendAction> action =
      std::make_unique<clang::EmitObjAction>();
  compiler.ExecuteAction(*action);

  // Check if object file exists
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOrErr =
      llvm::MemoryBuffer::getFile("kernel.c.o");
  if (!bufferOrErr) {
    llvm::report_fatal_error(llvm::Twine("Failed to open file: ") +
                             bufferOrErr.getError().message());
  }

  return filepath_out.str();
}

FailureOr<std::string> convertObjectToBinary(llvm::Twine filepath) {
  auto status = llvm::MemoryBuffer::getFile(filepath);
  if (!status) {
    llvm::report_fatal_error(llvm::Twine("Failed to open file: ") +
                             status.getError().message());
    return failure();
  }
  llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> objOrErr =
      llvm::object::ObjectFile::createELFObjectFile(*status->get(), false);
  if (!objOrErr) {
    return failure();
  }
  auto *obj =
      llvm::dyn_cast<llvm::object::ELFObjectFileBase>(objOrErr.get().get());
  if (!obj) {
    return failure();
  }
  std::string binBuffer;
  for (llvm::object::ELFSectionRef section : obj->sections()) {
    if ((section.getFlags() & llvm::ELF::SHF_ALLOC) &&
        section.getType() != llvm::ELF::SHT_NOBITS && section.getSize() > 0) {
      llvm::Expected<llvm::StringRef> contents = section.getContents();
      if (!contents) {
        return failure();
      }
      binBuffer += contents->str();
    }
  }

  return binBuffer;
}

} // namespace mlir::furiosa
