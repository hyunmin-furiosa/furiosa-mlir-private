#pragma once

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"

#include "lld/Common/Driver.h"

#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

// for linkObject
LLD_HAS_DRIVER(elf)

namespace mlir::furiosa {

LogicalResult InitializeAArch64() {
  LLVMInitializeAArch64AsmParser();
  LLVMInitializeAArch64AsmPrinter();
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetMC();
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
  invocation->getHeaderSearchOpts().AddPath(
      StringRef(CLANG_CROSS_COMPILE_INCLUDE), clang::frontend::Angled, false,
      true); // -I
  invocation->getHeaderSearchOpts().AddPath(
      StringRef(CLANG_CROSS_COMPILE_LIBPE), clang::frontend::Angled, false,
      true);                                          // -I
  invocation->getCodeGenOpts().OptimizationLevel = 3; // -O3
  invocation->getCodeGenOpts().setInlining(
      clang::CodeGenOptions::NormalInlining);
  invocation->getCodeGenOpts().RelocationModel = llvm::Reloc::PIC_; // -fPIC
  invocation->getLangOpts().PICLevel = 2;     // -pic-level 2
  invocation->getLangOpts().PIE = 1;          // -pic-is-pie
  invocation->getLangOpts().NoBuiltin = 1;    // -fno-builtin
  invocation->getLangOpts().Freestanding = 1; // -ffreestanding

  // Check compile command
  auto command = invocation->getCC1CommandLine();
  llvm::dbgs() << "clang ";
  for (auto arg : command) {
    llvm::dbgs() << arg << " ";
  }
  llvm::dbgs() << "\n";

  // Compile
  compiler.setInvocation(std::move(invocation));
  std::unique_ptr<clang::FrontendAction> action =
      std::make_unique<clang::EmitObjAction>();
  compiler.ExecuteAction(*action);

  // Check if object file exists
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOrErr =
      llvm::MemoryBuffer::getFile(filepath_out);
  if (!bufferOrErr) {
    llvm::report_fatal_error(llvm::Twine("Failed to open file: ") +
                             bufferOrErr.getError().message());
  }

  return filepath_out.str();
}

FailureOr<std::string> linkObject(llvm::Twine filepath) {
  llvm::Twine filepath_out = filepath + ".o";

  auto linker = std::string("-T") + CLANG_CROSS_COMPILE_LIBPE + "/linker.ld";
  auto input = filepath.str();
  auto output = std::string("-o") + filepath_out.str();
  const char *args[] = {"ld.lld", linker.c_str(), input.c_str(), output.c_str(),
                        "-e0"}; // -v to make verbose
  lld::lldMain(args, llvm::outs(), llvm::errs(),
               {{::lld::Gnu, &lld::elf::link}});

  // TODO: Use lld::elf::LinkerDriver directly
  // auto *context = new lld::elf::Ctx;
  // lld::elf::Ctx &ctx = *context;
  // lld::elf::LinkerDriver &driver = ctx.driver;
  // driver.addFile(
  //     "/root/furiosa-mlir/include/furiosa-mlir/Target/Furiosa/libpe/linker.ld");
  // driver.addFile(filepath.str());
  // ctx.arg.outputFile = filepath_out.str();
  // driver.link(args);
  // CommonLinkerContext::destroy();

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
      if (binBuffer.size() < section.getAddress()) {
        binBuffer.resize(section.getAddress());
      }
      binBuffer += contents->str();
    }
  }

  return binBuffer;
}

} // namespace mlir::furiosa
