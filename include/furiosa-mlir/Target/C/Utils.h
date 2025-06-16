#pragma once

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"

#include "lld/Common/Driver.h"

#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "furiosa-mlir/Target/C/FuriosaTaskToC.h"

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

LogicalResult convertArmCToObject(llvm::Twine filepath_in,
                                  llvm::Twine filepath_out) {
  if (failed(InitializeAArch64())) {
    llvm::report_fatal_error("Failed to initialize AArch64 target");
    return failure();
  }

  // Compile the C code
  auto invocation = std::make_shared<clang::CompilerInvocation>();
  const char *args[] = {""}; // -v to make verbose
  auto options = clang::DiagnosticOptions();
  auto diags = clang::CompilerInstance::createDiagnostics(
      *llvm::vfs::getRealFileSystem(), options);
  clang::CompilerInvocation::CreateFromArgs(*invocation, args, *diags);
  invocation->getTargetOpts().Triple =
      "aarch64-unknown-none-elf"; // -triple aarch64-unknown-none-elf
  invocation->getFrontendOpts().Inputs.clear(); // remove default input '-'
  invocation->getFrontendOpts().Inputs.push_back(clang::FrontendInputFile(
      filepath_in.str(), clang::Language::C)); // filepath
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
  // auto command = invocation->getCC1CommandLine();
  // llvm::dbgs() << "clang ";
  // for (auto arg : command) {
  //   llvm::dbgs() << arg << " ";
  // }
  // llvm::dbgs() << "\n";

  // Compile
  clang::CompilerInstance compiler(std::move(invocation));
  compiler.setDiagnostics(diags.get());
  std::unique_ptr<clang::FrontendAction> action =
      std::make_unique<clang::EmitObjAction>();
  compiler.ExecuteAction(*action);

  return success();
}

LogicalResult linkObject(llvm::Twine filepath_in, llvm::Twine filepath_out) {
  auto linker = std::string("-T") + CLANG_CROSS_COMPILE_LIBPE + "/linker.ld";
  auto input = filepath_in.str();
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

  return success();
}

LogicalResult convertObjectToBinary(llvm::Twine filepath_in,
                                    llvm::Twine filepath_out) {
  auto status = llvm::MemoryBuffer::getFile(filepath_in);
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
  binary_t binBuffer;
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

  llvm::Expected<std::unique_ptr<llvm::FileOutputBuffer>> fileBuffer =
      llvm::FileOutputBuffer::create(filepath_out.str(), binBuffer.size());
  if (!fileBuffer) {
    llvm::report_fatal_error(
        llvm::Twine(llvm::toString(fileBuffer.takeError())));
    return failure();
  }
  llvm::FileBufferByteStream fileStream(std::move(fileBuffer.get()),
                                        llvm::endianness::native);
  llvm::BinaryStreamWriter fileWriter(fileStream);
  if (fileWriter.writeFixedString(binBuffer)) {
    return failure();
  }
  if (fileStream.commit()) {
    return failure();
  }

  return success();
}

} // namespace mlir::furiosa
