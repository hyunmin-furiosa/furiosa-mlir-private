#pragma once

#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::furiosa {

FailureOr<std::string> convertArmCToObject(llvm::Twine filepath) {
  // Compile the C code
  llvm::Twine filepath_out = filepath + ".o";
  // std::string command = "aarch64-none-elf-gcc ";
  // TODO: add -fno-asynchronous-unwind-tables to remove .eh_frame section
  std::string command = "clang --target=aarch64-linux-gnu ";
  command += "-r ";
  // command += "-fno-builtin ";
  // command += "-fno-zero-initialized-in-bss ";
  // static constexpr std::uint32_t MAX_STACK_USAGE = 1020 * 1024;
  // command += "-Werror=stack-usage=" + std::to_string(MAX_STACK_USAGE) + " ";
  // command += "-nostdlib ";
  // command += "-fwrapv ";
  // command += "-static ";
  // command += "-Wl,-n ";
  // command += "-xc ";
  // command += "-Werror ";
  // command += "-fno-omit-frame-pointer ";
  command += "-O3 ";
  // command += "-std=c11 ";
  command += filepath.str() + " ";
  command += "-o " + filepath_out.str() + " ";
  system(command.c_str());

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
    // TODO: clang generates .note.gnu.build-id section which should not be
    // included
    // if ((section.getFlags() & llvm::ELF::SHF_ALLOC) &&
    //     section.getType() != llvm::ELF::SHT_NOBITS && section.getSize() > 0)
    //     {
    if (section.getType() == llvm::ELF::SHT_PROGBITS) {
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
