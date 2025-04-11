#include "furiosa-mlir/ExecutionEngine/DeviceRuntime.h"
#include "furiosa-mlir/Target/Furiosa/Binary.h"

int main(int argc, char **argv) {
  assert(argc == 2);
  llvm::StringRef filename = argv[1];

  mlir::furiosa::FuriosaBinary furiosaBinary =
      *mlir::furiosa::readFuriosaBinary(filename);

  launchKernel(furiosaBinary);

  return 0;
}
