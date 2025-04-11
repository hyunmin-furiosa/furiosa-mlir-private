#pragma once

#include "llvm/ADT/ArrayRef.h"

#include "furiosa-mlir/Target/Furiosa/Binary.h"

void launchKernel(mlir::furiosa::FuriosaBinary furiosaBinary);
