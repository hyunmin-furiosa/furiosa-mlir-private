//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef FURIOSA_PASSES_TD
#define FURIOSA_PASSES_TD

include "mlir/Pass/PassBase.td"

def FuriosaPromoteSlicePartitionLoopPass
    : Pass<"furiosa-promote-slice-partition-loop"> {
  let summary = "Promote slice partition loop to outer block";
  let dependentDialects = ["scf::SCFDialect, furiosa::FuriosaDialect"];
}

def FuriosaDeallocationPass : Pass<"furiosa-deallocation"> {
  let summary = "Insert deallocation operations for allocations";
  let dependentDialects = ["furiosa::FuriosaDialect"];
}

def FuriosaAllocateAddressPass : Pass<"furiosa-allocate-address"> {
  let summary = "Allocate addresses for SRAM memory operations";
  let dependentDialects = ["furiosa::FuriosaDialect"];
}

def FuriosaLoadTensorRegisterFilePass
    : Pass<"furiosa-load-tensor-register-file"> {
  let summary = "Add load tensor register file operations";
  let dependentDialects = ["furiosa::FuriosaDialect"];
}

#endif // FURIOSA_PASSES_TD
