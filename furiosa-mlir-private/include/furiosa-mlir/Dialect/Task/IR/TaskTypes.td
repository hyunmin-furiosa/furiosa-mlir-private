//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TASK_TYPES_TD
#define TASK_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

include "furiosa-mlir/Dialect/Task/IR/TaskBase.td"

//===----------------------------------------------------------------------===//
// Task type definitions
//===----------------------------------------------------------------------===//

class Task_Type<string name, string typeMnemonic, list<Trait> traits = [],
                string baseCppClass = "::mlir::Type">
    : TypeDef<Task_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let typeName = "furiosa."#typeMnemonic;
}

def SfrType : Task_Type<"Sfr", "sfr"> {
  let summary = "Special function register type";
  let description = [{}];
}

def DmaDescriptorType : Task_Type<"DmaDescriptor", "dma_descriptor"> {
  let summary = "Direct memory access descriptor type";
  let description = [{}];
}

#endif // TASK_TYPES_TD
