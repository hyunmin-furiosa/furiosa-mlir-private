//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TASK_ATTRIBUTES_TD
#define TASK_ATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

include "furiosa-mlir/Dialect/Task/IR/TaskBase.td"

//===----------------------------------------------------------------------===//
// Task attribute definitions
//===----------------------------------------------------------------------===//

class Task_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<Task_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = attrMnemonic;
}

#endif // TASK_ATTRIBUTES_TD
