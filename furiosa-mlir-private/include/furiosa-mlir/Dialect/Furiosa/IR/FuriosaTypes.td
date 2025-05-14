//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef FURIOSA_TYPES_TD
#define FURIOSA_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaBase.td"

//===----------------------------------------------------------------------===//
// Furiosa type definitions
//===----------------------------------------------------------------------===//

class Furiosa_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Furiosa_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let typeName = "furiosa."#typeMnemonic;
}

def BufferType : Furiosa_Type<"Buffer", "buffer"> {
  let summary = "Buffer type";
  let description = [{}];
}

#endif // FURIOSA_TYPES_TD
