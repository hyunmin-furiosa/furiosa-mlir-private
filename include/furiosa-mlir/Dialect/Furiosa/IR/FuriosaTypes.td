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
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaBase.td"

//===----------------------------------------------------------------------===//
// Furiosa attribute definitions
//===----------------------------------------------------------------------===//

class Furiosa_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<Furiosa_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = "furiosa."#attrMnemonic;
}

def Furiosa_AddressAttr : Furiosa_Attr<"Address", "address"> {
  let summary = "Furiosa address attribute";
  let description = [{}];
  let parameters = (ins
      "uint64_t":$address
  );
  let assemblyFormat = "`<` $address `>`";
}

def Furiosa_TargetAttr : Furiosa_Attr<"Target", "target"> {
  let summary = "Furiosa target attribute";
  let description = [{}];
  let parameters = (ins
      "uint64_t":$npu,
      "uint64_t":$pe_begin,
      "uint64_t":$pe_end
  );
  let assemblyFormat = "`<` `npu` $npu `pe` $pe_begin `:` $pe_end `>`";
}

//===----------------------------------------------------------------------===//
// Furiosa type definitions
//===----------------------------------------------------------------------===//

class Furiosa_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Furiosa_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let typeName = "furiosa."#typeMnemonic;
}

def Furiosa_TensorType : Furiosa_Type<"Tensor", "tensor"> {
  let summary = "Furiosa tensor type";
  let description = [{}];
  let assemblyFormat = "";
}

#endif
