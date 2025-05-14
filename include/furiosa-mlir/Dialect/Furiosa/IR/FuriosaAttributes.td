//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef FURIOSA_ATTRIBUTES_TD
#define FURIOSA_ATTRIBUTES_TD

include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

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

def TargetAttr : Furiosa_Attr<"Target", "target"> {
  let summary = "Target attribute";
  let description = [{}];
  let parameters = (ins
      "uint64_t":$npu,
      "uint64_t":$pe_begin,
      "uint64_t":$pe_end
  );
  let assemblyFormat = "`<` `npu` $npu `pe` $pe_begin `:` $pe_end `>`";
}

def MappingAttr
    : Furiosa_Attr<"Mapping", "mapping", [DeviceMappingAttrInterface]> {
  let summary = "Mapping attribute";
  let description = [{}];
  let parameters = (ins);
  let assemblyFormat = "";
  let extraClassDeclaration = [{
  int64_t getMappingId();
  bool isLinearMapping();
  int64_t getRelativeIndex();
  }];
}

#endif // FURIOSA_ATTRIBUTES_TD
