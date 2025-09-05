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

def MemoryType
    : I64EnumAttr<"MemoryType", "",
                  [I64EnumAttrCase<"dram", 0>, I64EnumAttrCase<"sram", 1>,
                   I64EnumAttrCase<"trf", 2>, I64EnumAttrCase<"vrf", 3>]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::furiosa";
}

def PartitionedMapAttr : Furiosa_Attr<"PartitionedMap", "partitioned_map"> {
  let summary = "Partitioned map attribute";
  let description = [{}];
  let parameters = (ins
      "AffineMap":$affine_map,
      ArrayRefParameter<"uint64_t">:$partitions
  );
  let hasCustomAssemblyFormat = 1;
  let genMnemonicAlias = 1;
}

def TensorAttr : Furiosa_Attr<"Tensor", "tensor"> {
  let summary = "Tensor attribute";
  let description = [{}];
  let parameters = (ins
      "MemoryType":$memory_type,
      OptionalParameter<"Attribute">:$memory_map
  );
  let assemblyFormat = "`<` $memory_type ( `,` $memory_map^)? `>`";
}

#endif // FURIOSA_ATTRIBUTES_TD
