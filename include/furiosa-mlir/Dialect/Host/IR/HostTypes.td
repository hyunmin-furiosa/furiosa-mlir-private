//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef HOST_TYPES_TD
#define HOST_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

include "furiosa-mlir/Dialect/Host/IR/HostBase.td"

//===----------------------------------------------------------------------===//
// Common attribute definitions
//===----------------------------------------------------------------------===//

def DenseByteArrayAttr
    : DenseArrayAttrBase<"DenseByteArrayAttr", "uint8_t", "ui8">;

def ByteArrayAttr
    : TypedArrayAttrBase<UI8Attr, "8-bit unsigned integer array attribute"> {
  let constBuilderCall = "$_builder.getByteArrayAttr($0)";
}

//===----------------------------------------------------------------------===//
// Host attribute definitions
//===----------------------------------------------------------------------===//

class Host_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<Host_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = attrMnemonic;
}

def TargetAttr : Host_Attr<"Target", "target"> {
  let summary = "Target attribute";
  let description = [{}];
  let parameters = (ins
      "uint64_t":$npu,
      "uint64_t":$pe_begin,
      "uint64_t":$pe_end
  );
  let assemblyFormat = "`<` `npu` $npu `pe` $pe_begin `:` $pe_end `>`";
}

//===----------------------------------------------------------------------===//
// Host type definitions
//===----------------------------------------------------------------------===//

class Host_Type<string name, string typeMnemonic, list<Trait> traits = [],
                string baseCppClass = "::mlir::Type">
    : TypeDef<Host_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let typeName = typeMnemonic;
}

def PeProgramType : Host_Type<"PeProgram", "pe_program"> {
  let summary = "PE program type";
  let description = [{}];
}

def HalProgramType : Host_Type<"HalProgram", "hal_program"> {
  let summary = "HAL program type";
  let description = [{}];
}

def DeviceType : Host_Type<"Device", "device"> {
  let summary = "Device type";
  let description = [{}];
}

def BufferType : Host_Type<"Buffer", "buffer"> {
  let summary = "Host memory buffer type";
  let description = [{}];
}

#endif // HOST_TYPES_TD
