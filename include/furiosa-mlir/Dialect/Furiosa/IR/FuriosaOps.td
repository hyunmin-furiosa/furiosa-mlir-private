//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef FURIOSA_OPS_TD
#define FURIOSA_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaAttributes.td"
include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaBase.td"
include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaTypes.td"

//===----------------------------------------------------------------------===//
// Furiosa operation definitions
//===----------------------------------------------------------------------===//

class Furiosa_Op<string mnemonic, list<Trait> traits = []>
    : Op<Furiosa_Dialect, mnemonic, traits> {}

def AllocOp : Furiosa_Op<"alloc"> {
  let summary = "Device memory allocation";
  let description = [{}];
  let arguments = (ins
      MemoryTypeAttr:$type,
      I64Attr:$size
  );
  let results = (outs
      AnyRankedTensor:$buffer
  );
  let assemblyFormat = "attr-dict `:` type($buffer)";
}

def DmaOp : Furiosa_Op<
                "dma", [OptionalTypesMatchWith<"", "result", "destination",
                                               "::llvm::cast<Type>($_self)">]> {
  let summary = "Device direct memory access";
  let description = [{}];
  let arguments = (ins
      AnyRankedTensor:$source,
      Optional<AnyRankedTensor>:$destination,
      I64ArrayAttr:$source_limits,
      I64ArrayAttr:$source_strides,
      I64ArrayAttr:$destination_limits,
      I64ArrayAttr:$destination_strides
  );
  let results = (outs
      Optional<AnyRankedTensor>:$result
  );
  let assemblyFormat = "$source `:` type($source) `->` ($destination^ `:` "
                       "type($destination))? type($result) attr-dict";
}

#endif // FURIOSA_OPS_TD
