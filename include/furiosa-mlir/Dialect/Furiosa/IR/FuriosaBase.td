//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef FURIOSA_BASE_TD
#define FURIOSA_BASE_TD

include "mlir/IR/OpBase.td"

def Furiosa_Dialect : Dialect {
  let name = "furiosa";
  let cppNamespace = "::mlir::furiosa";
  let description = [{Furiosa dialect}];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Furiosa operation type definitions
//===----------------------------------------------------------------------===//

class Furiosa_Op<string mnemonic, list<Trait> traits = []>
    : Op<Furiosa_Dialect, mnemonic, traits> {}

class HostOp<string mnemonic, list<Trait> traits = []>
    : Furiosa_Op<"host."#mnemonic, traits> {
  let cppNamespace = Furiosa_Dialect.cppNamespace#"::host";
}

class TaskOp<string mnemonic, list<Trait> traits = []>
    : Furiosa_Op<"task."#mnemonic, traits> {
  let cppNamespace = Furiosa_Dialect.cppNamespace#"::task";
}

class TensorUnitCommandOp<string mnemonic, list<Trait> traits = []>
    : Furiosa_Op<"tuc."#mnemonic, traits> {
  let cppNamespace = Furiosa_Dialect.cppNamespace#"::tuc";
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Shared dag type definitions
//===----------------------------------------------------------------------===//

class NamedDag<string m, dag d> {
  string mnemonic = m;
  dag data = d;
}

#endif // FURIOSA_BASE_TD
