//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef HOST_BASE_TD
#define HOST_BASE_TD

include "mlir/IR/OpBase.td"

def Host_Dialect : Dialect {
  let name = "furiosa_host";
  let cppNamespace = "::mlir::furiosa::host";
  let description = [{Furiosa host dialect}];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Host operation type definitions
//===----------------------------------------------------------------------===//

class HostOp<string mnemonic, list<Trait> traits = []>
    : Op<Host_Dialect, mnemonic, traits> {}

#endif // HOST_BASE_TD
