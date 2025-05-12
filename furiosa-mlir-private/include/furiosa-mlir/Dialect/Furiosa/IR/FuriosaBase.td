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
  let useDefaultAttributePrinterParser = 1;
}

#endif // FURIOSA_BASE_TD
