//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TASK_BASE_TD
#define TASK_BASE_TD

include "mlir/IR/OpBase.td"

def Task_Dialect : Dialect {
  let name = "furiosa_task";
  let cppNamespace = "::mlir::furiosa::task";
  let description = [{Furiosa task dialect}];
  let useDefaultTypePrinterParser = 1;
}

#endif // TASK_BASE_TD
