//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaDialect.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::furiosa;

//===----------------------------------------------------------------------===//
// Mapping Attributes
//===----------------------------------------------------------------------===//

int64_t MappingAttr::getMappingId() { return 0; }

bool MappingAttr::isLinearMapping() { return false; }

int64_t MappingAttr::getRelativeIndex() { return 0; }

//===----------------------------------------------------------------------===//
// Partition Map Attributes
//===----------------------------------------------------------------------===//

Attribute PartitionedMapAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }

  AffineMap map;
  if (failed(parser.parseAffineMap(map))) {
    return {};
  }

  if (failed(parser.parseGreater())) {
    return {};
  }

  return PartitionedMapAttr::get(parser.getContext(), map, {});
}

void PartitionedMapAttr::print(AsmPrinter &printer) const {
  printer << "<";

  getAffineMap().print(printer.getStream());

  printer << ">";
}

//===----------------------------------------------------------------------===//
// Register Dialect
//===----------------------------------------------------------------------===//

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaDialect.cpp.inc"

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaEnums.cpp.inc"

#define GET_OP_CLASSES
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaTypes.cpp.inc"

void FuriosaDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaTypes.cpp.inc"
      >();
}

void mlir::furiosa::registerFuriosaDialect(DialectRegistry &registry) {
  registry.insert<FuriosaDialect>();
}
