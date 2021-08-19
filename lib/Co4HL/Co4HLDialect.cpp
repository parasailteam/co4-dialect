//===- Co4HLDialect.cpp - Co4 High-Level Dialect ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Co4HL/Co4HLDialect.h"
#include "Co4HL/Co4HLOps.h"

using namespace mlir;
using namespace mlir::co4hl;

//===----------------------------------------------------------------------===//
// Co4HL dialect.
//===----------------------------------------------------------------------===//

void Co4HLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Co4HL/Co4HLOps.cpp.inc"
      >();
}
