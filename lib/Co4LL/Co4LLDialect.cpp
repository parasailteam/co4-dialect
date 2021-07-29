//===- Co4LLDialect.cpp - Co4 Low-Level Dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Co4LL/Co4LLDialect.h"
#include "Co4LL/Co4LLOps.h"

using namespace mlir;
using namespace mlir::co4ll;

//===----------------------------------------------------------------------===//
// Co4LL dialect.
//===----------------------------------------------------------------------===//

void Co4LLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Co4LL/Co4LLOps.cpp.inc"
      >();
}
