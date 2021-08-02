//===- Co4LLOps.cpp - Co4 Low-Level Dialect Ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Co4LL/Co4LLOps.h"
#include "Co4LL/Co4LLDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Co4LL/Co4LLOps.cpp.inc"