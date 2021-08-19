//===- Co4HLOps.cpp - Co4 High-Level Dialect Ops ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Co4HL/Co4HLOps.h"
#include "Co4HL/Co4HLDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Co4HL/Co4HLOps.cpp.inc"
