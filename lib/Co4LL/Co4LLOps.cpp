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

using namespace mlir;

static LogicalResult verify(co4ll::ConcatOp op) {
  ShapedType result = op.getResult().getType().cast<ShapedType>();
  unsigned total = 0;
  for (Value input : op->getOperands()) {
    ShapedType in = input.getType().cast<ShapedType>();
    if (in.getElementType() != result.getElementType())
      return op.emitOpError("expected input and result vectors to hold "
                            "elements of the same type");
    total += in.getNumElements();
  }
  if (result.getNumElements() != total)
    return op.emitOpError("expects total # of elements of inputs to equal # of elements in result");
  return success();
}

static LogicalResult verify(co4ll::TBOp op) {
  co4ll::ReturnOp ret = op.getReturnOp();
  if (!ret)
    return op.emitOpError(
        "expects body to be single basic block terminating in co4ll.return op");
  if (op->getNumResults() != ret->getNumOperands())
    return op.emitOpError(
        "expects number of results to equal number of operands in nested co4ll.return op");
  for (unsigned i = 0; i < op->getNumResults(); i++) {
    if (op->getResult(i).getType() != ret->getOperand(i).getType())
      return op.emitOpError(
                 "expects result types to match operands of nested "
                 "co4ll.return op, but encountered mismatch for operand ")
             << (i + 1) << " out of " << op->getNumResults();
  }
  return success();
}

#define GET_OP_CLASSES
#include "Co4LL/Co4LLOps.cpp.inc"
