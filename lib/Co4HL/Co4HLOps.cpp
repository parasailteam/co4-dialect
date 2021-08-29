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
#include "mlir/IR/Builders.h"

using namespace mlir;

co4hl::ReturnOp co4hl::AlgoOp::getReturnOp() {
  Operation *term = getRegion().front().getTerminator();
  return llvm::dyn_cast<co4hl::ReturnOp>(term);
}

static LogicalResult verify(co4hl::AlgoOp op) {
  co4hl::ReturnOp ret = op.getReturnOp();
  if (!ret)
    return op.emitOpError(
        "expects body to be single basic block terminating in co4hl.return op");
  if (op->getNumResults() != ret->getNumOperands())
    return op.emitOpError(
        "expects number of results to equal number of operands in nested co4hl.return op");
  for (unsigned i = 0; i < op->getNumResults(); i++) {
    if (op->getResult(i).getType() != ret->getOperand(i).getType())
      return op.emitOpError(
                 "expects result types to match operands of nested "
                 "co4hl.return op, but encountered mismatch for operand ")
             << (i + 1) << " out of " << op->getNumResults();
  }
  return success();
}

#define GET_OP_CLASSES
#include "Co4HL/Co4HLOps.cpp.inc"
