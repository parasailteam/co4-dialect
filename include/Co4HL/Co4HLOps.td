//===- Co4HLOps.td - Co4 High-Level Dialect Ops ------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CO4HL_OPS
#define CO4HL_OPS

include "Co4HLDialect.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Co4HL_FooOp : Co4HL_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `co4hl.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = co4hl.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def CO4HL_AlgoOp : Co4HL_Op<"algo", [ HasParent<"ModuleOp">,
  AffineScope, IsolatedFromAbove ]> {
  let summary = "A container for a single basic block corresponding to a whole distributed algorithm";
  let description = [{
    This co4hl.algo op must have a single block that terminates in a co4hl.return op.
  }];

  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins I64Attr:$numgpus, I64ArrayAttr:$argbufs);
  let results = (outs Variadic<1DTensorOf<[AnyType]>>:$res);

  let verifier = [{ return ::verify(*this); }];

  let extraClassDeclaration = [{
    co4hl::ReturnOp getReturnOp();
  }];
}

// Adapted from standard dialect
def Co4HL_ReturnOp : Co4HL_Op<"return", [NoSideEffect, HasParent<"AlgoOp">,
                                MemRefsNormalizable, ReturnLike, Terminator]> {
  let summary = "return operation";
  let description = [{
    The operation takes variable number of operands and produces no results.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
}

def Co4HL_AllReduceOp : Co4HL_Op<"all_reduce", [ NoSideEffect, SameOperandsAndResultType, HasParent<"AlgoOp"> ]> {
    let summary = "AllReduce collective";
    let description = [{
    }];

    let arguments = (ins 1DTensorOf<[AnyType]>:$operands, StrAttr:$func, I64Attr:$dstbuf);
    let results = (outs 1DTensorOf<[AnyType]>:$res);
}

def Co4HL_ReduceScatterOp : Co4HL_Op<"reduce_scatter", [ NoSideEffect, HasParent<"AlgoOp"> ]> {
    let summary = "ReduceScatter collective";
    let description = [{
    }];

    let arguments = (ins 1DTensorOf<[AnyType]>:$operands, StrAttr:$func, I64Attr:$dstbuf);
    let results = (outs 1DTensorOf<[AnyType]>:$res);
}

def Co4HL_AllGatherOp : Co4HL_Op<"all_gather", [ NoSideEffect, HasParent<"AlgoOp"> ]> {
    let summary = "AllGather collective";
    let description = [{
    }];

    let arguments = (ins 1DTensorOf<[AnyType]>:$operands, I64Attr:$dstbuf);
    let results = (outs 1DTensorOf<[AnyType]>:$res);
}

#endif // CO4HL_OPS
