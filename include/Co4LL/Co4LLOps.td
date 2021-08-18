//===- Co4LLOps.td - Co4 Low-Level Dialect Ops -------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CO4LL_OPS
#define CO4LL_OPS

include "Co4LLDialect.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Co4LL_FooOp : Co4LL_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `co4ll.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = co4ll.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def GPUOp : Co4LL_Op<"gpu", [ HasParent<"ModuleOp">,
  AffineScope, IsolatedFromAbove, NoRegionArguments]
  # GraphRegionNoTerminator.traits> {
  let summary = "A container for a single graph region holding thread blocks for a GPU";
  let description = [{
    TODO: Write a longer description here.
  }];

  let regions = (region SizedRegion<1>:$body);
}

def TBOp : Co4LL_Op<"tb", [ HasParent<"GPUOp"> ]> {
  let summary = "A container for a single basic block corresponding to a thread block";
  let description = [{
    This co4ll.tb op must have a single block that terminates in a co4ll.return op,
    and the operand(s) of that nested return op are made available as result(s) of this co4ll.tb op.
  }];

  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<VectorOfRank<[1]>>:$res);

  let verifier = [{ return ::verify(*this); }];

  let extraClassDeclaration = [{
    co4ll::ReturnOp getReturnOp() {
      Operation *term = getRegion().front().getTerminator();
      return llvm::dyn_cast<co4ll::ReturnOp>(term);
    }
  }];
}

// Adapted from standard dialect
def ReturnOp : Co4LL_Op<"return", [NoSideEffect, HasParent<"TBOp">,
                                MemRefsNormalizable, ReturnLike, Terminator]> {
  let summary = "return operation";
  let description = [{
    The `return` operation represents a return operation within a thread block.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
}

def Co4LL_SendOp : Co4LL_Op<"send", [ ]> {
    let summary = "transmit data";
    let description = [{
        The `co4ll.send` operation consumes a local value and transmits it
        to the GPU whose ID is given as the current threadblock's send attribute
    }];

    let arguments = (ins VectorOfRank<[1]>:$input);
}

def Co4LL_RecvOp : Co4LL_Op<"recv", [ ]> {
    let summary = "receive data";
    let description = [{
        The `co4ll.recv` operation receives a value transmitted from the GPU
        whose ID is given as the current threadblock's recv attribute.
        The received value is made available as a result of this operation
        to be consumed by subsequent local operations.
    }];

    let results = (outs VectorOfRank<[1]>:$res);
}

def Co4LL_RecvReduceSendOp : Co4LL_Op<"rrs", [ ]> {
    let summary = "receives and reduces data, and transmits the result";
    let description = [{
        The `co4ll.rrs` operation consumes a local value, and also receives
        data from another GPU, reduces these values together (elementwise),
        and finally transmits the resulting reduced data to another GPU.
        The GPU performing the `rrs` operation does update its local buffer
        or otherwise keep a local copy of either the received data or
        the reduced result that is transmitted.
        If you need to keep a local copy, use `co4ll.rrc`.
    }];

    let arguments = (ins VectorOfRank<[1]>:$input);
}

def Co4LL_RecvReduceCopyOp : Co4LL_Op<"rrc", [SameOperandsAndResultType]> {
    let summary = "receives and reduces data";
    let description = [{
        The `co4ll.rrc` operation consumes a local value, and also receives
        data from another GPU, reduces these values together (elementwise),
        and finally makes the resulting reduced value available to be used
        by subsequent local operaitons.
        Note that the received value prior to the reduction is not stored.
        If you need the received value in addition to the reduced value,
        use `co4ll.recv` and a separate local arithmetic operation.
    }];

    let arguments = (ins VectorOfRank<[1]>:$input);
    let results = (outs VectorOfRank<[1]>:$res);
}

def Co4LL_RecvCopySendOp : Co4LL_Op<"rcs", [ ]> {
    let summary = "receive and transmit data without modifying it";
    let description = [{
        This is essentially a fused `co4ll.recv` and `co4ll.send`.
        The `co4ll.rcs` operation receives a value transmitted from some GPU
        whose ID is given as the current threadblock's recv attribute.
        The received value is made available as a result of this operation
        to be consumed by subsequent local operations.
        Additionally, the same value is also transmitted to another GPU.
    }];

    let results = (outs VectorOfRank<[1]>:$res);
}

def Co4LL_ConcatOp : Co4LL_Op<"concat", [ ]> {
    let summary = "concatenates 1-D vectors into a longer 1-D vector";
    let description = [{
    This variadic operation takes one or more input vectors and produces one output vector.
    }];

    let arguments = (ins Variadic<VectorOfRank<[1]>>:$operands);
    let results = (outs VectorOfRank<[1]>:$res);

    let verifier = [{ return ::verify(*this); }];
}


#endif // CO4LL_OPS
