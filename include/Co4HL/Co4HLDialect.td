//===- Co4HLDialect.td - Co4 High-Level Dialect ------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CO4HL_DIALECT
#define CO4HL_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Co4HL dialect definition.
//===----------------------------------------------------------------------===//

def Co4HL_Dialect : Dialect {
    let name = "co4hl";
    let summary = "A standalone out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::co4hl";
}

//===----------------------------------------------------------------------===//
// Base Co4HL operation definition.
//===----------------------------------------------------------------------===//

class Co4HL_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Co4HL_Dialect, mnemonic, traits>;

#endif // CO4HL_DIALECT
