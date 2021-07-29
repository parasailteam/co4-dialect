//===- Co4LLDialect.td - Co4 Low-Level Dialect -------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CO4LL_DIALECT
#define CO4LL_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Co4LL dialect definition.
//===----------------------------------------------------------------------===//

def Co4LL_Dialect : Dialect {
    let name = "co4ll";
    let summary = "A standalone out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::co4ll";
}

//===----------------------------------------------------------------------===//
// Base Co4LL operation definition.
//===----------------------------------------------------------------------===//

class Co4LL_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Co4LL_Dialect, mnemonic, traits>;

#endif // CO4LL_DIALECT
