//===- co4-opt.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Co4HL/Co4HLDialect.h"
#include "Co4HL/Co4HLOpsDialect.cpp.inc"
#include "Co4LL/Co4LLDialect.h"
#include "Co4LL/Co4LLOpsDialect.cpp.inc"

#include "Co4HL/Lower.h"
#include "Co4LL/BufAlloc.h"
#include "Co4LL/EmitXML.h"
#include "Co4LL/LinkByGPUID.h"
#include "Co4LL/ThreadblockSSA.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  registerCo4LoweringPass();
  registerBufAllocPass();
  registerEmitXMLPass();
  registerLinkByGPUIDPass();
  registerThreadblockSSAPass();

  mlir::DialectRegistry registry;
  registry.insert<mlir::co4ll::Co4LLDialect>();
  registry.insert<mlir::co4hl::Co4HLDialect>();
  registry.insert<mlir::StandardOpsDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Co4 optimizer driver\n", registry));
}
