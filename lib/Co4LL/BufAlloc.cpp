#include "Co4LL/BufAlloc.h"

#include "Co4LL/Co4LLOps.h"
#include "Co4LL/Co4LLOps.h.inc"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>

using namespace mlir;

namespace {
struct BufAllocPass final
    : public PassWrapper<BufAllocPass, OperationPass<ModuleOp>> {

  /// Make sure that we have a valid default constructor and copy constructor to
  /// ensure that the options are initialized properly.
  BufAllocPass() = default;
  BufAllocPass(const BufAllocPass& pass) = default;

  StringRef getArgument() const override {
    return "co4-bufalloc";
  }
  StringRef getDescription() const override {
    return  "Assign each instruction an available buffer to write its output to";
  }

  void runOnOperation() override;
};

} // end anonymous namespace


void BufAllocPass::runOnOperation() {
  ModuleOp m = getOperation();

  for (auto &op : m.getOps()) {
    co4ll::GPUOp gpu = cast<co4ll::GPUOp>(op);
    for (auto &op : gpu.getOps()) {
      co4ll::TBOp tb = cast<co4ll::TBOp>(op);
      Region &r = tb.getRegion();
      Block &b = r.front();
      bool* usedBufs = new bool[r.getNumArguments()];
      for (BlockArgument &arg : r.getArguments()) {
        usedBufs[arg.getArgNumber()] = !arg.use_empty();
      }
      for (Operation &inst : llvm::reverse(b)) {
        if (inst.getNumResults() == 0) continue;

        IntegerAttr dstbufAttr =
            inst.getAttrOfType<IntegerAttr>("dstbuf");
        if (dstbufAttr) {
          //llvm::errs() << "Instruction using dstbuf: " << dstbufAttr.getInt() << "\n";
          //usedBufs[dstbufAttr.getInt()] = false;
        } else {
          IndexType indexType = IndexType::get(inst.getContext());
          int dstbuf = std::find(usedBufs, usedBufs + r.getNumArguments(), false) - usedBufs;
          //llvm::errs() << "Instruction asigned to use available dstbuf: " << dstbuf << "\n";
          inst.setAttr("dstbuf", IntegerAttr::get(indexType, dstbuf));
          inst.setAttr("dstoff", IntegerAttr::get(indexType, 0));
          inst.setAttr("cnt", IntegerAttr::get(indexType, 1));
          usedBufs[dstbuf] = true;
        }
      }
      delete[] usedBufs;
    }
  }
}

void registerBufAllocPass() {
  PassRegistration<BufAllocPass>();
}
