#include "Co4LL/BufAlloc.h"

#include "Co4LL/Co4LLOps.h"
#include "Co4LL/Co4LLOps.h.inc"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/VectorOps.h"
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

struct BufAlloc final {
  const unsigned numArguments;
  bool *const usedBufs;
  BufAlloc(unsigned numArguments)
      : numArguments(numArguments), usedBufs(new bool[numArguments]) {}
  ~BufAlloc() { delete[] usedBufs; }

  void setUsed(unsigned argNumber, bool used) { usedBufs[argNumber] = used; }
  bool setDst(Operation *op);
};

} // end anonymous namespace

bool BufAlloc::setDst(Operation *op) {
  IntegerAttr dstbufAttr = op->getAttrOfType<IntegerAttr>("dstbuf");
  if (dstbufAttr) {
    //llvm::errs() << "Instruction using dstbuf: " << dstbufAttr.getInt() << "\n";
    //usedBufs[dstbufAttr.getInt()] = false;
    return false;
  } else {
    IntegerType indexType = IntegerType::get(op->getContext(), 64);
    int dstbuf =
        std::find(usedBufs, usedBufs + numArguments, false) - usedBufs;
    //llvm::errs() << "Instruction asigned to use available dstbuf: " << dstbuf << "\n";
    op->setAttr("dstbuf", IntegerAttr::get(indexType, dstbuf));
    op->setAttr("dstoff", IntegerAttr::get(indexType, 0));
    usedBufs[dstbuf] = true;
    return true;
  }
}

void BufAllocPass::runOnOperation() {
  ModuleOp m = getOperation();

  for (auto &op : m.getOps()) {
    co4ll::GPUOp gpu = cast<co4ll::GPUOp>(op);
    for (auto &op : gpu.getOps()) {
      co4ll::TBOp tb = cast<co4ll::TBOp>(op);
      Region &r = tb.getRegion();
      Block &b = r.front();
      BufAlloc alloc{r.getNumArguments()};
      for (BlockArgument &arg : r.getArguments()) {
        alloc.setUsed(arg.getArgNumber(), !arg.use_empty());
      }
      bool changed = false;
      do {
        changed = false;
        for (Operation &inst : llvm::reverse(b)) {
          if (inst.getNumResults() == 0)
            continue;
          changed |=
              TypeSwitch<Operation *, bool>(&inst)
                  .Case<vector::ExtractStridedSliceOp>(
                      [&](auto extract) { return false; })
                  .Case<co4ll::ConcatOp>([&](auto) { return false; })
                  .Default([&](Operation *op) { return alloc.setDst(op); });
        }
      } while (changed);
    }
  }
}

void registerBufAllocPass() {
  PassRegistration<BufAllocPass>();
}
