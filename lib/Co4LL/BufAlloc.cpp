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
    : public PassWrapper<BufAllocPass, OperationPass<co4ll::GPUOp>> {

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
  bool pickDst(Operation *op);
};

} // end anonymous namespace

static bool setDstBufferAndOffset(Operation *op, int dstbuf, unsigned offset);

// end forward declarations

static bool propagateDsts(co4ll::ConcatOp concat) {
  IntegerAttr dstbufAttr = concat->getAttrOfType<IntegerAttr>("dstbuf");
  IntegerAttr dstoffAttr = concat->getAttrOfType<IntegerAttr>("dstoff");
  if (!(dstbufAttr && dstoffAttr))
    return false;

  bool changed = false;

  unsigned offset = dstoffAttr.getInt();
  for (Value v : concat.operands()) {
    // Does it even make sense to concatenate an entire arg array into an even larger array,
    // or should array sizes and offsets be limited to the size of a single arg array?
    assert(!v.isa<BlockArgument>() &&
           "We don't support using an arg array in a concatenation");
    Operation *inputOp = v.cast<OpResult>().getOwner();
    changed |= setDstBufferAndOffset(inputOp, dstbufAttr.getInt(), offset);

    offset += v.getType().cast<ShapedType>().getNumElements();
  }

  return changed;
}

static bool setDstBufferAndOffset(Operation *op, int dstbuf, unsigned offset) {
  IntegerType indexType = IntegerType::get(op->getContext(), 64);

  assert(op->getNumResults() == 1);

  bool changed = false;

  IntegerAttr dstbufAttr = op->getAttrOfType<IntegerAttr>("dstbuf");
  if (dstbufAttr) {
    assert(dstbufAttr.getInt() == dstbuf && "target buffer already set inconsistently");
  } else {
    op->setAttr("dstbuf", IntegerAttr::get(indexType, dstbuf));
    changed = true;
  }

  IntegerAttr dstoffAttr = op->getAttrOfType<IntegerAttr>("dstoff");
  if (dstoffAttr) {
    assert(dstoffAttr.getInt() == offset && "offset already set inconsistently");
  } else {
    op->setAttr("dstoff", IntegerAttr::get(indexType, offset));
    changed = true;
  }

  if (co4ll::ConcatOp concat = dyn_cast<co4ll::ConcatOp>(op))
    changed |= propagateDsts(concat);

  return changed;
}

bool BufAlloc::pickDst(Operation *op) {
  IntegerAttr dstbufAttr = op->getAttrOfType<IntegerAttr>("dstbuf");
  if (dstbufAttr) {
    //llvm::errs() << "Instruction using dstbuf: " << dstbufAttr.getInt() << "\n";
    //usedBufs[dstbufAttr.getInt()] = false;
    return false;
  }

  int dstbuf = std::find(usedBufs, usedBufs + numArguments, false) - usedBufs;
  //llvm::errs() << "Instruction asigned to use available dstbuf: " << dstbuf << "\n";
  setDstBufferAndOffset(op, dstbuf, 0);

  usedBufs[dstbuf] = true;
  return true;
}

void BufAllocPass::runOnOperation() {
  co4ll::GPUOp gpu = getOperation();
  BufAlloc alloc(32/*TODO: replace hard-coded constant with an attribute somewhere?*/);
  for (auto &op : gpu.getOps()) {
    co4ll::TBOp tb = cast<co4ll::TBOp>(op);
    Region &r = tb.getRegion();
    Block &b = r.front();
    for (BlockArgument &arg : r.getArguments()) {
      alloc.setUsed(arg.getArgNumber(), !arg.use_empty());
    }
    for (Operation &inst : b) {
      IntegerAttr dstbuf = inst.getAttrOfType<IntegerAttr>("dstbuf");
      if (dstbuf)
        alloc.setUsed(dstbuf.getInt(), true);
    }
  }
  for (auto &op : gpu.getOps()) {
    co4ll::TBOp tb = cast<co4ll::TBOp>(op);
    Region &r = tb.getRegion();
    Block &b = r.front();
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
                .Case<co4ll::ConcatOp>([&](co4ll::ConcatOp concat) {
                  return propagateDsts(concat);
                })
                .Default([&](Operation *op) { return alloc.pickDst(op); });
      }
    } while (changed);
  }
}

void registerBufAllocPass() {
  PassRegistration<BufAllocPass>();
}
