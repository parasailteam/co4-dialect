#include "Standalone/EmitXML.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

#include <tuple>

using namespace mlir;

namespace {
struct EmitXMLPass final
    : public PassWrapper<EmitXMLPass, OperationPass<ModuleOp>> {

  /// Make sure that we have a valid default constructor and copy constructor to
  /// ensure that the options are initialized properly.
  EmitXMLPass() = default;
  EmitXMLPass(const EmitXMLPass& pass) = default;

  StringRef getArgument() const override {
    return "co4-emitxml";
  }
  StringRef getDescription() const override {
    return  "Emit XML instructions to run on the SCCL interpreter";
  }

  void runOnOperation() override;
};

struct StepEmitter {
  unsigned stepcount = 0;

  void emitOp(Operation *inst, StringRef type, unsigned numSources);
};
} // end anonymous namespace

static std::tuple<int, int> getDstBufferAndOffset(const Value v) {
  if (const OpResult o = v.dyn_cast<OpResult>()) {
    IntegerAttr dstbufAttr = o.getOwner()->getAttrOfType<IntegerAttr>("dstbuf");
    assert(dstbufAttr && "dstbuf attr missing");
    int dstbuf = dstbufAttr.getInt();

    IntegerAttr dstoffAttr = o.getOwner()->getAttrOfType<IntegerAttr>("dstoff");
    assert(dstoffAttr && "dstoff attr missing");
    int dstoff = dstoffAttr.getInt();

    return std::make_tuple(dstbuf, dstoff);
  } else if (const BlockArgument arg = v.dyn_cast<BlockArgument>()) {
    return std::make_tuple(arg.getArgNumber(), 0);
  } else {
    llvm_unreachable("unexpected operand type");
  }
}

void StepEmitter::emitOp(Operation *inst, StringRef type, unsigned numSources) {
  llvm::errs() << "<step s=\"" << stepcount++ << "\" "
               << "type=\"" << type << "\" ";
  if (numSources > 0) {
    int srcbuf, srcoff;
    std::tie(srcbuf, srcoff) = getDstBufferAndOffset(inst->getOperand(0));
    llvm::errs() << "srcbuf=\"" << "a" << srcbuf << "\" "
                 << "srcoff=\"" << srcoff << "\" ";
  }
  if (numSources > 1) {
    int srcbuf, srcoff;
    std::tie(srcbuf, srcoff) = getDstBufferAndOffset(inst->getOperand(1));
    llvm::errs() << "src2buf=\"" << "a" << srcbuf << "\" "
                 << "src2off=\"" << srcoff << "\" ";
  }

  int dstbuf, dstoff;
  std::tie(dstbuf, dstoff) = getDstBufferAndOffset(inst->getResult(0));
  llvm::errs() << "dstbuf=\"" << "a" << dstbuf << "\" "
               << "dstoff=\"" << dstoff << "\" ";

  IntegerAttr cnt = inst->getAttrOfType<IntegerAttr>("cnt");
  assert(cnt && "cnt attr missing");
  llvm::errs() << "cnt=\"" << cnt.getInt() << "\" ";

  // TODO: handle dependencies
  llvm::errs() << "depid=\"" << -1 << "\" ";
  llvm::errs() << "deps=\"" << -1 << "\" ";
  llvm::errs() << "hasdep=\"" << 0 << "\" \\>\n";
}
void EmitXMLPass::runOnOperation() {
  // Get the current FuncOp operation being operated on.
  ModuleOp m = getOperation();

  for (auto &op : m.getOps()) {
    FuncOp f = cast<FuncOp>(op);
    llvm::errs() << "FuncStart\n";
    StepEmitter e;
    for (Operation &inst : f.getOps()) {
      TypeSwitch<Operation *>(&inst)
          .Case<AddFOp>([&](auto addf) { e.emitOp(addf, "addf", 2); })
          .Case<SubFOp>([&](auto mulf) { e.emitOp(mulf, "subf", 2); })
          .Case<MulFOp>([&](auto mulf) { e.emitOp(mulf, "mulf", 2); })
          .Case<math::RsqrtOp>([&](auto rsqrt) { e.emitOp(rsqrt, "rsqrt", 1); })
          .Case<ReturnOp>([&](auto) {})
          .Default([&](Operation *op) {
            llvm::errs() << "Unexpected instruction type:\n  " << *op << "\n";
          });
    }
    llvm::errs() << "FuncEnd\n";
  }
}

void registerEmitXMLPass() {
  PassRegistration<EmitXMLPass>();
}
