#include "Co4LL/EmitXML.h"

#include "Utils.h"

#include "Co4LL/Co4LLOps.h"
#include "Co4LL/Co4LLOps.h.inc"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
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

  void emitOp(Operation *inst, StringRef type);
};
} // end anonymous namespace

std::tuple<int, int> mlir::co4ll::getBufferAndOffset(const Value v) {
  if (const OpResult o = v.dyn_cast<OpResult>()) {
    int dstbuf, dstoff;
    if (vector::ExtractStridedSliceOp extract =
            llvm::dyn_cast<vector::ExtractStridedSliceOp>(o.getOwner())) {
      assert(llvm::all_of(extract.strides(), [](Attribute attr) {
        return attr.cast<IntegerAttr>().getInt() == 1;
      }));

      assert(extract.offsets().size() == 1);

      std::tie(dstbuf, dstoff) = getBufferAndOffset(extract.getOperand());
      dstoff += extract.offsets()[0].cast<IntegerAttr>().getInt();
    } else if (co4ll::TBOp producerTB = llvm::dyn_cast<co4ll::TBOp>(o.getOwner())) {
      std::tie(dstbuf, dstoff) = getBufferAndOffset(
          producerTB.getReturnOp().getOperand(o.getResultNumber()));
    } else {
      IntegerAttr dstbufAttr =
          o.getOwner()->getAttrOfType<IntegerAttr>("dstbuf");
      assert(dstbufAttr && "dstbuf attr missing");
      dstbuf = dstbufAttr.getInt();

      IntegerAttr dstoffAttr =
          o.getOwner()->getAttrOfType<IntegerAttr>("dstoff");
      assert(dstoffAttr && "dstoff attr missing");
      dstoff = dstoffAttr.getInt();
    }

    return std::make_tuple(dstbuf, dstoff);
  } else if (const BlockArgument arg = v.dyn_cast<BlockArgument>()) {
    return std::make_tuple(arg.getArgNumber(), 0);
  } else {
    llvm_unreachable("unexpected operand type");
  }
}

static bool isEmittedAsXML(Operation *op) {
  return !(isa<vector::ExtractStridedSliceOp>(op) || isa<co4ll::ConcatOp>(op) ||
           isa<co4ll::ReturnOp>(op));
}

static bool isUsedByOtherTB(OpResult v) {
  assert(!isa<co4ll::TBOp>(v.getOwner()));
  co4ll::TBOp producerTB = llvm::cast<co4ll::TBOp>(v.getOwner()->getParentOp());
  for (OpOperand& use : v.getUses()) {
    Operation *user = use.getOwner();
    assert(!isa<co4ll::TBOp>(user));
    assert(user->getParentOp() == producerTB);
    if (co4ll::ReturnOp ret = llvm::dyn_cast<co4ll::ReturnOp>(user)) {
      Value sharedVal = producerTB->getResult(use.getOperandNumber());
      for (Operation *user : sharedVal.getUsers()) {
        assert(!isa<co4ll::TBOp>(user));
        co4ll::TBOp userTB = llvm::cast<co4ll::TBOp>(user->getParentOp());
        //llvm::errs() << "\n\n User: " << *user << "\n\nUsed: " << sharedVal
        //             << "\n " << v << "\n\n";
        assert(producerTB != userTB &&
               "senseless for a TB to use its own returned value");
        assert(producerTB->getParentOp() == userTB->getParentOp() &&
               "only TBs within a single GPU can directly share local values");
        return true;
      }
    }
  }

  // If the following instruction is a concat instruction producing a result
  // used by another TB, then the current instruction should be treated as
  // having a dependence, to ensure that at least one instruction emited to XML
  // will have the hasdep attribute set to 1.
  for (Operation *nextOp = v.getOwner()->getNextNode();
       nextOp && !isEmittedAsXML(nextOp); nextOp = nextOp->getNextNode()) {
    for (OpResult r : nextOp->getResults())
      if (isUsedByOtherTB(r))
        return true;
  }

  return false;
}

static Operation* getUniqueDependencyFromOtherTB(Operation *op) {
  OpResult dep;
  co4ll::TBOp consumerTB = cast<co4ll::TBOp>(op->getParentOp());
  for (Value in : op->getOperands())
    if (const OpResult tbResult = in.dyn_cast<OpResult>())
      if (tbResult.getOwner()->getParentOp() != consumerTB) {
        assert(!dep &&
               "TODO: emit no-ops to XML to support when an instruction "
               "needs to wait on values produced in multiple other TBs. "
               "For now we only support operations that have at most one"
               "cross-TB dependency.");
        dep = tbResult;
      }

  if (!dep) return nullptr;

  co4ll::TBOp producerTB = cast<co4ll::TBOp>(dep.getOwner());
  Value producer =
      producerTB.getReturnOp().getOperand(dep.getResultNumber());
  assert(!producer.isa<BlockArgument>() &&
         "Threadblock returns a value that it did not produce? "
         "TODO: Handle this case correctly.");
  return producer.cast<OpResult>().getOwner();
}

static int getTBID(co4ll::TBOp tb) {
  int count = 0;
  for (auto &op : cast<co4ll::GPUOp>(tb->getParentOp()).getOps()) {
    if (&op == tb)
      return count;
    count++;
  }
  llvm_unreachable("");
}

static int getStepWithinTB(Operation *op) {
  int count = 0;
  for (auto &o : cast<co4ll::TBOp>(op->getParentOp()).getOps()) {
    if (isEmittedAsXML(&o))
      count++;
    if (&o == op)
      return count - 1;
  }
  llvm_unreachable("");
}

void StepEmitter::emitOp(Operation *inst, StringRef type) {
  llvm::errs() << "      <step s=\"" << stepcount++ << "\" "
               << "type=\"" << type << "\" ";
  unsigned numSources = inst->getNumOperands();
  int srcbuf, srcoff;
  if (numSources > 0)
    std::tie(srcbuf, srcoff) = co4ll::getBufferAndOffset(inst->getOperand(0));
  else
    // No meaningful source, but XML interpreter expects to parse at least
    // 1 src per instruction.
    std::tie(srcbuf, srcoff) = std::make_tuple(-1, 0);
  llvm::errs() << "srcbuf=\"" << "a" << srcbuf << "\" "
               << "srcoff=\"" << srcoff << "\" ";
  if (numSources > 1) {
    int srcbuf, srcoff;
    std::tie(srcbuf, srcoff) = co4ll::getBufferAndOffset(inst->getOperand(1));
    llvm::errs() << "src2buf=\"" << "a" << srcbuf << "\" "
                 << "src2off=\"" << srcoff << "\" ";
  }

  assert(inst->getNumResults() <= 1);

  int dstbuf, dstoff;
  if (inst->getNumResults() >= 1)
    std::tie(dstbuf, dstoff) = co4ll::getBufferAndOffset(inst->getResult(0));
  else
    // No meaningful destination, but XML interpreter expects to parse
    // 1 dst per instruction.
    std::tie(dstbuf, dstoff) = std::make_tuple(-1, 0);
  llvm::errs() << "dstbuf=\"" << "a" << dstbuf << "\" "
               << "dstoff=\"" << dstoff << "\" ";

  if (inst->getNumResults() == 1) {
    ShapedType resultType = inst->getResult(0).getType().dyn_cast<ShapedType>();
    assert(resultType);
    llvm::errs() << "cnt=\"" << resultType.getNumElements() << "\" ";
  } else if (numSources > 0) {
    ShapedType sourceType = inst->getOperand(0).getType().dyn_cast<ShapedType>();
    llvm::errs() << "cnt=\"" << sourceType.getNumElements() << "\" ";
  } else {
    llvm_unreachable(
        "Don't know what to use for cnt if op has no operands nor results");
  }

  Operation *dep = getUniqueDependencyFromOtherTB(inst);
  llvm::errs() << "depid=\""
               << (dep ? getTBID(cast<co4ll::TBOp>(dep->getParentOp())) : -1)
               << "\" ";
  llvm::errs() << "deps=\"" << (dep ? getStepWithinTB(dep) : -1) << "\" ";

  llvm::errs() << "hasdep=\""
               << (int)llvm::any_of(
                      inst->getResults(),
                      [](OpResult out) { return isUsedByOtherTB(out); })
               << "\" \\>\n";
}

void EmitXMLPass::runOnOperation() {
  ModuleOp m = getOperation();

  llvm::errs() << "<algo name=\"Co4LL\" nchunksperloop=\"1\" nchannels=\"1\" proto=\"Simple\">\n";
  for (auto &op : m.getOps()) {
    co4ll::GPUOp gpu = cast<co4ll::GPUOp>(op);
    llvm::errs() << "  <gpu id=\"" << gpu.gpuid() << "\" i_chunks=\""
                 << gpu.numchunks() << "\" o_chunks=\"" << gpu.numchunks()
                 << "\" s_chunks=\"" << gpu.numchunks() << "\" >\n";
    unsigned tbid = 0;
    for (auto &op : gpu.getOps()) {
      co4ll::TBOp tb = cast<co4ll::TBOp>(op);
      llvm::errs() << "    <tb id=\"" << tbid++
                   << "\" send=\"-1\" recv=\"-1\" chan=\"0\">\n";
      StepEmitter e;
      for (Operation &inst : tb.getOps()) {
        if (!isEmittedAsXML(&inst)) continue;
        TypeSwitch<Operation *>(&inst)
            .Case<AddFOp>([&](auto addf) { e.emitOp(addf, "addf"); })
            .Case<SubFOp>([&](auto mulf) { e.emitOp(mulf, "subf"); })
            .Case<MulFOp>([&](auto mulf) { e.emitOp(mulf, "mulf"); })
            .Case<math::RsqrtOp>([&](auto rsqrt) { e.emitOp(rsqrt, "rsqrt"); })
            .Case<co4ll::SendOp>([&](auto send) { e.emitOp(send, "s"); })
            .Case<co4ll::RecvOp>([&](auto recv) { e.emitOp(recv, "r"); })
            .Case<co4ll::RecvReduceSendOp>(
                [&](auto send) { e.emitOp(send, "rrs"); })
            .Case<co4ll::RecvReduceCopyOp>(
                [&](auto send) { e.emitOp(send, "rrc"); })
            .Case<co4ll::RecvCopySendOp>(
                [&](auto send) { e.emitOp(send, "rcs"); })
            .Default([&](Operation *op) {
              llvm::errs() << "Unexpected instruction type:\n  " << *op << "\n";
            });
      }
      llvm::errs() << "    </tb>\n";
    }
    llvm::errs() << "  </gpu>\n";
  }
  llvm::errs() << "</algo>\n";
}

void registerEmitXMLPass() {
  PassRegistration<EmitXMLPass>();
}
