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

void StepEmitter::emitOp(Operation *inst, StringRef type) {
  llvm::errs() << "   <step s=\"" << stepcount++ << "\" "
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

  // TODO: handle dependencies
  llvm::errs() << "depid=\"" << -1 << "\" ";
  llvm::errs() << "deps=\"" << -1 << "\" ";
  llvm::errs() << "hasdep=\"" << 0 << "\" \\>\n";
}

void EmitXMLPass::runOnOperation() {
  ModuleOp m = getOperation();

  llvm::errs() << "<algo name=\"Co4LL\" nchunksperloop=\"1\" nchannels=\"1\" proto=\"Simple\">\n";
  for (auto &op : m.getOps()) {
    co4ll::GPUOp gpu = cast<co4ll::GPUOp>(op);
    IntegerAttr gpuid = gpu->getAttrOfType<IntegerAttr>("gpuid");
    assert(gpuid && "gpuid attr missing");
    llvm::errs() << " <gpu id=\"" << gpuid.getInt()
                 << "\" i_chunks=\"1\" o_chunks=\"1\" s_chunks=\"1\" >\n";
    unsigned tbid = 0;
    for (auto &op : gpu.getOps()) {
      co4ll::TBOp tb = cast<co4ll::TBOp>(op);
      llvm::errs() << "  <tb id=\"" << tbid++
                   << "\" send=\"-1\" recv=\"-1\" chan=\"0\">\n";
      StepEmitter e;
      for (Operation &inst : tb.getOps()) {
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
            .Case<co4ll::ReturnOp>([&](auto) {})
            .Case<vector::ExtractStridedSliceOp>([&](auto) {})
            .Case<co4ll::ConcatOp>([&](auto) {})
            .Default([&](Operation *op) {
              llvm::errs() << "Unexpected instruction type:\n  " << *op << "\n";
            });
      }
      llvm::errs() << "  </tb>\n";
    }
    llvm::errs() << " </gpu>\n";
  }
  llvm::errs() << "</algo>\n";
}

void registerEmitXMLPass() {
  PassRegistration<EmitXMLPass>();
}
