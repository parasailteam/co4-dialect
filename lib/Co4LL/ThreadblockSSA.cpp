#include "Co4LL/ThreadblockSSA.h"

#include "Co4LL/Co4LLOps.h"
#include "Co4LL/Co4LLOps.h.inc"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;

namespace {
struct ThreadblockSSAPass final
    : public PassWrapper<ThreadblockSSAPass, OperationPass<co4ll::GPUOp>> {

  /// Make sure that we have a valid default constructor and copy constructor to
  /// ensure that the options are initialized properly.
  ThreadblockSSAPass() = default;
  ThreadblockSSAPass(const ThreadblockSSAPass& pass) = default;

  StringRef getArgument() const override {
    return "co4-threadblockssa";
  }
  StringRef getDescription() const override {
    return  "Connect up the dataflow graph across threadblocks within each GPU";
  }

  void runOnOperation() override;
};

} // end anonymous namespace

void ThreadblockSSAPass::runOnOperation() {
  co4ll::GPUOp gpu = getOperation();

  llvm::StringMap<Value> localValues;
  for (auto &op : gpu.getOps()) {
    co4ll::TBOp tb = cast<co4ll::TBOp>(op);
    ArrayAttr outputs = tb->getAttrOfType<ArrayAttr>("localoutputs");
    if (!outputs) continue;
    assert(outputs.size() == tb->getNumResults());
    for (unsigned i = 0; i < outputs.size(); i++) {
      //llvm::errs() << "attribute " << i << " is: " << outputs[i] << "\n";
      assert(outputs[i].isa<StringAttr>());
      llvm::StringRef outputName = outputs[i].cast<StringAttr>().getValue();
      assert(!localValues.count(outputName));
      localValues[outputName] = tb.getResult(i);
    }
  }
  for (auto &op : gpu.getOps()) {
    co4ll::TBOp tb = cast<co4ll::TBOp>(op);
    ArrayAttr inputs = tb->getAttrOfType<ArrayAttr>("localinputs");
    if (!inputs) continue;
    for (unsigned i = 0; i < inputs.size(); i++) {
      assert(inputs[i].isa<ArrayAttr>());
      ArrayAttr input = inputs[i].cast<ArrayAttr>();
      assert(input.size() == 2);
      assert(input[0].isa<StringAttr>());
      assert(input[1].isa<IntegerAttr>());
      llvm::StringRef inputName = input[0].cast<StringAttr>().getValue();
      unsigned argNum = input[1].cast<IntegerAttr>().getInt();
      assert(localValues.count(inputName));
      Value otherTBResult = localValues[inputName];
      BlockArgument inputArg = tb.getRegion().getArgument(argNum);
      assert(!inputArg.use_empty());
      inputArg.replaceAllUsesWith(otherTBResult);
      assert(inputArg.use_empty());
    }
  }
}

void registerThreadblockSSAPass() {
  PassRegistration<ThreadblockSSAPass>();
}
