#include "Co4LL/LinkByGPUID.h"

#include "Co4LL/Co4LLOps.h"
#include "Co4LL/Co4LLOps.h.inc"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"

#include <unordered_map>

using namespace mlir;

namespace {
struct LinkByGPUIDPass final
    : public PassWrapper<LinkByGPUIDPass, OperationPass<ModuleOp>> {

  /// Make sure that we have a valid default constructor and copy constructor to
  /// ensure that the options are initialized properly.
  LinkByGPUIDPass() = default;
  LinkByGPUIDPass(const LinkByGPUIDPass& pass) = default;

  StringRef getArgument() const override {
    return "co4-linkbygpuid";
  }
  StringRef getDescription() const override {
    return  "Merge subroutines by combining GPU blocks that share a GPU ID";
  }

  void runOnOperation() override;
};

} // end anonymous namespace

void LinkByGPUIDPass::runOnOperation() {
  ModuleOp m = getOperation();

  std::unordered_map<int, co4ll::GPUOp> gpus;
  for (auto i = m.getRegion().op_begin(), e = m.getRegion().op_end(); i != e; ) {
    co4ll::GPUOp gpu = cast<co4ll::GPUOp>(*(i++));
    IntegerAttr gpuidAttr = gpu->getAttrOfType<IntegerAttr>("gpuid");
    assert(gpuidAttr && "gpuid attr missing");
    int gpuid = gpuidAttr.getInt();
    auto it = gpus.find(gpuid);
    if (it == gpus.end()) { // First occurance of this GPU ID
      gpus[gpuid] = gpu;
    } else {
      co4ll::GPUOp prevGPU = it->second;
      Block& prevBlock = prevGPU.getRegion().front();
      for (auto it = gpu.getRegion().op_begin(), e = gpu.getRegion().op_end(); it != e; ) {
        co4ll::TBOp tb = cast<co4ll::TBOp>(*(it++));
        tb->moveBefore(&prevBlock, prevBlock.end());
      }
      assert(gpu.getRegion().op_begin() == gpu.getRegion().op_end() &&
             "this GPU block now has an empty body");
      gpu.erase();
    }
  }
}

void registerLinkByGPUIDPass() {
  PassRegistration<LinkByGPUIDPass>();
}
