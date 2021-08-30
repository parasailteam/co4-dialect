#include "Co4HL/Lower.h"

#include "Co4HL/Co4HLOps.h"
#include "Co4HL/Co4HLOps.h.inc"
#include "Co4LL/Co4LLOps.h"
#include "Co4LL/Co4LLOps.h.inc"
#include "Co4LL/Co4LLDialect.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {
struct Co4LoweringPass final
    : public PassWrapper<Co4LoweringPass, OperationPass<ModuleOp>> {

  /// Make sure that we have a valid default constructor and copy constructor to
  /// ensure that the options are initialized properly.
  Co4LoweringPass() = default;
  Co4LoweringPass(const Co4LoweringPass& pass) = default;

  StringRef getArgument() const override {
    return "co4-lower";
  }
  StringRef getDescription() const override {
    return  "Convert Co4HL dialect to Co4LL dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<co4ll::Co4LLDialect>();
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override;
};

class LoweringBuilder final {
  const int numGPUs;
  const int maxNumChunks;
  ModuleOp m;
  MLIRContext *const ctx;
  SmallVector<mlir::OpBuilder, 16> builders;
  SmallVector<co4ll::GPUOp, 16> gpus;
  SmallVector<DenseMap<Value, Value>, 16> valueMaps;

public:
  LoweringBuilder(int numGPUs, int maxNumChunks, ModuleOp m)
      : numGPUs(numGPUs), maxNumChunks(maxNumChunks), m(m), ctx(m->getContext()),
        valueMaps(numGPUs) {
    mlir::OpBuilder b(ctx);
    b.setInsertionPointToEnd(&m.getRegion().back());
    for (int gpuid = 0; gpuid < numGPUs; gpuid++) {
      co4ll::GPUOp gpu = b.create<co4ll::GPUOp>(m.getLoc(), gpuid);
      gpus.push_back(gpu);
      assert(gpu.getRegion().empty());
      gpu.getRegion().emplaceBlock();
      builders.emplace_back(gpu.getRegion());
      // TODO: set return type
      co4ll::TBOp tb = builders.back().create<co4ll::TBOp>(m.getLoc(), llvm::None);
      assert(tb.getRegion().empty());
      Block &b = tb.getRegion().emplaceBlock();
      builders.back().setInsertionPoint(&b, b.begin());
    }
  }

  template <class Type, class... Args>
  void create(Operation *OldOp, Args... args) {
    for (int gpuid = 0; gpuid < numGPUs; gpuid++) {
      Operation *NewOp =
          builders[gpuid].create<Type>(OldOp->getLoc(), map(gpuid, args)...);
      NewOp->setAttrs(OldOp->getAttrs());
      assert(NewOp->getNumResults() == OldOp->getNumResults());
      for (unsigned i = 0; i < NewOp->getNumResults(); i++)
        valueMaps[gpuid][OldOp->getResult(i)] = NewOp->getResult(i);
    }
  }

  void initNewThreadblock(int gpuid, Value v, const Twine &argname,
                          unsigned argbuf) {
    // TODO: set return value
    builders[gpuid].create<co4ll::ReturnOp>(m.getLoc(), llvm::None);
    co4ll::TBOp prevTB =
        cast<co4ll::TBOp>(builders[gpuid].getBlock()->getParentOp());
    builders[gpuid].setInsertionPointAfter(prevTB);
    // TODO: set return type
    co4ll::TBOp newTB =
        builders[gpuid].create<co4ll::TBOp>(m.getLoc(), llvm::None);
    assert(newTB.getRegion().empty());
    Block &b = newTB.getRegion().emplaceBlock();
    builders[gpuid].setInsertionPoint(&b, b.begin());
    threadblockArg(gpuid, v, argbuf);
    newTB->setAttr("localinputs",
                   builders[gpuid].getArrayAttr({builders[gpuid].getArrayAttr(
                       {builders[gpuid].getStringAttr(argname),
                        builders[gpuid].getI64IntegerAttr(argbuf)})}));
  }

private:
  void initGPUs() {
  }

  Value threadblockArg(int gpuid, Value oldVal, unsigned argbuf) {
    Block& threadblock = *builders[gpuid].getBlock();
    VectorType loweredArgTy = VectorType::get({maxNumChunks}, FloatType::getF32(ctx));
    while (argbuf >= threadblock.getNumArguments())
      threadblock.addArgument(loweredArgTy);
    BlockArgument newArg = threadblock.getArgument(argbuf);

    int chunks = oldVal.getType().cast<ShapedType>().getNumElements();
    Value newVal = newArg;
    if (chunks < maxNumChunks) {
      size_t dim = 1;
      SmallVector<int64_t> offsets(dim, 0);
      SmallVector<int64_t> sizes(dim, chunks);
      SmallVector<int64_t> strides(dim, 1);
      newVal = builders[gpuid]
                   .create<vector::ExtractStridedSliceOp>(
                       oldVal.getLoc(), newArg, offsets, sizes, strides)
                   .getResult();
    }
    valueMaps[gpuid][oldVal] = newVal;
    return newVal;
  }

  // Note this template is explicitly specialized below.
  template <class T>
  T map(int gpuid, T x) {
    return x;
  }
};
// Template specialization to apply the map if the 2nd argument is a Value
template <>
Value LoweringBuilder::map(int gpuid, Value x) {
  if (valueMaps[gpuid].count(x)) {
    return valueMaps[gpuid][x];
  } else if (BlockArgument arg = x.dyn_cast<BlockArgument>()) {
    co4hl::AlgoOp algo = cast<co4hl::AlgoOp>(arg.getOwner()->getParentOp());
    assert(arg.getOwner() == &(algo.getRegion().front()));
    unsigned origArgNumber = arg.getArgNumber();
    assert(origArgNumber < algo.argbufs().size());
    int argbuf = algo.argbufs()[origArgNumber].cast<IntegerAttr>().getInt();
    Value newArg = threadblockArg(gpuid, x, argbuf);
    return newArg;
  }
  llvm::errs() << "Looking for value: " << x << "\n";
  llvm_unreachable("Unable to find value in map");
}

} // end anonymous namespace

void Co4LoweringPass::runOnOperation() {
  ModuleOp m = getOperation();

  co4hl::AlgoOp algo;
  int64_t maxNumChunks = 0;
  for (Operation &op : m.body().getOps()) {
    if (co4hl::AlgoOp a = dyn_cast<co4hl::AlgoOp>(&op)) {
      assert(!algo && "Multiple algo ops not supported");
      algo = a;

      for (Operation &op : a.body().getOps()) {
        for (Value result : op.getResults()) {
          assert(result.getType().isa<ShapedType>());
          maxNumChunks =
              std::max(maxNumChunks,
                       result.getType().cast<ShapedType>().getNumElements());
        }
      }
    }
  }
  if (!algo)
    return;
  assert(maxNumChunks > 0 && "No tensor values found in algo?");

  LoweringBuilder builder(algo.numgpus(), maxNumChunks, m);

  unsigned uniqueOutputID = 0;

  SmallVector<ModuleOp, 16> submodules;
  for (Operation &op : m.body().getOps())
    if (ModuleOp submodule = dyn_cast<ModuleOp>(&op))
      submodules.push_back(submodule);
  auto emitCollective = [&](StringRef collectiveName, Value oldVal) -> void {
    assert(oldVal.isa<OpResult>());
    unsigned dstbuf = oldVal.cast<OpResult>()
                          .getOwner()
                          ->getAttrOfType<IntegerAttr>("dstbuf")
                          .getInt();
    auto it = llvm::find_if(submodules, [&](ModuleOp submodule) {
      StringAttr collectiveAttr =
          submodule->getAttrOfType<StringAttr>("co4hl.collective");
      assert(collectiveAttr && "nested module lacks co4hl.collective attribute");
      return collectiveAttr.getValue() == collectiveName;
    });
    assert(it != submodules.end() &&
           "Unable to find collective implementation");
    ModuleOp submodule = *it;
    for (Operation &gpu : submodule.body().getOps()) {
      Operation *newGpu = gpu.clone();
      algo->getBlock()->getOperations().insert(algo->getIterator(),
                                               newGpu);
      assert(gpu.getRegion(0).hasOneBlock());
      Block &gpuBody = newGpu->getRegion(0).front();
      assert(std::distance(gpuBody.begin(), gpuBody.end()) == 1);
      co4ll::TBOp tb = cast<co4ll::TBOp>(&gpuBody.front());
      Builder b(algo);
      std::string uniqueOutputName;
      llvm::raw_string_ostream os(uniqueOutputName);
      os << collectiveName << "Output" << uniqueOutputID;
      os.flush();
      tb->setAttr("localoutputs",
                  b.getArrayAttr({b.getStringAttr(uniqueOutputName)}));
      builder.initNewThreadblock(cast<co4ll::GPUOp>(newGpu).gpuid(), oldVal,
                                 uniqueOutputName, dstbuf);
      uniqueOutputID++;
    }
  };

  for (auto &op : algo.getOps()) {
    TypeSwitch<Operation *>(&op)
        .Case<MulFOp>([&](auto mulf) {
          builder.create<MulFOp>(mulf, mulf.getOperand(0), mulf.getOperand(1));
        })
        .Case<AddFOp>([&](auto addf) {
          builder.create<AddFOp>(addf, addf.getOperand(0), addf.getOperand(1));
        })
        .Case<SubFOp>([&](auto subf) {
          builder.create<SubFOp>(subf, subf.getOperand(0), subf.getOperand(1));
        })
        .Case<math::RsqrtOp>([&](auto rsqrt) {
          builder.create<math::RsqrtOp>(rsqrt, rsqrt.getOperand());
        })
        .Case<co4hl::AllReduceOp>([&](auto ar) {
          emitCollective("all_reduce", ar.res());
        })
        .Case<co4hl::ReduceScatterOp>([&](auto rs) {
          emitCollective("reduce_scatter", rs.res());
        })
        .Case<co4hl::AllGatherOp>([&](auto ag) {
          emitCollective("all_gather", ag.res());
        })
        .Case<co4hl::ReturnOp>([&](auto ret) {
          // TODO: Set return value
          builder.create<co4ll::ReturnOp>(ret, llvm::None);
        })
        .Default([&](Operation *op) {
          llvm::errs() << "Unexpected op: " << *op << "\n";
          llvm_unreachable("Unexpected op type");
        });
  }

  algo->erase();
  for (ModuleOp submodule : submodules)
    submodule->erase();
}

void registerCo4LoweringPass() {
  PassRegistration<Co4LoweringPass>();
}
