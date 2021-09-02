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

class Lowerer final {
  const int numGPUs;
  const unsigned numBuffers;
  const int maxNumChunks;
  ModuleOp m;
  MLIRContext *const ctx;
  const Type elemTy;

  // These following parallel vectors will hold one element per rank
  SmallVector<co4ll::GPUOp, 16> gpus;
  SmallVector<mlir::OpBuilder, 16> builders;
  SmallVector<SmallVector<Value>, 16> returnValues;
  // map from old values in high-level algo to newly generated values in
  // low-level threadblocks.
  SmallVector<DenseMap<Value, Value>, 16> valueMaps;

  // A collection of possible implementations of collectives.
  // TODO: replace this with some more flexible SCCL/ncclize-type compilation
  // that can generate appropriate implementations on the fly?
  SmallVector<ModuleOp, 16> submodules;

  unsigned uniqueOutputID = 0;

public:
  Lowerer(co4hl::AlgoOp algo, int maxNumChunks)
      : numGPUs(algo.numgpus()), numBuffers(algo.numbufs()),
        maxNumChunks(maxNumChunks), m(cast<ModuleOp>(algo->getParentOp())),
        ctx(m->getContext()), elemTy(FloatType::getF32(ctx)),
        returnValues(numGPUs), valueMaps(numGPUs) {
    // For each rank, create one GPU op that will contain all compute ops.
    mlir::OpBuilder b(ctx);
    b.setInsertionPointToEnd(&m.getRegion().back());
    for (int gpuid = 0; gpuid < numGPUs; gpuid++) {
      co4ll::GPUOp gpu = b.create<co4ll::GPUOp>(m.getLoc(), gpuid);
      gpus.push_back(gpu);
      assert(gpu.getRegion().empty());
      gpu.getRegion().emplaceBlock();
      builders.emplace_back(ctx);
      startComputeThreadblock(gpuid, &algo.body().front().front());
    }
  }

  void addSubmodule(ModuleOp submodule) {
    submodules.push_back(submodule);
  }

  void eraseSubmodules() {
    for (ModuleOp submodule : submodules)
      submodule->erase();
  }

  // Emit an operation performing computation (not inter-GPU communication)
  template <class OpType, class... Args>
  void emitCompute(Operation *OldOp, Args... args) {
    assert(isCompute(OldOp));
    for (int gpuid = 0; gpuid < numGPUs; gpuid++) {
      Operation *NewOp =
          builders[gpuid].create<OpType>(OldOp->getLoc(), map(gpuid, args)...);
      NewOp->setAttrs(OldOp->getAttrs());
      assert(NewOp->getNumResults() == OldOp->getNumResults());
      for (unsigned i = 0; i < NewOp->getNumResults(); i++)
        valueMaps[gpuid][OldOp->getResult(i)] = NewOp->getResult(i);
    }
  }

  // Emit a collective, which performs inter-GPU communication
  void emitCollective(StringRef collectiveName, Operation *collective) {
    assert(collective->getNumResults() == 1 &&
           "TODO: Support collectives not producing output on every rank?");
    OpResult oldVal = collective->getResult(0);
    const int chunks = oldVal.getType().cast<ShapedType>().getNumElements();
    assert(elemTy == oldVal.getType().cast<ShapedType>().getElementType());

    // Find the implementation of the collective from the "library" of submodules
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
      // Create a copy of the implementation.
      Operation *newGpu = gpu.clone();
      m.body().back().getOperations().insert(m.body().back().end(), newGpu);
      assert(gpu.getRegion(0).hasOneBlock());
      Block &gpuBody = newGpu->getRegion(0).front();
      assert(std::distance(gpuBody.begin(), gpuBody.end()) == 1);
      co4ll::TBOp collImplTb = cast<co4ll::TBOp>(&gpuBody.front());
      const int gpuid = cast<co4ll::GPUOp>(newGpu).gpuid();

      // End the thread block that previous compute ops were generated in.
      finishComputeThreadblock(gpuid, m.getLoc(), collImplTb);

      // Start a new threadblock into which we will emit subsequent compute ops.
      co4ll::TBOp newTB =
          startComputeThreadblock(gpuid, collective->getNextNode());

      BlockArgument newArg =
          newTB.getRegion().addArgument(VectorType::get({chunks}, elemTy));
      connectOutputToNextThreadblocks(gpuid, collImplTb, oldVal, newTB, newArg);
    }
  }

  void finishCompute(Location loc) {
    for (int gpuid = 0; gpuid < numGPUs; gpuid++) {
      finishComputeThreadblock(gpuid, loc);
    }
  }

private:
  bool isCompute(const Operation *op) {
    return isa<MulFOp>(op) || isa<AddFOp>(op) || isa<SubFOp>(op) ||
           isa<math::RsqrtOp>(op);
  }

  /// Emit a threadblock and set this rank's builder to insert into its body,
  /// so that subsequent compute ops will be emitted there.
  co4ll::TBOp startComputeThreadblock(int gpuid, Operation *startOp);

  co4ll::ReturnOp finishComputeThreadblock(int gpuid, Location loc,
                                           co4ll::TBOp collImpl=co4ll::TBOp());

  void connectOutputToNextThreadblocks(int gpuid, co4ll::TBOp producerTB,
                                       Value oldVal, co4ll::TBOp newTB,
                                       BlockArgument newArg);

  // Note this template is explicitly specialized below.
  template <class T>
  T map(int gpuid, T x) {
    // This primary template returns its argument unchanged
    return x;
  }
};

} // end anonymous namespace

// Template specialization to apply the map if the 2nd argument is a Value:
// this takes a Value used in the high-level algo and returns the equivalent
// Value that should be used in the emitted low-level dialect for a given gpu.
template <>
Value Lowerer::map(int gpuid, Value x) {
  if (valueMaps[gpuid].count(x)) {
    return valueMaps[gpuid][x];
  } else if (BlockArgument arg = x.dyn_cast<BlockArgument>()) {
    co4hl::AlgoOp algo = cast<co4hl::AlgoOp>(arg.getOwner()->getParentOp());
    assert(arg.getOwner() == &(algo.getRegion().front()));
    unsigned origArgNumber = arg.getArgNumber();
    assert(origArgNumber < algo.argbufs().size());
    unsigned argbuf =
        algo.argbufs()[origArgNumber].cast<IntegerAttr>().getInt();
    const int chunks = x.getType().cast<ShapedType>().getNumElements();
    assert(elemTy == x.getType().cast<ShapedType>().getElementType());
    Block& threadblock = *builders[gpuid].getBlock();
    assert(argbuf < numBuffers);
    // This arg refers to one of the input buffers based on its index
    BlockArgument newArg = threadblock.getArgument(argbuf);
    Value newVal = newArg;
    if (chunks < maxNumChunks) {
      size_t dim = 1;
      SmallVector<int64_t> offsets(dim, 0);
      SmallVector<int64_t> sizes(dim, chunks);
      SmallVector<int64_t> strides(dim, 1);
      newVal = builders[gpuid]
                   .create<vector::ExtractStridedSliceOp>(
                       x.getLoc(), newArg, offsets, sizes, strides)
                   .getResult();
    }

    valueMaps[gpuid][x] = newVal;
    return newVal;
  }
  llvm::errs() << "Looking for value: " << x << "\n";
  llvm_unreachable("Unable to find value in map");
}

co4ll::TBOp Lowerer::startComputeThreadblock(int gpuid, Operation *startOp) {
  OpBuilder &builder = builders[gpuid];
  assert(returnValues[gpuid].empty() && !builder.getInsertionBlock() &&
         "You forget to finishComputeThreadblock() on the previous block?");
  // First, determine what what this threadblock will return.  The block will
  // consist of all compute operations up until the next collective
  Operation *endOp = startOp;
  while (isCompute(endOp))
    endOp = endOp->getNextNode();
  for (Operation *op = startOp; isCompute(op); op = op->getNextNode())
    for (Value result : op->getResults())
      if (llvm::any_of(result.getUsers(), [endOp](Operation *user) {
            return user == endOp || endOp->isBeforeInBlock(user);
          })) {
        returnValues[gpuid].push_back(result);
      }
  SmallVector<Type> ReturnTypes;
  for (Value v : returnValues[gpuid]) {
    unsigned chunks = v.getType().cast<ShapedType>().getNumElements();
    ReturnTypes.push_back(VectorType::get({chunks}, elemTy));
  }

  // Okay, now we're ready to create this threadblock
  builder.setInsertionPoint(&gpus[gpuid].getRegion().back(),
                            gpus[gpuid].getRegion().back().end());
  co4ll::TBOp tb = builder.create<co4ll::TBOp>(m.getLoc(), ReturnTypes);
  assert(tb.getRegion().empty());
  Block &newTBBlock = tb.getRegion().emplaceBlock();
  VectorType loweredArgTy = VectorType::get({maxNumChunks}, elemTy);
  while (newTBBlock.getNumArguments() < numBuffers)
    newTBBlock.addArgument(loweredArgTy);
  builder.setInsertionPoint(&newTBBlock, newTBBlock.begin());
  return tb;
}

co4ll::ReturnOp Lowerer::finishComputeThreadblock(int gpuid, Location loc, co4ll::TBOp collImpl) {
  assert(builders[gpuid].getInsertionBlock() &&
         "Did you forget to startComputeThreadblock()?");
  SmallVector<Value> mappedReturnVals;
  for (Value v : returnValues[gpuid])
    mappedReturnVals.push_back(map(gpuid, v));
  co4ll::ReturnOp ret =
      builders[gpuid].create<co4ll::ReturnOp>(loc, mappedReturnVals);
  // Hook up output from this compute block to the input of the collective.
  // FIXME(victory): This assumes that the input to the collective is either an
  // input to the whole algo or produced in this compute block immediately
  // preceeding the collective.  This code won't work if a result from some
  // local compute needs to be fed to, say, multiple collectives, or a
  // collective that occurs much later in the algo.
  if (collImpl)
    for (Value v : returnValues[gpuid]) {
      assert(collImpl.getRegion().getNumArguments() == 1);
      BlockArgument collArg = collImpl.getRegion().getArgument(0);
      connectOutputToNextThreadblocks(
          gpuid, cast<co4ll::TBOp>(ret->getParentOp()), v, collImpl, collArg);
    }
  returnValues[gpuid].clear();
  builders[gpuid].clearInsertionPoint();


  return ret;
}

void Lowerer::connectOutputToNextThreadblocks(int gpuid, co4ll::TBOp producerTB,
                                              Value oldVal, co4ll::TBOp newTB,
                                              BlockArgument newArg) {
  Builder b(ctx);

  // Because co4ll::GPUOp has trait IsolatedFromAbove, it is not legal for
  // ops nested inside one gpu to directly use SSA Values from another gpu.
  // So we'll have the LinkByGPUID pass merge gpu ops and then the
  // ThreadblockSSA pass can link up the use of values within one rank.
  // To that end, we generate a unique label for the produced output
  // which will tell ThreadblockSSA what connections to make.
  std::string uniqueOutputName;
  llvm::raw_string_ostream os(uniqueOutputName);
  os << "threadblock_output_" << uniqueOutputID++;
  os.flush();
  // Apply label to collective's output
  producerTB->setAttr("localoutputs",
                      b.getArrayAttr({b.getStringAttr(uniqueOutputName)}));
  // Create a labeled value to use for the new block's input
  // FIXME(victory): This works only assuming that the produced result is only
  //     directly used by newTB (the block immediately following the producer).
  //     We ought to identify all blocks that use the produced local output.
  valueMaps[gpuid][oldVal] = newArg;
  newTB->setAttr("localinputs",
                 b.getArrayAttr({b.getArrayAttr(
                     {b.getStringAttr(uniqueOutputName),
                      b.getI64IntegerAttr(newArg.getArgNumber())})}));
}

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

  Lowerer lower(algo, maxNumChunks);

  // Collect a "library" of implementations for collectives
  for (Operation &op : m.body().getOps())
    if (ModuleOp submodule = dyn_cast<ModuleOp>(&op))
      lower.addSubmodule(submodule);

  for (auto &op : algo.getOps()) {
    TypeSwitch<Operation *>(&op)
        .Case<MulFOp>([&](auto mulf) {
          lower.emitCompute<MulFOp>(mulf, mulf.getOperand(0), mulf.getOperand(1));
        })
        .Case<AddFOp>([&](auto addf) {
          lower.emitCompute<AddFOp>(addf, addf.getOperand(0), addf.getOperand(1));
        })
        .Case<SubFOp>([&](auto subf) {
          lower.emitCompute<SubFOp>(subf, subf.getOperand(0), subf.getOperand(1));
        })
        .Case<math::RsqrtOp>([&](auto rsqrt) {
          lower.emitCompute<math::RsqrtOp>(rsqrt, rsqrt.getOperand());
        })
        .Case<co4hl::AllReduceOp>([&](auto ar) {
          lower.emitCollective("all_reduce", ar);
        })
        .Case<co4hl::ReduceScatterOp>([&](auto rs) {
          lower.emitCollective("reduce_scatter", rs);
        })
        .Case<co4hl::AllGatherOp>([&](auto ag) {
          lower.emitCollective("all_gather", ag);
        })
        .Case<co4hl::ReturnOp>([&](auto ret) {
          assert(&op == algo.getRegion().back().getTerminator());
          lower.finishCompute(ret.getLoc());
        })
        .Default([&](Operation *) {
          llvm::errs() << "Unexpected op: " << op << "\n";
          llvm_unreachable("Unexpected op type");
        });
  }

  lower.eraseSubmodules();
  algo->erase();
}

void registerCo4LoweringPass() {
  PassRegistration<Co4LoweringPass>();
}
