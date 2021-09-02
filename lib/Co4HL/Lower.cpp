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
  // map from old values in high-level algo to newly generated values in
  // low-level threadblocks.
  SmallVector<DenseMap<Value, Value>, 16> valueMaps;

  // A collection of possible implementations of collectives.
  // TODO: replace this with some more flexible SCCL/ncclize-type compilation
  // that can generate appropriate implementations on the fly?
  SmallVector<ModuleOp, 16> submodules;

  unsigned uniqueOutputID = 0;

public:
  Lowerer(int numGPUs, unsigned numBuffers, int maxNumChunks,
                  ModuleOp m)
      : numGPUs(numGPUs), numBuffers(numBuffers), maxNumChunks(maxNumChunks),
        m(m), ctx(m->getContext()), elemTy(FloatType::getF32(ctx)),
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
      VectorType loweredArgTy = VectorType::get({maxNumChunks}, elemTy);
      while (b.getNumArguments() < numBuffers)
        b.addArgument(loweredArgTy);
      builders.back().setInsertionPoint(&b, b.begin());
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

  // Emit a collective, which performs inter-GPU communication
  void emitCollective(StringRef collectiveName, Value oldVal) {
    assert(oldVal.isa<OpResult>());
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
      co4ll::TBOp tb = cast<co4ll::TBOp>(&gpuBody.front());
      const int gpuid = cast<co4ll::GPUOp>(newGpu).gpuid();

      OpBuilder& b = builders[gpuid];

      // End the thread block that previous compute ops were generated in.
      // TODO: set return value
      b.create<co4ll::ReturnOp>(m.getLoc(), llvm::None);

      // Start a new threadblock into which we will emit subsequent compute ops.
      // This new threadblock goes right after the previous one.
      Operation *prevTB = b.getBlock()->getParentOp();
      b.setInsertionPointAfter(prevTB);
      // TODO: set return type
      co4ll::TBOp newTB = b.create<co4ll::TBOp>(m.getLoc(), llvm::None);
      assert(newTB.getRegion().empty());
      Block &newTBBlock = newTB.getRegion().emplaceBlock();
      VectorType loweredArgTy = VectorType::get({maxNumChunks}, elemTy);
      while (newTBBlock.getNumArguments() < numBuffers)
        newTBBlock.addArgument(loweredArgTy);
      b.setInsertionPoint(&newTBBlock, newTBBlock.begin());

      // Because co4ll::GPUOp has trait IsolatedFromAbove, it is not legal for
      // ops nested inside one gpu to directly use SSA Values from another gpu.
      // So we'll have the LinkByGPUID pass merge gpu ops and then the
      // ThreadblockSSA pass can link up the use of values within one rank.
      // To that end, we generate a unique label for the output of the
      // collective which will tell ThreadblockSSA what connections to make.
      std::string uniqueOutputName;
      llvm::raw_string_ostream os(uniqueOutputName);
      os << collectiveName << "Output" << uniqueOutputID++;
      os.flush();
      // Apply label to collective's output
      tb->setAttr("localoutputs",
                  b.getArrayAttr({b.getStringAttr(uniqueOutputName)}));
      // Create a labeled value to use for the compute-block's input
      // FIXME(victory): This works by assuming that the collective's result
      // is only directly used by the compute operations immediately following
      // that collective and preceeding the next collective in the algo.
      // We really should identify all the blocks that use the collective's result.
      // As a minor optimization, for communicating the result of a collective
      // between threadblocks, we can avoid the need to refer to a whole buffer
      // and have the user use an extract_strided_slice op if the number of
      // chunks is less than a whole buffer, by just directly adding an
      // additional argument to communicate a smaller number of chunks.
      BlockArgument newArg =
          newTBBlock.addArgument(VectorType::get({chunks}, elemTy));
      valueMaps[gpuid][oldVal] = newArg;
      newTB->setAttr("localinputs",
                     b.getArrayAttr({b.getArrayAttr(
                         {b.getStringAttr(uniqueOutputName),
                          b.getI64IntegerAttr(newArg.getArgNumber())})}));
    }
  }

private:
  // Note this template is explicitly specialized below.
  template <class T>
  T map(int gpuid, T x) {
    return x;
  }
};
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

  Lowerer lower(algo.numgpus(), algo.numbufs(), maxNumChunks, m);

  // Collect a "library" of implementations for collectives
  for (Operation &op : m.body().getOps())
    if (ModuleOp submodule = dyn_cast<ModuleOp>(&op))
      lower.addSubmodule(submodule);

  for (auto &op : algo.getOps()) {
    TypeSwitch<Operation *>(&op)
        .Case<MulFOp>([&](auto mulf) {
          lower.create<MulFOp>(mulf, mulf.getOperand(0), mulf.getOperand(1));
        })
        .Case<AddFOp>([&](auto addf) {
          lower.create<AddFOp>(addf, addf.getOperand(0), addf.getOperand(1));
        })
        .Case<SubFOp>([&](auto subf) {
          lower.create<SubFOp>(subf, subf.getOperand(0), subf.getOperand(1));
        })
        .Case<math::RsqrtOp>([&](auto rsqrt) {
          lower.create<math::RsqrtOp>(rsqrt, rsqrt.getOperand());
        })
        .Case<co4hl::AllReduceOp>([&](auto ar) {
          lower.emitCollective("all_reduce", ar.res());
        })
        .Case<co4hl::ReduceScatterOp>([&](auto rs) {
          lower.emitCollective("reduce_scatter", rs.res());
        })
        .Case<co4hl::AllGatherOp>([&](auto ag) {
          lower.emitCollective("all_gather", ag.res());
        })
        .Case<co4hl::ReturnOp>([&](auto ret) {
          // TODO: Set return value
          lower.create<co4ll::ReturnOp>(ret, llvm::None);
        })
        .Default([&](Operation *op) {
          llvm::errs() << "Unexpected op: " << *op << "\n";
          llvm_unreachable("Unexpected op type");
        });
  }

  lower.eraseSubmodules();
  algo->erase();
}

void registerCo4LoweringPass() {
  PassRegistration<Co4LoweringPass>();
}
