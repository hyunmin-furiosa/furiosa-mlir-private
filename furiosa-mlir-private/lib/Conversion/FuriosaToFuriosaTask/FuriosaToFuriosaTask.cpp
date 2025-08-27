#include "furiosa-mlir/Conversion/FuriosaToFuriosaTask/FuriosaToFuriosaTask.h"
#include "furiosa-mlir/Conversion/FuriosaToFuriosaTask/Utils.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"
#include "furiosa-mlir/Dialect/Task/IR/RenegadeSfr.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskDialect.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class Pass;

namespace furiosa {
#define GEN_PASS_DEF_CONVERTFURIOSATOFURIOSATASKPASS
#include "furiosa-mlir/Conversion/Passes.h.inc"

using namespace mlir;
namespace {

struct DmaOpLowering : public OpRewritePattern<furiosa::DmaOp> {
public:
  DmaOpLowering(MLIRContext *context)
      : OpRewritePattern<furiosa::DmaOp>(context) {}

  LogicalResult matchAndRewrite(furiosa::DmaOp dma_op,
                                PatternRewriter &rewriter) const final;
};

struct LoadTrfOpLowering : public OpRewritePattern<furiosa::LoadTrfOp> {
public:
  LoadTrfOpLowering(MLIRContext *context)
      : OpRewritePattern<furiosa::LoadTrfOp>(context) {}

  LogicalResult matchAndRewrite(furiosa::LoadTrfOp load_trf_op,
                                PatternRewriter &rewriter) const final;
};

struct ContractOpLowering : public OpRewritePattern<linalg::ContractOp> {
public:
  ContractOpLowering(MLIRContext *context)
      : OpRewritePattern<linalg::ContractOp>(context) {}

  LogicalResult matchAndRewrite(linalg::ContractOp contract_op,
                                PatternRewriter &rewriter) const final;
};

struct AllocOpLowering : public OpRewritePattern<furiosa::AllocOp> {
public:
  AllocOpLowering(MLIRContext *context)
      : OpRewritePattern<furiosa::AllocOp>(context) {}

  LogicalResult matchAndRewrite(furiosa::AllocOp alloc_op,
                                PatternRewriter &rewriter) const final;
};

struct DeallocOpLowering : public OpRewritePattern<furiosa::DeallocOp> {
public:
  DeallocOpLowering(MLIRContext *context)
      : OpRewritePattern<furiosa::DeallocOp>(context) {}

  LogicalResult matchAndRewrite(furiosa::DeallocOp dealloc_op,
                                PatternRewriter &rewriter) const final;
};

struct ConvertFuriosaToFuriosaTask
    : public impl::ConvertFuriosaToFuriosaTaskPassBase<
          ConvertFuriosaToFuriosaTask> {
  using Base::Base;

public:
  void runOnOperation() final;
};

} // namespace

LogicalResult DmaOpLowering::matchAndRewrite(furiosa::DmaOp dma_op,
                                             PatternRewriter &rewriter) const {
  // If source or destination is from function argument, use the Value directly.
  // Otherwise, we need to extract the address from the AllocOp.
  auto source = Value();
  auto destination = Value();
  auto source_base_attr = IntegerAttr();
  auto destination_base_attr = IntegerAttr();

  if (llvm::isa<BlockArgument>(dma_op.getSource())) {
    source = dma_op.getSource();
  } else {
    auto source_base = *getAddress(dma_op.getSource());
    auto memory_type = *getMemoryType(dma_op.getSource());
    if (memory_type == furiosa::MemoryType::dram) {
      source_base_attr = rewriter.getI64IntegerAttr(DRAM_BASE + source_base);
    } else if (memory_type == furiosa::MemoryType::sram) {
      source_base_attr = rewriter.getI64IntegerAttr(SRAM_BASE + source_base);
    } else {
      return rewriter.notifyMatchFailure(
          dma_op, "unsupported memory type for DMA operand");
    }
  }

  if (llvm::isa<BlockArgument>(dma_op.getDestination())) {
    destination = dma_op.getDestination();
  } else {
    auto destination_base = *getAddress(dma_op.getDestination());
    auto memory_type = *getMemoryType(dma_op.getDestination());
    if (memory_type == furiosa::MemoryType::dram) {
      destination_base_attr =
          rewriter.getI64IntegerAttr(DRAM_BASE + destination_base);
    } else if (memory_type == furiosa::MemoryType::sram) {
      destination_base_attr =
          rewriter.getI64IntegerAttr(SRAM_BASE + destination_base);
    } else {
      return rewriter.notifyMatchFailure(
          dma_op, "unsupported memory type for DMA operand");
    }
  }

  rewriter.setInsertionPoint(dma_op);

  std::uint64_t opcode = 0;
  std::uint64_t indirect = 0;
  auto source_limits = dma_op.getSourceLimits();
  auto source_strides = dma_op.getSourceStrides();
  auto destination_limits = dma_op.getDestinationLimits();
  auto destination_strides = dma_op.getDestinationStrides();
  auto descriptor_op = rewriter.create<task::DmaDescriptorOp>(
      dma_op.getLoc(), source, destination, opcode, indirect, source_base_attr,
      destination_base_attr, source_limits, source_strides, destination_limits,
      destination_strides);

  std::uint64_t dma_tag_id = 0;
  bool profile = 0;
  std::uint64_t profile_id = 0;
  auto dynamic_dmaw_op = rewriter.create<task::DynamicDmawOp>(
      dma_op.getLoc(), descriptor_op, ValueRange(), dma_tag_id, profile,
      profile_id);

  bool type = 1;           // 0 for exec, 1 for DMA
  bool target_context = 0; // ignored for DMA
  rewriter.create<task::tuc::WaitOp>(dma_op.getLoc(),
                                     ValueRange({dynamic_dmaw_op}), dma_tag_id,
                                     type, target_context);

  rewriter.eraseOp(dma_op);

  return success();
}

LogicalResult
LoadTrfOpLowering::matchAndRewrite(furiosa::LoadTrfOp load_trf_op,
                                   PatternRewriter &rewriter) const {
  rewriter.setInsertionPoint(load_trf_op);

  auto input_base = *getAddress(load_trf_op.getSource());
  auto input_shape = *getShape(load_trf_op.getSource());
  auto input_element_size = *getElementSize(load_trf_op.getSource());
  assert(input_shape.size() == 3 &&
         "load trf op source should be a 2D tensor with batch");
  auto output_base = *getAddress(load_trf_op.getDestination());

  std::int64_t batch = input_shape[0];
  std::int64_t dim_a = input_shape[1];
  std::int64_t dim_b = input_shape[2];
  std::int64_t size = batch * dim_a * dim_b * input_element_size;

  bool target_context = 1; // 0 for main, 1 for sub
  bool context_id = 0;     // always 0 for sub context
  std::uint64_t context_id_offset = context_id ? CONTEXT_ID_OFFSET : 0x0;
  std::uint32_t subunit_bitmap = SubUnit::DataMemorySlice |
                                 SubUnit::SubFetchUnit |
                                 SubUnit::TensorRegisterFile;
  std::uint64_t route_bitmap = DataPathUnitRoute::TensorRegisterFile;

  SmallVector<Value> mtosfr_ops;

  { // set sub fetch unit sfr
    auto sfr = task::sfr::slice::SubFetchUnit<task::sfr_data_t>();
    sfr.base = input_base;
    sfr.limits_element0 = 4;
    sfr.limits_element1 = 4;
    sfr.limits_element2 = dim_b / 32;
    sfr.limits_element3 = 2;
    sfr.limits_element4 = dim_a / 64;
    sfr.limits_element5 = 8;
    sfr.limits_element6 = 1;
    sfr.limits_element7 = 1;
    sfr.strides_element0 = 8;
    sfr.strides_element1 = dim_b * 16;
    sfr.strides_element2 = 32;
    sfr.strides_element3 = dim_b;
    sfr.strides_element4 = dim_b * 64;
    sfr.strides_element5 = dim_b * 2;
    sfr.strides_element6 = 0;
    sfr.strides_element7 = 0;
    sfr.flit_count = size / FLIT_SIZE;
    sfr.words_per_packet = SUB_FETCH_WORDS_PER_PACKET;

    auto sfr_op = createSfrSubFetchUnitOp(rewriter, load_trf_op.getLoc(), sfr);
    std::uint64_t sfr_address =
        SFR_BROADCAST + sfr.get_base() + context_id_offset;
    auto mtosfr_op = rewriter.create<task::DynamicMtosfrOp>(
        load_trf_op.getLoc(), sfr_op, ValueRange(), sfr_address);
    mtosfr_ops.push_back(mtosfr_op);
  }

  { // set tensor register file sfr
    auto sfr =
        task::sfr::slice::DotProductEngineRegisterFile<task::sfr_data_t>();
    sfr.write_mac_rows = MAC_ROWS;
    sfr.write_row_base = output_base / TENSOR_REGISTER_FILE_ROW_SIZE;
    sfr.write_row_count = size / MAC_ROWS / TENSOR_REGISTER_FILE_ROW_SIZE;
    sfr.write_flits_per_period = 1;
    sfr.write_valid_flits_per_period = 1;

    auto sfr_op =
        createSfrTensorRegisterFileOp(rewriter, load_trf_op.getLoc(), sfr);
    std::uint64_t sfr_address =
        SFR_BROADCAST + sfr.get_base() + context_id_offset;
    auto mtosfr_op = rewriter.create<task::DynamicMtosfrOp>(
        load_trf_op.getLoc(), sfr_op, ValueRange(), sfr_address);
    mtosfr_ops.push_back(mtosfr_op);
  }

  { // set sub data path unit sfr
    auto sfr = task::sfr::slice::OperationDataPath<task::sfr_data_t>();
    std::uint64_t data_path_route_sub_context = route_bitmap;
    auto sfr_op = rewriter.create<task::sfr::SfrSubDataPathUnitOp>(
        load_trf_op.getLoc(), data_path_route_sub_context);

    std::uint64_t sfr_address =
        SFR_BROADCAST + sfr.get_base() + context_id_offset;
    auto mtosfr_op = rewriter.create<task::DynamicMtosfrOp>(
        load_trf_op.getLoc(), sfr_op, ValueRange(), sfr_address);
    mtosfr_ops.push_back(mtosfr_op);
  }

  // tensor unit execution
  auto execution_op = rewriter.create<task::tuc::ExecutionOp>(
      load_trf_op.getLoc(), ValueRange(mtosfr_ops), subunit_bitmap, context_id,
      target_context);

  // wait
  std::uint64_t dma_tag_id = 0;
  bool type = 0; // 0 for exec, 1 for DMA
  rewriter.create<task::tuc::WaitOp>(load_trf_op.getLoc(),
                                     ValueRange({execution_op}), dma_tag_id,
                                     type, target_context);

  rewriter.eraseOp(load_trf_op);

  return success();
}

LogicalResult
ContractOpLowering::matchAndRewrite(linalg::ContractOp contract_op,
                                    PatternRewriter &rewriter) const {
  rewriter.setInsertionPoint(contract_op);

  assert(contract_op.getInputs().size() == 2 &&
         "contract op should have exactly two inputs");
  auto weight_base = *getAddress(contract_op.getInputs()[0]);
  auto weight_shape = *getShape(contract_op.getInputs()[0]);
  assert(weight_shape.size() == 3 &&
         "contract op weight should be a 2D tensor with batch");
  auto input_base = *getAddress(contract_op.getInputs()[1]);
  auto input_shape = *getShape(contract_op.getInputs()[1]);
  // auto input_element_size = *getElementSize(contract_op.getInputs()[1]);
  // auto input_element_type = *getElementType(contract_op.getInputs()[1]);
  assert(input_shape.size() == 3 &&
         "contract op input should be a 2D tensor with batch");
  assert(contract_op.getOutputs().size() == 1 &&
         "contract op should have exactly one output");
  auto output_base = *getAddress(contract_op.getOutputs()[0]);

  std::int64_t dim_a = weight_shape[1];
  std::int64_t dim_b = input_shape[1];
  std::int64_t dim_c = input_shape[2];

  bool context_id =
      contract_op->getAttrOfType<BoolAttr>("context_id").getValue();

  bool target_context = 0; // 0 for main, 1 for sub
  std::uint64_t context_id_offset = context_id ? CONTEXT_ID_OFFSET : 0x0;
  std::uint32_t subunit_bitmap = SubUnit::DataMemorySlice | SubUnit::FetchUnit |
                                 SubUnit::DotProductEngine |
                                 SubUnit::VectorEngine | SubUnit::CommitUnit;
  std::uint64_t route_bitmap = DataPathUnitRoute::DotProductEngine |
                               DataPathUnitRoute::VectorEngine |
                               DataPathUnitRoute::Commit;

  SmallVector<Value> mtosfr_ops;

  { // set main fetch unit sfr
    auto sfr = task::sfr::slice::FetchUnitMainContext<task::sfr_data_t>();

    sfr.base = input_base;
    sfr.fetch_size = 16; // to fit contract op
    sfr.limits_element0 = 2;
    sfr.limits_element1 = 2;
    sfr.limits_element2 = dim_b / 2;
    sfr.limits_element3 = 2;
    sfr.limits_element4 = dim_a / 64;
    sfr.limits_element5 = dim_c / 32;
    sfr.limits_element6 = 1;
    sfr.limits_element7 = 1;
    sfr.strides_element0 = dim_c;
    sfr.strides_element1 = 16;
    sfr.strides_element2 = dim_c * 2;
    sfr.strides_element3 = 0;
    sfr.strides_element4 = 0;
    sfr.strides_element5 = 32;
    sfr.strides_element6 = 0;
    sfr.strides_element7 = 0;
    sfr.flit_count = dim_a * dim_b * dim_c / 32 / FLIT_SIZE;
    sfr.words_per_packet = 4;

    auto sfr_op = createSfrMainFetchUnitOp(rewriter, contract_op.getLoc(), sfr);
    std::uint64_t sfr_address =
        SFR_BROADCAST + sfr.get_base() + context_id_offset;
    auto mtosfr_op = rewriter.create<task::DynamicMtosfrOp>(
        contract_op.getLoc(), sfr_op, ValueRange(), sfr_address);
    mtosfr_ops.push_back(mtosfr_op);
  }

  { // set dot product engine sfr
    auto sfr =
        task::sfr::slice::DotProductEngineMainContext<task::sfr_data_t>();
    sfr.reg_indexer_base = weight_base;
    sfr.acc_indexer_base = 0;
    sfr.flits_per_input = 2;
    sfr.feed_input_transpose = 1;
    sfr.initial_shift_dim = 0;
    sfr.shift_stride = 0;
    sfr.pop_dim = 1;
    sfr.shift_dim = 0;
    sfr.feed_data_type = 1;
    sfr.iter_seq_limits_element0 = 4;
    sfr.iter_seq_limits_element1 = 16;
    sfr.iter_seq_limits_element2 = dim_b / 16;
    sfr.iter_seq_limits_element3 = dim_a / 64;
    sfr.iter_seq_limits_element4 = dim_c / 32;
    sfr.iter_seq_limits_element5 = 1;
    sfr.iter_seq_limits_element6 = 1;
    sfr.iter_seq_limits_element7 = 1;
    sfr.reg_indexer_strides_element0 = 32;
    sfr.reg_indexer_strides_element1 = 2;
    sfr.reg_indexer_strides_element2 = 128;
    sfr.reg_indexer_strides_element3 = 128 * dim_b / 16;
    sfr.reg_indexer_strides_element4 = 0;
    sfr.reg_indexer_strides_element5 = 0;
    sfr.reg_indexer_strides_element6 = 0;
    sfr.reg_indexer_strides_element7 = 0;
    sfr.acc_indexer_strides_element0 = 1;
    sfr.acc_indexer_strides_element1 = 0;
    sfr.acc_indexer_strides_element2 = 0;
    sfr.acc_indexer_strides_element3 = 0;
    sfr.acc_indexer_strides_element4 = 0;
    sfr.acc_indexer_strides_element5 = 0;
    sfr.acc_indexer_strides_element6 = 0;
    sfr.acc_indexer_strides_element7 = 0;
    sfr.acc_limit = dim_b / 2;
    sfr.acc_cols = 32;
    sfr.output_major = 0;
    sfr.mac_tree_depth = 1;
    sfr.mac_type = 1;
    sfr.mac_rows = MAC_ROWS;
    sfr.data_type = 2;
    sfr.reg_read_log_size = 1;

    auto sfr_op =
        createSfrDotProductEngineOp(rewriter, contract_op.getLoc(), sfr);
    std::uint64_t sfr_address =
        SFR_BROADCAST + sfr.get_base() + context_id_offset;
    auto mtosfr_op = rewriter.create<task::DynamicMtosfrOp>(
        contract_op.getLoc(), sfr_op, ValueRange(), sfr_address);
    mtosfr_ops.push_back(mtosfr_op);
  }

  { // set vector engine sfr
    auto sfr = task::sfr::slice::VectorRouteUnitMainContext<task::sfr_data_t>();
    sfr.route_info_data_out_source = 1;
    sfr.cast_compaction_mode = 2;
    sfr.compaction_mode_cast_compaction_count = 4;

    auto sfr_op =
        createSfrVectorRouteUnitOp(rewriter, contract_op.getLoc(), sfr);
    std::uint64_t sfr_address =
        SFR_BROADCAST + sfr.get_base() + context_id_offset;
    auto mtosfr_op = rewriter.create<task::DynamicMtosfrOp>(
        contract_op.getLoc(), sfr_op, ValueRange(), sfr_address);
    mtosfr_ops.push_back(mtosfr_op);
  }

  { // set main commit unit sfr
    auto sfr = task::sfr::slice::CommitUnitMainContext<task::sfr_data_t>();
    sfr.base = output_base;
    sfr.commit_in_size = 32;
    sfr.commit_size = 32;
    sfr.limits_element0 = 8;
    sfr.limits_element1 = 4;
    sfr.limits_element2 = 2;
    sfr.limits_element3 = dim_a / 64;
    sfr.limits_element4 = dim_c / 32;
    sfr.limits_element5 = 1;
    sfr.limits_element6 = 1;
    sfr.limits_element7 = 1;
    sfr.strides_element0 = dim_c * 2;
    sfr.strides_element1 = dim_c * 16;
    sfr.strides_element2 = dim_c;
    sfr.strides_element3 = dim_c * 64;
    sfr.strides_element4 = 32;
    sfr.strides_element5 = 0;
    sfr.strides_element6 = 0;
    sfr.strides_element7 = 0;
    sfr.commit_unit_slice_enable_bitmap0 = 0xffffffffffffffff;
    sfr.commit_unit_slice_enable_bitmap1 = 0xffffffffffffffff;
    sfr.commit_unit_slice_enable_bitmap2 = 0xffffffffffffffff;
    sfr.commit_unit_slice_enable_bitmap3 = 0xffffffffffffffff;

    auto sfr_op =
        createSfrMainCommitUnitOp(rewriter, contract_op.getLoc(), sfr);
    std::uint64_t sfr_address =
        SFR_BROADCAST + sfr.get_base() + context_id_offset;
    auto mtosfr_op = rewriter.create<task::DynamicMtosfrOp>(
        contract_op.getLoc(), sfr_op, ValueRange(), sfr_address);
    mtosfr_ops.push_back(mtosfr_op);
  }

  { // set main data path unit sfr
    auto sfr =
        task::sfr::slice::OperationDataPathMainContext<task::sfr_data_t>();
    uint64_t main_context = route_bitmap;
    uint64_t channel_config = 0;
    auto sfr_op = rewriter.create<task::sfr::SfrMainDataPathUnitOp>(
        contract_op.getLoc(), main_context, channel_config);

    std::uint64_t sfr_address =
        SFR_BROADCAST + sfr.get_base() + context_id_offset;
    auto mtosfr_op = rewriter.create<task::DynamicMtosfrOp>(
        contract_op.getLoc(), sfr_op, ValueRange(), sfr_address);
    mtosfr_ops.push_back(mtosfr_op);
  }

  // tensor unit execution
  auto execution_op = rewriter.create<task::tuc::ExecutionOp>(
      contract_op.getLoc(), ValueRange(mtosfr_ops), subunit_bitmap, context_id,
      target_context);

  // wait
  std::uint64_t dma_tag_id = 0;
  bool type = 0; // 0 for exec, 1 for DMA
  rewriter.create<task::tuc::WaitOp>(contract_op.getLoc(),
                                     ValueRange({execution_op}), dma_tag_id,
                                     type, target_context);

  rewriter.replaceAllUsesWith(contract_op.getResults(),
                              contract_op.getOutputs());

  rewriter.eraseOp(contract_op);

  return success();
}

LogicalResult
AllocOpLowering::matchAndRewrite(furiosa::AllocOp alloc_op,
                                 PatternRewriter &rewriter) const {
  if (alloc_op->getParentOfType<func::FuncOp>()->hasAttr("target")) {
    rewriter.eraseOp(alloc_op);
  }

  return success();
}

LogicalResult
DeallocOpLowering::matchAndRewrite(furiosa::DeallocOp dealloc_op,
                                   PatternRewriter &rewriter) const {
  if (dealloc_op->getParentOfType<func::FuncOp>()->hasAttr("target")) {
    rewriter.eraseOp(dealloc_op);
  }

  return success();
}

void ConvertFuriosaToFuriosaTask::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<task::TaskDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<DmaOpLowering>(patterns.getContext());
  patterns.add<LoadTrfOpLowering>(patterns.getContext());
  patterns.add<ContractOpLowering>(patterns.getContext());
  patterns.add<AllocOpLowering>(patterns.getContext());
  patterns.add<DeallocOpLowering>(patterns.getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

} // namespace furiosa
} // namespace mlir
