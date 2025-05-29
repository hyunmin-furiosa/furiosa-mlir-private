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
  auto tensor_type = llvm::dyn_cast_or_null<RankedTensorType>(
      load_trf_op.getSource().getType());
  auto size = tensor_type.getNumElements() *
              tensor_type.getElementTypeBitWidth() / CHAR_BIT;

  auto output_base = *getAddress(load_trf_op.getDestination());

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
    std::uint64_t base = input_base;
    std::uint64_t type_conversion = 0;
    std::uint64_t num_zero_points = 0;
    std::uint64_t zero_point0 = 0;
    std::uint64_t zero_point1 = 0;
    ArrayAttr limits = rewriter.getI64ArrayAttr({4, 8, 1, 1, 1, 1, 1, 1});
    ArrayAttr strides = rewriter.getI64ArrayAttr({8, 32, 0, 0, 0, 0, 0, 0});
    std::uint64_t flit_count = 1;
    std::uint64_t words_per_packet = SUB_FETCH_WORDS_PER_PACKET;
    std::uint64_t topology = 0;
    std::uint64_t outer_slice_log_size = 0;
    std::uint64_t outer_dim0_log_size = 0;
    std::uint64_t outer_dim1_log_size = 0;
    std::uint64_t outer_dim0_chunk_size = 0;
    std::uint64_t outer_dim1_chunk_size = 0;
    ArrayAttr custom_snoop_bitmap_mask = nullptr;
    auto sfr_op = rewriter.create<task::sfr::SfrSubFetchUnitOp>(
        load_trf_op.getLoc(), base, type_conversion, num_zero_points,
        zero_point0, zero_point1, limits, strides, flit_count, words_per_packet,
        topology, outer_slice_log_size, outer_dim0_log_size,
        outer_dim1_log_size, outer_dim0_chunk_size, outer_dim1_chunk_size,
        custom_snoop_bitmap_mask);

    std::uint64_t sfr_address =
        SFR_BROADCAST + sfr.get_base() + context_id_offset;
    auto mtosfr_op = rewriter.create<task::DynamicMtosfrOp>(
        load_trf_op.getLoc(), sfr_op, ValueRange(), sfr_address);
    mtosfr_ops.push_back(mtosfr_op);
  }

  { // set tensor register file sfr
    auto sfr =
        task::sfr::slice::DotProductEngineRegisterFile<task::sfr_data_t>();
    std::uint64_t write_interleaving_flit_count = 0;
    std::uint64_t write_mode = 0;
    std::uint64_t write_mac_rows = 8;
    std::uint64_t write_skip_flit_count = 0;
    std::uint64_t write_row_base = output_base / TENSOR_REGISTER_FILE_ROW_SIZE;
    std::uint64_t write_mac_row_interleaving = 0;
    std::uint64_t write_row_count = 1;
    std::uint64_t write_flits_per_period = 1;
    std::uint64_t write_valid_flits_per_period = 1;
    auto sfr_op = rewriter.create<task::sfr::SfrTensorRegisterFileOp>(
        load_trf_op.getLoc(), write_interleaving_flit_count, write_mode,
        write_mac_rows, write_skip_flit_count, write_row_base,
        write_mac_row_interleaving, write_row_count, write_flits_per_period,
        write_valid_flits_per_period);

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
  auto input0_base = *getAddress(contract_op.getInputs()[0]);
  auto input1_base = *getAddress(contract_op.getInputs()[1]);
  assert(contract_op.getOutputs().size() == 1 &&
         "contract op should have exactly one output");
  auto output_base = *getAddress(contract_op.getOutputs()[0]);
  bool context_id =
      contract_op->getAttrOfType<BoolAttr>("context_id").getValue();

  bool target_context = 0; // 0 for main, 1 for sub
  std::uint64_t context_id_offset = context_id ? CONTEXT_ID_OFFSET : 0x0;
  std::uint32_t subunit_bitmap = SubUnit::DataMemorySlice | SubUnit::FetchUnit |
                                 SubUnit::DotProductEngine |
                                 SubUnit::CommitUnit;
  std::uint64_t route_bitmap =
      DataPathUnitRoute::DotProductEngine | DataPathUnitRoute::Commit;

  SmallVector<Value> mtosfr_ops;

  { // set main fetch unit sfr
    auto sfr = task::sfr::slice::FetchUnitMainContext<task::sfr_data_t>();
    std::uint64_t fetch_mode = 0;
    std::uint64_t num_zero_points = 0;
    std::uint64_t zero_point0 = 0;
    std::uint64_t zero_point1 = 0;
    std::uint64_t table_entry_size = 0;
    std::uint64_t tables = 0;
    std::uint64_t indirect_base = 0;
    std::uint64_t indirect_dim = 0;
    std::uint64_t table_base_mode = 0;
    std::uint64_t indirect_pointer_size = 0;
    std::uint64_t zeropoint_tail_mode = 0;
    std::uint64_t last_dim_pad_value = 0;
    std::uint64_t last_dim = 0;
    std::uint64_t pad_order = 0;
    std::uint64_t last_dim_rightmost_valid_count_dim = 0;
    std::uint64_t last_dim_left_pad_count = 0;
    std::uint64_t type_conversion = 0;
    std::uint64_t last_dim_left_pad_mode = 0;
    std::uint64_t zeropoint_dims = 0;
    ArrayAttr last_dim_rightmost_valid_count = nullptr;
    std::uint64_t base = input0_base;
    std::uint64_t fetch_size = 0;
    ArrayAttr limits = rewriter.getI64ArrayAttr({1});
    ArrayAttr strides = rewriter.getI64ArrayAttr({0});
    std::uint64_t flit_count = 0;
    std::uint64_t words_per_packet = 0;
    std::uint64_t zeropoint_fetch_limit = 0;
    std::uint64_t topology = 0;
    std::uint64_t channel_config = 0;
    std::uint64_t outer_slice_log_size = 0;
    std::uint64_t outer_dim0_log_size = 0;
    std::uint64_t outer_dim1_log_size = 0;
    std::uint64_t outer_dim0_chunk_size = 0;
    std::uint64_t outer_dim1_chunk_size = 0;
    ArrayAttr custom_snoop_bitmap_mask = nullptr;

    auto sfr_op = rewriter.create<task::sfr::SfrMainFetchUnitOp>(
        contract_op.getLoc(), fetch_mode, num_zero_points, zero_point0,
        zero_point1, table_entry_size, tables, indirect_base, indirect_dim,
        table_base_mode, indirect_pointer_size, zeropoint_tail_mode,
        last_dim_pad_value, last_dim, pad_order,
        last_dim_rightmost_valid_count_dim, last_dim_left_pad_count,
        type_conversion, last_dim_left_pad_mode, zeropoint_dims,
        last_dim_rightmost_valid_count, base, fetch_size, limits, strides,
        flit_count, words_per_packet, zeropoint_fetch_limit, topology,
        channel_config, outer_slice_log_size, outer_dim0_log_size,
        outer_dim1_log_size, outer_dim0_chunk_size, outer_dim1_chunk_size,
        custom_snoop_bitmap_mask);

    std::uint64_t sfr_address =
        SFR_BROADCAST + sfr.get_base() + context_id_offset;
    auto mtosfr_op = rewriter.create<task::DynamicMtosfrOp>(
        contract_op.getLoc(), sfr_op, ValueRange(), sfr_address);
    mtosfr_ops.push_back(mtosfr_op);
  }

  { // set dot product engine sfr
    auto sfr =
        task::sfr::slice::DotProductEngineMainContext<task::sfr_data_t>();
    std::uint64_t reg_indexer_base = 1;
    std::uint64_t acc_indexer_base = 0;
    std::uint64_t flits_per_input = 0;
    std::uint64_t feed_input_transpose = 0;
    std::uint64_t initial_shift_dim = 0;
    std::uint64_t shift_stride = 0;
    std::uint64_t pop_dim = 0;
    std::uint64_t shift_dim = 0;
    std::uint64_t channel_config = 0;
    std::uint64_t feed_data_type = 0;
    ArrayAttr initial_shift = nullptr;
    ArrayAttr iter_seq_limits = nullptr;
    ArrayAttr reg_indexer_strides = rewriter.getI64ArrayAttr({0});
    ArrayAttr acc_indexer_strides = rewriter.getI64ArrayAttr({0});
    std::uint64_t acc_limit = 0;
    std::uint64_t acc_cols = 0;
    std::uint64_t acc_reset = 0;
    std::uint64_t output_major = 0;
    std::uint64_t acc_init_value = 0;
    std::uint64_t mac_tree_operation = 0;
    std::uint64_t mac_tree_depth = 0;
    std::uint64_t mac_type = 0;
    std::uint64_t mac_rows = 0;
    std::uint64_t fp_ieee_nan_multiplication = 0;
    std::uint64_t fxp_shift_rounding_mode = 0;
    std::uint64_t data_type = 0;
    std::uint64_t reg_read_log_size = 0;
    std::uint64_t reg_read_mode = 0;
    std::uint64_t reg_read_cache_mode = 0;

    auto sfr_op = rewriter.create<task::sfr::SfrDotProductEngineOp>(
        contract_op.getLoc(), reg_indexer_base, acc_indexer_base,
        flits_per_input, feed_input_transpose, initial_shift_dim, shift_stride,
        pop_dim, shift_dim, channel_config, feed_data_type, initial_shift,
        iter_seq_limits, reg_indexer_strides, acc_indexer_strides, acc_limit,
        acc_cols, acc_reset, output_major, acc_init_value, mac_tree_operation,
        mac_tree_depth, mac_type, mac_rows, fp_ieee_nan_multiplication,
        fxp_shift_rounding_mode, data_type, reg_read_log_size, reg_read_mode,
        reg_read_cache_mode);

    std::uint64_t sfr_address =
        SFR_BROADCAST + sfr.get_base() + context_id_offset;
    auto mtosfr_op = rewriter.create<task::DynamicMtosfrOp>(
        contract_op.getLoc(), sfr_op, ValueRange(), sfr_address);
    mtosfr_ops.push_back(mtosfr_op);
  }

  { // set main commit unit sfr
    auto sfr = task::sfr::slice::CommitUnitMainContext<task::sfr_data_t>();
    std::uint64_t type_conversion = 0;
    std::uint64_t base = output_base;
    std::uint64_t commit_in_size = 0;
    std::uint64_t commit_size = 0;
    ArrayAttr limits = rewriter.getI64ArrayAttr({1});
    ArrayAttr strides = rewriter.getI64ArrayAttr({1});
    ArrayAttr slice_enable_bitmap_mask = nullptr;

    auto sfr_op = rewriter.create<task::sfr::SfrMainCommitUnitOp>(
        contract_op.getLoc(), type_conversion, base, commit_in_size,
        commit_size, limits, strides, slice_enable_bitmap_mask);

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
