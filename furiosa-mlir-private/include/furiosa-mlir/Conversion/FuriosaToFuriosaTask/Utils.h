#pragma once

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/Utils.h"
#include "furiosa-mlir/Dialect/Task/IR/RenegadeSfr.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::furiosa {

static constexpr std::uint64_t SFR_BROADCAST = 0xff0000;
static constexpr std::uint64_t CONTEXT_ID_OFFSET = 0x800;
static constexpr std::uint64_t MAIN_FETCH_SIZE = 0x20;
static constexpr std::uint64_t SUB_FETCH_SIZE = 0x8;
static constexpr std::uint64_t TENSOR_REGISTER_FILE_ROW_SIZE = 0x20;
static constexpr std::uint64_t SUB_FETCH_WORDS_PER_PACKET = 4;
static constexpr std::uint64_t DRAM_BASE = 0xc000000000;
static constexpr std::uint64_t SRAM_BASE = 0x10000000;
static constexpr std::uint64_t MAC_ROWS = 8;
static constexpr std::uint64_t FLIT_SIZE = 32;

struct DataPathUnitRoute {
  static constexpr std::uint64_t DotProductEngine = 1 << 0;
  static constexpr std::uint64_t VectorEngine = 1 << 1;
  static constexpr std::uint64_t TransposeEngine = 1 << 2;
  static constexpr std::uint64_t Commit = 1 << 3;
  static constexpr std::uint64_t TensorRegisterFile = 1 << 4;
  static constexpr std::uint64_t VectorRegisterFile = 1 << 5;
  static constexpr std::uint64_t RegisterConfigUnit = 1 << 6;
};

struct SubUnit {
  static constexpr std::uint64_t DataMemorySlice = 1 << 0;
  static constexpr std::uint64_t FetchUnit = 1 << 1;
  static constexpr std::uint64_t DotProductEngine = 1 << 2;
  static constexpr std::uint64_t VectorEngine = 1 << 3;
  static constexpr std::uint64_t TransposeEngine = 1 << 4;
  static constexpr std::uint64_t CommitUnit = 1 << 5;
  static constexpr std::uint64_t SubFetchUnit = 1 << 6;
  static constexpr std::uint64_t SubCommitUnit = 1 << 7;
  static constexpr std::uint64_t TensorRegisterFile = 1 << 8;
  static constexpr std::uint64_t VectorRegisterFile = 1 << 9;
  static constexpr std::uint64_t RegisterConfig = 1 << 10;
};

enum class MacTreeOperation { Add = 0, Max = 1 };
enum class MacType : int {
  Int4 = 0,
  Int8 = 1,
  Fp8e4m3 = 2,
  Fp8e5m2 = 3,
  Bf16 = 4,
};

FailureOr<std::int64_t> getAddress(Value value) {
  auto defining_op = value.getDefiningOp();
  furiosa::AllocOp alloc_op;
  if (llvm::isa<furiosa::AllocOp>(defining_op)) {
    alloc_op = llvm::dyn_cast_or_null<furiosa::AllocOp>(defining_op);
  } else if (llvm::isa<linalg::ContractOp>(defining_op)) {
    auto contract_op = llvm::dyn_cast_or_null<linalg::ContractOp>(defining_op);
    assert(contract_op.getOutputs().size() == 1 &&
           "contract op should have exactly one output");
    auto contract_output_op = contract_op.getOutputs()[0].getDefiningOp();
    if (llvm::isa<furiosa::AllocOp>(contract_output_op)) {
      alloc_op = llvm::dyn_cast_or_null<furiosa::AllocOp>(contract_output_op);
    }
  }
  if (alloc_op && alloc_op->hasAttr("address")) {
    return alloc_op->getAttrOfType<IntegerAttr>("address").getInt();
  } else {
    return failure();
  }
}

FailureOr<SmallVector<std::int64_t>> getShape(Value value) {
  auto tensor_type = llvm::dyn_cast_or_null<RankedTensorType>(value.getType());
  if (!tensor_type) {
    return failure();
  }

  return SmallVector<std::int64_t>(tensor_type.getShape());
}

FailureOr<std::uint64_t> getElementSize(Value value) {
  auto tensor_type = llvm::dyn_cast_or_null<RankedTensorType>(value.getType());
  if (!tensor_type) {
    return failure();
  }

  return tensor_type.getElementTypeBitWidth() / CHAR_BIT;
}

FailureOr<MacType> getElementType(Value value) {
  auto tensor_type = llvm::dyn_cast_or_null<RankedTensorType>(value.getType());
  if (!tensor_type) {
    return failure();
  }
  auto element_type = tensor_type.getElementType();
  if (auto integer_type = llvm::dyn_cast_or_null<IntegerType>(element_type)) {
    if (integer_type.getWidth() == 4) {
      return MacType::Int4;
    } else if (integer_type.getWidth() == 8) {
      return MacType::Int8;
    }
  } else if (llvm::isa<Float8E4M3FNUZType>(element_type)) {
    return MacType::Fp8e4m3;
  } else if (llvm::isa<Float8E5M2FNUZType>(element_type)) {
    return MacType::Fp8e5m2;
  } else if (llvm::isa<BFloat16Type>(element_type)) {
    return MacType::Bf16;
  }

  return failure();
}

task::sfr::SfrMainFetchUnitOp createSfrMainFetchUnitOp(
    PatternRewriter &rewriter, Location loc,
    task::sfr::slice::FetchUnitMainContext<task::sfr_data_t> &sfr) {
  ArrayAttr last_dim_rightmost_valid_count = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.last_dim_rightmost_valid_count0),
       static_cast<std::int64_t>(sfr.last_dim_rightmost_valid_count1),
       static_cast<std::int64_t>(sfr.last_dim_rightmost_valid_count2),
       static_cast<std::int64_t>(sfr.last_dim_rightmost_valid_count3),
       static_cast<std::int64_t>(sfr.last_dim_rightmost_valid_count4),
       static_cast<std::int64_t>(sfr.last_dim_rightmost_valid_count5),
       static_cast<std::int64_t>(sfr.last_dim_rightmost_valid_count6),
       static_cast<std::int64_t>(sfr.last_dim_rightmost_valid_count7)});
  ArrayAttr limits = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.limits_element0),
       static_cast<std::int64_t>(sfr.limits_element1),
       static_cast<std::int64_t>(sfr.limits_element2),
       static_cast<std::int64_t>(sfr.limits_element3),
       static_cast<std::int64_t>(sfr.limits_element4),
       static_cast<std::int64_t>(sfr.limits_element5),
       static_cast<std::int64_t>(sfr.limits_element6),
       static_cast<std::int64_t>(sfr.limits_element7)});
  ArrayAttr strides = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.strides_element0),
       static_cast<std::int64_t>(sfr.strides_element1),
       static_cast<std::int64_t>(sfr.strides_element2),
       static_cast<std::int64_t>(sfr.strides_element3),
       static_cast<std::int64_t>(sfr.strides_element4),
       static_cast<std::int64_t>(sfr.strides_element5),
       static_cast<std::int64_t>(sfr.strides_element6),
       static_cast<std::int64_t>(sfr.strides_element7)});
  ArrayAttr custom_snoop_bitmap_mask = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.custom_snoop_bitmap_mask_element0),
       static_cast<std::int64_t>(sfr.custom_snoop_bitmap_mask_element1),
       static_cast<std::int64_t>(sfr.custom_snoop_bitmap_mask_element2),
       static_cast<std::int64_t>(sfr.custom_snoop_bitmap_mask_element3)});

  return rewriter.create<task::sfr::SfrMainFetchUnitOp>(
      loc, sfr.fetch_mode, sfr.num_zero_points, sfr.zero_point0,
      sfr.zero_point1, sfr.table_entry_size, sfr.tables, sfr.indirect_base,
      sfr.indirect_dim, sfr.table_base_mode, sfr.indirect_pointer_size,
      sfr.zeropoint_tail_mode, sfr.last_dim_pad_value, sfr.last_dim,
      sfr.pad_order, sfr.last_dim_rightmost_valid_count_dim,
      sfr.last_dim_left_pad_count, sfr.type_conversion,
      sfr.last_dim_left_pad_mode, sfr.zeropoint_dims,
      last_dim_rightmost_valid_count, sfr.base, sfr.fetch_size, limits, strides,
      sfr.flit_count, sfr.words_per_packet, sfr.zeropoint_fetch_limit,
      sfr.topology, sfr.channel_config, sfr.outer_slice_log_size,
      sfr.outer_dim0_log_size, sfr.outer_dim1_log_size,
      sfr.outer_dim0_chunk_size, sfr.outer_dim1_chunk_size,
      custom_snoop_bitmap_mask);
}

task::sfr::SfrSubFetchUnitOp
createSfrSubFetchUnitOp(PatternRewriter &rewriter, Location loc,
                        task::sfr::slice::SubFetchUnit<task::sfr_data_t> &sfr) {
  ArrayAttr limits = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.limits_element0),
       static_cast<std::int64_t>(sfr.limits_element1),
       static_cast<std::int64_t>(sfr.limits_element2),
       static_cast<std::int64_t>(sfr.limits_element3),
       static_cast<std::int64_t>(sfr.limits_element4),
       static_cast<std::int64_t>(sfr.limits_element5),
       static_cast<std::int64_t>(sfr.limits_element6),
       static_cast<std::int64_t>(sfr.limits_element7)});
  ArrayAttr strides = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.strides_element0),
       static_cast<std::int64_t>(sfr.strides_element1),
       static_cast<std::int64_t>(sfr.strides_element2),
       static_cast<std::int64_t>(sfr.strides_element3),
       static_cast<std::int64_t>(sfr.strides_element4),
       static_cast<std::int64_t>(sfr.strides_element5),
       static_cast<std::int64_t>(sfr.strides_element6),
       static_cast<std::int64_t>(sfr.strides_element7)});
  ArrayAttr custom_snoop_bitmap_mask = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.custom_snoop_bitmap_mask_element0),
       static_cast<std::int64_t>(sfr.custom_snoop_bitmap_mask_element1),
       static_cast<std::int64_t>(sfr.custom_snoop_bitmap_mask_element2),
       static_cast<std::int64_t>(sfr.custom_snoop_bitmap_mask_element3)});

  return rewriter.create<task::sfr::SfrSubFetchUnitOp>(
      loc, sfr.base, sfr.type_conversion, sfr.num_zero_points, sfr.zero_point0,
      sfr.zero_point1, limits, strides, sfr.flit_count, sfr.words_per_packet,
      sfr.topology, sfr.outer_slice_log_size, sfr.outer_dim0_log_size,
      sfr.outer_dim1_log_size, sfr.outer_dim0_chunk_size,
      sfr.outer_dim1_chunk_size, custom_snoop_bitmap_mask);
}

task::sfr::SfrDotProductEngineOp createSfrDotProductEngineOp(
    PatternRewriter &rewriter, Location loc,
    task::sfr::slice::DotProductEngineMainContext<task::sfr_data_t> &sfr) {
  ArrayAttr initial_shift = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.initial_shift_element0),
       static_cast<std::int64_t>(sfr.initial_shift_element1),
       static_cast<std::int64_t>(sfr.initial_shift_element2),
       static_cast<std::int64_t>(sfr.initial_shift_element3),
       static_cast<std::int64_t>(sfr.initial_shift_element4),
       static_cast<std::int64_t>(sfr.initial_shift_element5),
       static_cast<std::int64_t>(sfr.initial_shift_element6),
       static_cast<std::int64_t>(sfr.initial_shift_element7)});
  ArrayAttr iter_seq_limits = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.iter_seq_limits_element0),
       static_cast<std::int64_t>(sfr.iter_seq_limits_element1),
       static_cast<std::int64_t>(sfr.iter_seq_limits_element2),
       static_cast<std::int64_t>(sfr.iter_seq_limits_element3),
       static_cast<std::int64_t>(sfr.iter_seq_limits_element4),
       static_cast<std::int64_t>(sfr.iter_seq_limits_element5),
       static_cast<std::int64_t>(sfr.iter_seq_limits_element6),
       static_cast<std::int64_t>(sfr.iter_seq_limits_element7)});
  ArrayAttr reg_indexer_strides = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.reg_indexer_strides_element0),
       static_cast<std::int64_t>(sfr.reg_indexer_strides_element1),
       static_cast<std::int64_t>(sfr.reg_indexer_strides_element2),
       static_cast<std::int64_t>(sfr.reg_indexer_strides_element3),
       static_cast<std::int64_t>(sfr.reg_indexer_strides_element4),
       static_cast<std::int64_t>(sfr.reg_indexer_strides_element5),
       static_cast<std::int64_t>(sfr.reg_indexer_strides_element6),
       static_cast<std::int64_t>(sfr.reg_indexer_strides_element7)});
  ArrayAttr acc_indexer_strides = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.acc_indexer_strides_element0),
       static_cast<std::int64_t>(sfr.acc_indexer_strides_element1),
       static_cast<std::int64_t>(sfr.acc_indexer_strides_element2),
       static_cast<std::int64_t>(sfr.acc_indexer_strides_element3),
       static_cast<std::int64_t>(sfr.acc_indexer_strides_element4),
       static_cast<std::int64_t>(sfr.acc_indexer_strides_element5),
       static_cast<std::int64_t>(sfr.acc_indexer_strides_element6),
       static_cast<std::int64_t>(sfr.acc_indexer_strides_element7)});

  return rewriter.create<task::sfr::SfrDotProductEngineOp>(
      loc, sfr.reg_indexer_base, sfr.acc_indexer_base, sfr.flits_per_input,
      sfr.feed_input_transpose, sfr.initial_shift_dim, sfr.shift_stride,
      sfr.pop_dim, sfr.shift_dim, sfr.channel_config, sfr.feed_data_type,
      initial_shift, iter_seq_limits, reg_indexer_strides, acc_indexer_strides,
      sfr.acc_limit, sfr.acc_cols, sfr.acc_reset, sfr.output_major,
      sfr.acc_init_value, sfr.mac_tree_operation, sfr.mac_tree_depth,
      sfr.mac_type, sfr.mac_rows, sfr.fp_ieee_nan_multiplication,
      sfr.fxp_shift_rounding_mode, sfr.data_type, sfr.reg_read_log_size,
      sfr.reg_read_mode, sfr.reg_read_cache_mode);
}

task::sfr::SfrTensorRegisterFileOp createSfrTensorRegisterFileOp(
    PatternRewriter &rewriter, Location loc,
    task::sfr::slice::DotProductEngineRegisterFile<task::sfr_data_t> &sfr) {
  return rewriter.create<task::sfr::SfrTensorRegisterFileOp>(
      loc, sfr.write_interleaving_flit_count, sfr.write_mode,
      sfr.write_mac_rows, sfr.write_skip_flit_count, sfr.write_row_base,
      sfr.write_mac_row_interleaving, sfr.write_row_count,
      sfr.write_flits_per_period, sfr.write_valid_flits_per_period);
}

task::sfr::SfrVectorArithmeticUnitOp createSfrVectorArithmeticUnitOp(
    PatternRewriter &rewriter, Location loc,
    task::sfr::slice::VectorArithmeticUnitMainContext<task::sfr_data_t> &sfr) {
  ArrayAttr reduce_layer_acc_indexer_limits = rewriter.getI64ArrayAttr({
      static_cast<std::int64_t>(
          sfr.reduce_layer_acc_limit0_acc_indexer_limit_element0),
      static_cast<std::int64_t>(
          sfr.reduce_layer_acc_limit0_acc_indexer_limit_element1),
      static_cast<std::int64_t>(
          sfr.reduce_layer_acc_limit0_acc_indexer_limit_element2),
      static_cast<std::int64_t>(
          sfr.reduce_layer_acc_limit0_acc_indexer_limit_element3),
      static_cast<std::int64_t>(
          sfr.reduce_layer_acc_limit1_acc_indexer_limit_element4),
      static_cast<std::int64_t>(
          sfr.reduce_layer_acc_limit1_acc_indexer_limit_element5),
      static_cast<std::int64_t>(
          sfr.reduce_layer_acc_limit1_acc_indexer_limit_element6),
      static_cast<std::int64_t>(
          sfr.reduce_layer_acc_limit1_acc_indexer_limit_element7),
  });
  ArrayAttr reduce_layer_acc_indexer_strides = rewriter.getI64ArrayAttr({
      static_cast<std::int64_t>(
          sfr.reduce_layer_acc_stride_acc_indexer_stride_element0),
      static_cast<std::int64_t>(
          sfr.reduce_layer_acc_stride_acc_indexer_stride_element1),
      static_cast<std::int64_t>(
          sfr.reduce_layer_acc_stride_acc_indexer_stride_element2),
      static_cast<std::int64_t>(
          sfr.reduce_layer_acc_stride_acc_indexer_stride_element3),
      static_cast<std::int64_t>(
          sfr.reduce_layer_acc_stride_acc_indexer_stride_element4),
      static_cast<std::int64_t>(
          sfr.reduce_layer_acc_stride_acc_indexer_stride_element5),
      static_cast<std::int64_t>(
          sfr.reduce_layer_acc_stride_acc_indexer_stride_element6),
      static_cast<std::int64_t>(
          sfr.reduce_layer_acc_stride_acc_indexer_stride_element7),
  });
  ArrayAttr read_indexer0_limits = rewriter.getI64ArrayAttr({
      static_cast<std::int64_t>(
          sfr.read_indexer0_limit_info0_read_indexer0_limit_element0),
      static_cast<std::int64_t>(
          sfr.read_indexer0_limit_info0_read_indexer0_limit_element1),
      static_cast<std::int64_t>(
          sfr.read_indexer0_limit_info0_read_indexer0_limit_element2),
      static_cast<std::int64_t>(
          sfr.read_indexer0_limit_info0_read_indexer0_limit_element3),
      static_cast<std::int64_t>(
          sfr.read_indexer0_limit_info1_read_indexer0_limit_element4),
      static_cast<std::int64_t>(
          sfr.read_indexer0_limit_info1_read_indexer0_limit_element5),
      static_cast<std::int64_t>(
          sfr.read_indexer0_limit_info1_read_indexer0_limit_element6),
      static_cast<std::int64_t>(
          sfr.read_indexer0_limit_info1_read_indexer0_limit_element7),
  });
  ArrayAttr read_indexer0_strides = rewriter.getI64ArrayAttr({
      static_cast<std::int64_t>(
          sfr.read_indexer0_stride_info0_read_indexer0_stride_element0),
      static_cast<std::int64_t>(
          sfr.read_indexer0_stride_info0_read_indexer0_stride_element1),
      static_cast<std::int64_t>(
          sfr.read_indexer0_stride_info0_read_indexer0_stride_element2),
      static_cast<std::int64_t>(
          sfr.read_indexer0_stride_info0_read_indexer0_stride_element3),
      static_cast<std::int64_t>(
          sfr.read_indexer0_stride_info1_read_indexer0_stride_element4),
      static_cast<std::int64_t>(
          sfr.read_indexer0_stride_info1_read_indexer0_stride_element5),
      static_cast<std::int64_t>(
          sfr.read_indexer0_stride_info1_read_indexer0_stride_element6),
      static_cast<std::int64_t>(
          sfr.read_indexer0_stride_info1_read_indexer0_stride_element7),
  });
  ArrayAttr read_indexer1_limits = rewriter.getI64ArrayAttr({
      static_cast<std::int64_t>(
          sfr.read_indexer1_limit_info0_read_indexer1_limit_element0),
      static_cast<std::int64_t>(
          sfr.read_indexer1_limit_info0_read_indexer1_limit_element1),
      static_cast<std::int64_t>(
          sfr.read_indexer1_limit_info0_read_indexer1_limit_element2),
      static_cast<std::int64_t>(
          sfr.read_indexer1_limit_info0_read_indexer1_limit_element3),
      static_cast<std::int64_t>(
          sfr.read_indexer1_limit_info1_read_indexer1_limit_element4),
      static_cast<std::int64_t>(
          sfr.read_indexer1_limit_info1_read_indexer1_limit_element5),
      static_cast<std::int64_t>(
          sfr.read_indexer1_limit_info1_read_indexer1_limit_element6),
      static_cast<std::int64_t>(
          sfr.read_indexer1_limit_info1_read_indexer1_limit_element7),
  });
  ArrayAttr read_indexer1_strides = rewriter.getI64ArrayAttr({
      static_cast<std::int64_t>(
          sfr.read_indexer1_stride_info0_read_indexer1_stride_element0),
      static_cast<std::int64_t>(
          sfr.read_indexer1_stride_info0_read_indexer1_stride_element1),
      static_cast<std::int64_t>(
          sfr.read_indexer1_stride_info0_read_indexer1_stride_element2),
      static_cast<std::int64_t>(
          sfr.read_indexer1_stride_info0_read_indexer1_stride_element3),
      static_cast<std::int64_t>(
          sfr.read_indexer1_stride_info1_read_indexer1_stride_element4),
      static_cast<std::int64_t>(
          sfr.read_indexer1_stride_info1_read_indexer1_stride_element5),
      static_cast<std::int64_t>(
          sfr.read_indexer1_stride_info1_read_indexer1_stride_element6),
      static_cast<std::int64_t>(
          sfr.read_indexer1_stride_info1_read_indexer1_stride_element7),
  });
  ArrayAttr read_indexer2_limits = rewriter.getI64ArrayAttr({
      static_cast<std::int64_t>(
          sfr.read_indexer2_limit_info0_read_indexer2_limit_element0),
      static_cast<std::int64_t>(
          sfr.read_indexer2_limit_info0_read_indexer2_limit_element1),
      static_cast<std::int64_t>(
          sfr.read_indexer2_limit_info0_read_indexer2_limit_element2),
      static_cast<std::int64_t>(
          sfr.read_indexer2_limit_info0_read_indexer2_limit_element3),
      static_cast<std::int64_t>(
          sfr.read_indexer2_limit_info1_read_indexer2_limit_element4),
      static_cast<std::int64_t>(
          sfr.read_indexer2_limit_info1_read_indexer2_limit_element5),
      static_cast<std::int64_t>(
          sfr.read_indexer2_limit_info1_read_indexer2_limit_element6),
      static_cast<std::int64_t>(
          sfr.read_indexer2_limit_info1_read_indexer2_limit_element7),
  });
  ArrayAttr read_indexer2_strides = rewriter.getI64ArrayAttr({
      static_cast<std::int64_t>(
          sfr.read_indexer2_stride_info0_read_indexer2_stride_element0),
      static_cast<std::int64_t>(
          sfr.read_indexer2_stride_info0_read_indexer2_stride_element1),
      static_cast<std::int64_t>(
          sfr.read_indexer2_stride_info0_read_indexer2_stride_element2),
      static_cast<std::int64_t>(
          sfr.read_indexer2_stride_info0_read_indexer2_stride_element3),
      static_cast<std::int64_t>(
          sfr.read_indexer2_stride_info1_read_indexer2_stride_element4),
      static_cast<std::int64_t>(
          sfr.read_indexer2_stride_info1_read_indexer2_stride_element5),
      static_cast<std::int64_t>(
          sfr.read_indexer2_stride_info1_read_indexer2_stride_element6),
      static_cast<std::int64_t>(
          sfr.read_indexer2_stride_info1_read_indexer2_stride_element7),
  });
  ArrayAttr read_indexer3_limits = rewriter.getI64ArrayAttr({
      static_cast<std::int64_t>(
          sfr.read_indexer3_limit_info0_read_indexer3_limit_element0),
      static_cast<std::int64_t>(
          sfr.read_indexer3_limit_info0_read_indexer3_limit_element1),
      static_cast<std::int64_t>(
          sfr.read_indexer3_limit_info0_read_indexer3_limit_element2),
      static_cast<std::int64_t>(
          sfr.read_indexer3_limit_info0_read_indexer3_limit_element3),
      static_cast<std::int64_t>(
          sfr.read_indexer3_limit_info1_read_indexer3_limit_element4),
      static_cast<std::int64_t>(
          sfr.read_indexer3_limit_info1_read_indexer3_limit_element5),
      static_cast<std::int64_t>(
          sfr.read_indexer3_limit_info1_read_indexer3_limit_element6),
      static_cast<std::int64_t>(
          sfr.read_indexer3_limit_info1_read_indexer3_limit_element7),
  });
  ArrayAttr read_indexer3_strides = rewriter.getI64ArrayAttr({
      static_cast<std::int64_t>(
          sfr.read_indexer3_stride_info0_read_indexer3_stride_element0),
      static_cast<std::int64_t>(
          sfr.read_indexer3_stride_info0_read_indexer3_stride_element1),
      static_cast<std::int64_t>(
          sfr.read_indexer3_stride_info0_read_indexer3_stride_element2),
      static_cast<std::int64_t>(
          sfr.read_indexer3_stride_info0_read_indexer3_stride_element3),
      static_cast<std::int64_t>(
          sfr.read_indexer3_stride_info1_read_indexer3_stride_element4),
      static_cast<std::int64_t>(
          sfr.read_indexer3_stride_info1_read_indexer3_stride_element5),
      static_cast<std::int64_t>(
          sfr.read_indexer3_stride_info1_read_indexer3_stride_element6),
      static_cast<std::int64_t>(
          sfr.read_indexer3_stride_info1_read_indexer3_stride_element7),
  });
  ArrayAttr operand_indexer_limits = rewriter.getI64ArrayAttr({
      static_cast<std::int64_t>(
          sfr.operand_indexer_limit_info0_operand_indexer_limit_element0),
      static_cast<std::int64_t>(
          sfr.operand_indexer_limit_info0_operand_indexer_limit_element1),
      static_cast<std::int64_t>(
          sfr.operand_indexer_limit_info0_operand_indexer_limit_element2),
      static_cast<std::int64_t>(
          sfr.operand_indexer_limit_info0_operand_indexer_limit_element3),
      static_cast<std::int64_t>(
          sfr.operand_indexer_limit_info1_operand_indexer_limit_element4),
      static_cast<std::int64_t>(
          sfr.operand_indexer_limit_info1_operand_indexer_limit_element5),
      static_cast<std::int64_t>(
          sfr.operand_indexer_limit_info1_operand_indexer_limit_element6),
      static_cast<std::int64_t>(
          sfr.operand_indexer_limit_info1_operand_indexer_limit_element7),
  });
  ArrayAttr operand_indexer_strides = rewriter.getI64ArrayAttr({
      static_cast<std::int64_t>(
          sfr.operand_indexer_stride_info0_operand_indexer_stride_element0),
      static_cast<std::int64_t>(
          sfr.operand_indexer_stride_info0_operand_indexer_stride_element1),
      static_cast<std::int64_t>(
          sfr.operand_indexer_stride_info0_operand_indexer_stride_element2),
      static_cast<std::int64_t>(
          sfr.operand_indexer_stride_info0_operand_indexer_stride_element3),
      static_cast<std::int64_t>(
          sfr.operand_indexer_stride_info1_operand_indexer_stride_element4),
      static_cast<std::int64_t>(
          sfr.operand_indexer_stride_info1_operand_indexer_stride_element5),
      static_cast<std::int64_t>(
          sfr.operand_indexer_stride_info1_operand_indexer_stride_element6),
      static_cast<std::int64_t>(
          sfr.operand_indexer_stride_info1_operand_indexer_stride_element7),
  });
  ArrayAttr write_indexer_limits = rewriter.getI64ArrayAttr({
      static_cast<std::int64_t>(
          sfr.write_indexer_limit_info0_write_indexer_limit_element0),
      static_cast<std::int64_t>(
          sfr.write_indexer_limit_info0_write_indexer_limit_element1),
      static_cast<std::int64_t>(
          sfr.write_indexer_limit_info0_write_indexer_limit_element2),
      static_cast<std::int64_t>(
          sfr.write_indexer_limit_info0_write_indexer_limit_element3),
      static_cast<std::int64_t>(
          sfr.write_indexer_limit_info1_write_indexer_limit_element4),
      static_cast<std::int64_t>(
          sfr.write_indexer_limit_info1_write_indexer_limit_element5),
      static_cast<std::int64_t>(
          sfr.write_indexer_limit_info1_write_indexer_limit_element6),
      static_cast<std::int64_t>(
          sfr.write_indexer_limit_info1_write_indexer_limit_element7),
  });
  ArrayAttr write_indexer_strides = rewriter.getI64ArrayAttr({
      static_cast<std::int64_t>(
          sfr.write_indexer_stride_info0_write_indexer_stride_element0),
      static_cast<std::int64_t>(
          sfr.write_indexer_stride_info0_write_indexer_stride_element1),
      static_cast<std::int64_t>(
          sfr.write_indexer_stride_info0_write_indexer_stride_element2),
      static_cast<std::int64_t>(
          sfr.write_indexer_stride_info0_write_indexer_stride_element3),
      static_cast<std::int64_t>(
          sfr.write_indexer_stride_info1_write_indexer_stride_element4),
      static_cast<std::int64_t>(
          sfr.write_indexer_stride_info1_write_indexer_stride_element5),
      static_cast<std::int64_t>(
          sfr.write_indexer_stride_info1_write_indexer_stride_element6),
      static_cast<std::int64_t>(
          sfr.write_indexer_stride_info1_write_indexer_stride_element7),
  });

  return rewriter.create<task::sfr::SfrVectorArithmeticUnitOp>(
      loc, sfr.branch_mode_mode, sfr.branch_mode_format,
      sfr.branch_mode_compare_operation0, sfr.branch_mode_compare_operation1,
      sfr.branch_mode_compare_operation2, sfr.branch_mode_compare_operation3,
      sfr.branch_mode_group_size, sfr.branch_mode_branch_read_base,
      sfr.branch_mode_branch_read_limit,
      sfr.branch_data0_scalar_register_element0,
      sfr.branch_data0_scalar_register_element1,
      sfr.branch_data1_scalar_register_element2,
      sfr.branch_data1_scalar_register_element3,
      sfr.register_file_write_mode_branch_write_mode,
      sfr.register_file_write_mode_branch_write_base,
      sfr.register_file_write_mode_branch_write_limit,
      sfr.register_file_write_mode_write_cmp_op,
      sfr.register_file_write_mode_write_execution_id,
      sfr.register_file_write_mode_write_execution_id_mask,
      sfr.logic_cluster_route_logic_and_source,
      sfr.logic_cluster_route_logic_or_source,
      sfr.logic_cluster_route_logic_xor_source,
      sfr.logic_cluster_route_logic_left_shift_source,
      sfr.logic_cluster_route_logic_right_shift_source,
      sfr.logic_cluster_route_logic_cluster_source,
      sfr.logic_and_control_op_mode, sfr.logic_and_control_arg_mode,
      sfr.logic_and_control_reg0_cmp_op, sfr.logic_and_control_reg1_cmp_op,
      sfr.logic_and_control_reg2_cmp_op, sfr.logic_and_control_rf_cmp_op,
      sfr.logic_and_control_reg0_execution_id,
      sfr.logic_and_control_reg0_execution_id_mask,
      sfr.logic_and_control_reg1_execution_id,
      sfr.logic_and_control_reg1_execution_id_mask,
      sfr.logic_and_control_reg2_execution_id,
      sfr.logic_and_control_reg2_execution_id_mask,
      sfr.logic_and_control_rf_execution_id,
      sfr.logic_and_control_rf_execution_id_mask,
      sfr.logic_and_data0_scalar_register_element0,
      sfr.logic_and_data0_scalar_register_element1,
      sfr.logic_and_data1_scalar_register_element2,
      sfr.logic_or_control_op_mode, sfr.logic_or_control_arg_mode,
      sfr.logic_or_control_reg0_cmp_op, sfr.logic_or_control_reg1_cmp_op,
      sfr.logic_or_control_reg2_cmp_op, sfr.logic_or_control_rf_cmp_op,
      sfr.logic_or_control_reg0_execution_id,
      sfr.logic_or_control_reg0_execution_id_mask,
      sfr.logic_or_control_reg1_execution_id,
      sfr.logic_or_control_reg1_execution_id_mask,
      sfr.logic_or_control_reg2_execution_id,
      sfr.logic_or_control_reg2_execution_id_mask,
      sfr.logic_or_control_rf_execution_id,
      sfr.logic_or_control_rf_execution_id_mask,
      sfr.logic_or_data0_scalar_register_element0,
      sfr.logic_or_data0_scalar_register_element1,
      sfr.logic_or_data1_scalar_register_element2,
      sfr.logic_xor_control_op_mode, sfr.logic_xor_control_arg_mode,
      sfr.logic_xor_control_reg0_cmp_op, sfr.logic_xor_control_reg1_cmp_op,
      sfr.logic_xor_control_reg2_cmp_op, sfr.logic_xor_control_rf_cmp_op,
      sfr.logic_xor_control_reg0_execution_id,
      sfr.logic_xor_control_reg0_execution_id_mask,
      sfr.logic_xor_control_reg1_execution_id,
      sfr.logic_xor_control_reg1_execution_id_mask,
      sfr.logic_xor_control_reg2_execution_id,
      sfr.logic_xor_control_reg2_execution_id_mask,
      sfr.logic_xor_control_rf_execution_id,
      sfr.logic_xor_control_rf_execution_id_mask,
      sfr.logic_xor_data0_scalar_register_element0,
      sfr.logic_xor_data0_scalar_register_element1,
      sfr.logic_xor_data1_scalar_register_element2,
      sfr.logic_left_shift_control_op_mode,
      sfr.logic_left_shift_control_arg_mode,
      sfr.logic_left_shift_control_reg0_cmp_op,
      sfr.logic_left_shift_control_reg1_cmp_op,
      sfr.logic_left_shift_control_reg2_cmp_op,
      sfr.logic_left_shift_control_rf_cmp_op,
      sfr.logic_left_shift_control_reg0_execution_id,
      sfr.logic_left_shift_control_reg0_execution_id_mask,
      sfr.logic_left_shift_control_reg1_execution_id,
      sfr.logic_left_shift_control_reg1_execution_id_mask,
      sfr.logic_left_shift_control_reg2_execution_id,
      sfr.logic_left_shift_control_reg2_execution_id_mask,
      sfr.logic_left_shift_control_rf_execution_id,
      sfr.logic_left_shift_control_rf_execution_id_mask,
      sfr.logic_left_shift_data0_scalar_register_element0,
      sfr.logic_left_shift_data0_scalar_register_element1,
      sfr.logic_left_shift_data1_scalar_register_element2,
      sfr.logic_right_shift_control_op_mode,
      sfr.logic_right_shift_control_arg_mode,
      sfr.logic_right_shift_control_reg0_cmp_op,
      sfr.logic_right_shift_control_reg1_cmp_op,
      sfr.logic_right_shift_control_reg2_cmp_op,
      sfr.logic_right_shift_control_rf_cmp_op,
      sfr.logic_right_shift_control_reg0_execution_id,
      sfr.logic_right_shift_control_reg0_execution_id_mask,
      sfr.logic_right_shift_control_reg1_execution_id,
      sfr.logic_right_shift_control_reg1_execution_id_mask,
      sfr.logic_right_shift_control_reg2_execution_id,
      sfr.logic_right_shift_control_reg2_execution_id_mask,
      sfr.logic_right_shift_control_rf_execution_id,
      sfr.logic_right_shift_control_rf_execution_id_mask,
      sfr.logic_right_shift_data0_scalar_register_element0,
      sfr.logic_right_shift_data0_scalar_register_element1,
      sfr.logic_right_shift_data1_scalar_register_element2,
      sfr.fxp_cluster_route_fxp_add_source,
      sfr.fxp_cluster_route_fxp_left_shift_source,
      sfr.fxp_cluster_route_fxp_mul_source,
      sfr.fxp_cluster_route_fxp_right_shift_source,
      sfr.fxp_cluster_route_fxp_cluster_source, sfr.fxp_add_control_op_mode,
      sfr.fxp_add_control_arg_mode, sfr.fxp_add_control_reg0_cmp_op,
      sfr.fxp_add_control_reg1_cmp_op, sfr.fxp_add_control_reg2_cmp_op,
      sfr.fxp_add_control_rf_cmp_op, sfr.fxp_add_control_reg0_execution_id,
      sfr.fxp_add_control_reg0_execution_id_mask,
      sfr.fxp_add_control_reg1_execution_id,
      sfr.fxp_add_control_reg1_execution_id_mask,
      sfr.fxp_add_control_reg2_execution_id,
      sfr.fxp_add_control_reg2_execution_id_mask,
      sfr.fxp_add_control_rf_execution_id,
      sfr.fxp_add_control_rf_execution_id_mask,
      sfr.fxp_add_data0_scalar_register_element0,
      sfr.fxp_add_data0_scalar_register_element1,
      sfr.fxp_add_data1_scalar_register_element2,
      sfr.fxp_left_shift_control_op_mode, sfr.fxp_left_shift_control_arg_mode,
      sfr.fxp_left_shift_control_reg0_cmp_op,
      sfr.fxp_left_shift_control_reg1_cmp_op,
      sfr.fxp_left_shift_control_reg2_cmp_op,
      sfr.fxp_left_shift_control_rf_cmp_op,
      sfr.fxp_left_shift_control_reg0_execution_id,
      sfr.fxp_left_shift_control_reg0_execution_id_mask,
      sfr.fxp_left_shift_control_reg1_execution_id,
      sfr.fxp_left_shift_control_reg1_execution_id_mask,
      sfr.fxp_left_shift_control_reg2_execution_id,
      sfr.fxp_left_shift_control_reg2_execution_id_mask,
      sfr.fxp_left_shift_control_rf_execution_id,
      sfr.fxp_left_shift_control_rf_execution_id_mask,
      sfr.fxp_left_shift_data0_scalar_register_element0,
      sfr.fxp_left_shift_data0_scalar_register_element1,
      sfr.fxp_left_shift_data1_scalar_register_element2,
      sfr.fxp_mul_control_op_mode, sfr.fxp_mul_control_arg_mode,
      sfr.fxp_mul_control_reg0_cmp_op, sfr.fxp_mul_control_reg1_cmp_op,
      sfr.fxp_mul_control_reg2_cmp_op, sfr.fxp_mul_control_rf_cmp_op,
      sfr.fxp_mul_control_reg0_execution_id,
      sfr.fxp_mul_control_reg0_execution_id_mask,
      sfr.fxp_mul_control_reg1_execution_id,
      sfr.fxp_mul_control_reg1_execution_id_mask,
      sfr.fxp_mul_control_reg2_execution_id,
      sfr.fxp_mul_control_reg2_execution_id_mask,
      sfr.fxp_mul_control_rf_execution_id,
      sfr.fxp_mul_control_rf_execution_id_mask,
      sfr.fxp_mul_data0_scalar_register_element0,
      sfr.fxp_mul_data0_scalar_register_element1,
      sfr.fxp_mul_data1_scalar_register_element2,
      sfr.fxp_right_shift_control_op_mode, sfr.fxp_right_shift_control_arg_mode,
      sfr.fxp_right_shift_control_reg0_cmp_op,
      sfr.fxp_right_shift_control_reg1_cmp_op,
      sfr.fxp_right_shift_control_reg2_cmp_op,
      sfr.fxp_right_shift_control_rf_cmp_op,
      sfr.fxp_right_shift_control_reg0_execution_id,
      sfr.fxp_right_shift_control_reg0_execution_id_mask,
      sfr.fxp_right_shift_control_reg1_execution_id,
      sfr.fxp_right_shift_control_reg1_execution_id_mask,
      sfr.fxp_right_shift_control_reg2_execution_id,
      sfr.fxp_right_shift_control_reg2_execution_id_mask,
      sfr.fxp_right_shift_control_rf_execution_id,
      sfr.fxp_right_shift_control_rf_execution_id_mask,
      sfr.fxp_right_shift_data0_scalar_register_element0,
      sfr.fxp_right_shift_data0_scalar_register_element1,
      sfr.fxp_right_shift_data1_scalar_register_element2,
      sfr.fp_cluster_route_fp_fma_source, sfr.fp_cluster_route_fp_fpu_source,
      sfr.fp_cluster_route_fp_exp_source, sfr.fp_cluster_route_fp_mul0_source,
      sfr.fp_cluster_route_fp_mul1_source,
      sfr.fp_cluster_route_fp_cluster_source, sfr.fp_fma_control_op_mode,
      sfr.fp_fma_control_arg_mode, sfr.fp_fma_control_reg0_cmp_op,
      sfr.fp_fma_control_reg1_cmp_op, sfr.fp_fma_control_reg2_cmp_op,
      sfr.fp_fma_control_rf_cmp_op, sfr.fp_fma_control_reg0_execution_id,
      sfr.fp_fma_control_reg0_execution_id_mask,
      sfr.fp_fma_control_reg1_execution_id,
      sfr.fp_fma_control_reg1_execution_id_mask,
      sfr.fp_fma_control_reg2_execution_id,
      sfr.fp_fma_control_reg2_execution_id_mask,
      sfr.fp_fma_control_rf_execution_id,
      sfr.fp_fma_control_rf_execution_id_mask,
      sfr.fp_fma_data0_scalar_register_element0,
      sfr.fp_fma_data0_scalar_register_element1,
      sfr.fp_fma_data1_scalar_register_element2,
      sfr.fp_fma_data1_secondary_scalar_register_element0,
      sfr.fp_fma_data2_secondary_scalar_register_element1,
      sfr.fp_fma_data2_secondary_scalar_register_element2,
      sfr.fp_fpu_control_op_mode, sfr.fp_fpu_control_arg_mode,
      sfr.fp_fpu_control_reg0_cmp_op, sfr.fp_fpu_control_reg1_cmp_op,
      sfr.fp_fpu_control_reg2_cmp_op, sfr.fp_fpu_control_rf_cmp_op,
      sfr.fp_fpu_control_reg0_execution_id,
      sfr.fp_fpu_control_reg0_execution_id_mask,
      sfr.fp_fpu_control_reg1_execution_id,
      sfr.fp_fpu_control_reg1_execution_id_mask,
      sfr.fp_fpu_control_reg2_execution_id,
      sfr.fp_fpu_control_reg2_execution_id_mask,
      sfr.fp_fpu_control_rf_execution_id,
      sfr.fp_fpu_control_rf_execution_id_mask,
      sfr.fp_fpu_data0_scalar_register_element0,
      sfr.fp_fpu_data0_scalar_register_element1,
      sfr.fp_fpu_data1_scalar_register_element2, sfr.fp_exp_control_op_mode,
      sfr.fp_exp_control_arg_mode, sfr.fp_exp_control_reg0_cmp_op,
      sfr.fp_exp_control_reg1_cmp_op, sfr.fp_exp_control_reg2_cmp_op,
      sfr.fp_exp_control_rf_cmp_op, sfr.fp_exp_control_reg0_execution_id,
      sfr.fp_exp_control_reg0_execution_id_mask,
      sfr.fp_exp_control_reg1_execution_id,
      sfr.fp_exp_control_reg1_execution_id_mask,
      sfr.fp_exp_control_reg2_execution_id,
      sfr.fp_exp_control_reg2_execution_id_mask,
      sfr.fp_exp_control_rf_execution_id,
      sfr.fp_exp_control_rf_execution_id_mask, sfr.fp_mul0_control_op_mode,
      sfr.fp_mul0_control_arg_mode, sfr.fp_mul0_control_reg0_cmp_op,
      sfr.fp_mul0_control_reg1_cmp_op, sfr.fp_mul0_control_reg2_cmp_op,
      sfr.fp_mul0_control_rf_cmp_op, sfr.fp_mul0_control_reg0_execution_id,
      sfr.fp_mul0_control_reg0_execution_id_mask,
      sfr.fp_mul0_control_reg1_execution_id,
      sfr.fp_mul0_control_reg1_execution_id_mask,
      sfr.fp_mul0_control_reg2_execution_id,
      sfr.fp_mul0_control_reg2_execution_id_mask,
      sfr.fp_mul0_control_rf_execution_id,
      sfr.fp_mul0_control_rf_execution_id_mask,
      sfr.fp_mul0_data0_scalar_register_element0,
      sfr.fp_mul0_data0_scalar_register_element1,
      sfr.fp_mul0_data1_scalar_register_element2, sfr.fp_mul1_control_op_mode,
      sfr.fp_mul1_control_arg_mode, sfr.fp_mul1_control_reg0_cmp_op,
      sfr.fp_mul1_control_reg1_cmp_op, sfr.fp_mul1_control_reg2_cmp_op,
      sfr.fp_mul1_control_rf_cmp_op, sfr.fp_mul1_control_reg0_execution_id,
      sfr.fp_mul1_control_reg0_execution_id_mask,
      sfr.fp_mul1_control_reg1_execution_id,
      sfr.fp_mul1_control_reg1_execution_id_mask,
      sfr.fp_mul1_control_reg2_execution_id,
      sfr.fp_mul1_control_reg2_execution_id_mask,
      sfr.fp_mul1_control_rf_execution_id,
      sfr.fp_mul1_control_rf_execution_id_mask,
      sfr.fp_mul1_data0_scalar_register_element0,
      sfr.fp_mul1_data0_scalar_register_element1,
      sfr.fp_mul1_data1_scalar_register_element2,
      sfr.reduce_layer_mode_reduce_data_path, sfr.reduce_layer_mode_reduce_rows,
      sfr.reduce_layer_mode_reduce_tree_depth, sfr.reduce_layer_mode_acc_mode,
      sfr.reduce_layer_mode_acc_indexer_proceed,
      sfr.reduce_layer_mode_fxp_shift_rounding_mode,
      sfr.reduce_layer_mode_reduce_row0_op_mode,
      sfr.reduce_layer_mode_reduce_row1_op_mode,
      sfr.reduce_layer_mode_accumulation_limit,
      sfr.reduce_layer_mode_acc_indexer_base,
      sfr.reduce_layer_acc_init_reduce_row0_acc_init,
      sfr.reduce_layer_acc_init_reduce_row1_acc_init,
      reduce_layer_acc_indexer_limits, reduce_layer_acc_indexer_strides,
      sfr.fp_div_control_op_mode, sfr.fp_div_control_arg_mode,
      sfr.fp_div_control_reg0_cmp_op, sfr.fp_div_control_reg1_cmp_op,
      sfr.fp_div_control_reg2_cmp_op, sfr.fp_div_control_rf_cmp_op,
      sfr.fp_div_control_acc_cmp_op, sfr.fp_div_control_reg0_execution_id,
      sfr.fp_div_control_reg0_execution_id_mask,
      sfr.fp_div_control_reg1_execution_id,
      sfr.fp_div_control_reg1_execution_id_mask,
      sfr.fp_div_control_reg2_execution_id,
      sfr.fp_div_control_reg2_execution_id_mask,
      sfr.fp_div_control_rf_execution_id,
      sfr.fp_div_control_rf_execution_id_mask,
      sfr.fp_div_control_acc_execution_id,
      sfr.fp_div_control_acc_execution_id_mask,
      sfr.fp_div_data0_scalar_register_element0,
      sfr.fp_div_data0_scalar_register_element1,
      sfr.fp_div_data1_scalar_register_element2,
      sfr.float_adapter_fxp_to_fp_mode, sfr.float_adapter_fxp_to_fp_int_width,
      sfr.float_adapter_fxp_to_fp_round_mode, sfr.float_adapter_fp_to_fxp_mode,
      sfr.float_adapter_fp_to_fxp_int_width,
      sfr.float_adapter_fp_to_fxp_round_mode,
      sfr.float_adapter_split_layer_mode, sfr.float_adapter_concat_layer_mode,
      sfr.clip_cluster_route_clip_add_source,
      sfr.clip_cluster_route_clip_max_source,
      sfr.clip_cluster_route_clip_min_source,
      sfr.clip_cluster_route_clip_cluster_source, sfr.clip_add_control_op_mode,
      sfr.clip_add_control_arg_mode, sfr.clip_add_control_reg0_cmp_op,
      sfr.clip_add_control_reg1_cmp_op, sfr.clip_add_control_reg2_cmp_op,
      sfr.clip_add_control_rf_cmp_op, sfr.clip_add_control_reg0_execution_id,
      sfr.clip_add_control_reg0_execution_id_mask,
      sfr.clip_add_control_reg1_execution_id,
      sfr.clip_add_control_reg1_execution_id_mask,
      sfr.clip_add_control_reg2_execution_id,
      sfr.clip_add_control_reg2_execution_id_mask,
      sfr.clip_add_control_rf_execution_id,
      sfr.clip_add_control_rf_execution_id_mask,
      sfr.clip_add_data0_scalar_register_element0,
      sfr.clip_add_data0_scalar_register_element1,
      sfr.clip_add_data1_scalar_register_element2, sfr.clip_max_control_op_mode,
      sfr.clip_max_control_arg_mode, sfr.clip_max_control_reg0_cmp_op,
      sfr.clip_max_control_reg1_cmp_op, sfr.clip_max_control_reg2_cmp_op,
      sfr.clip_max_control_rf_cmp_op, sfr.clip_max_control_reg0_execution_id,
      sfr.clip_max_control_reg0_execution_id_mask,
      sfr.clip_max_control_reg1_execution_id,
      sfr.clip_max_control_reg1_execution_id_mask,
      sfr.clip_max_control_reg2_execution_id,
      sfr.clip_max_control_reg2_execution_id_mask,
      sfr.clip_max_control_rf_execution_id,
      sfr.clip_max_control_rf_execution_id_mask,
      sfr.clip_max_data0_scalar_register_element0,
      sfr.clip_max_data0_scalar_register_element1,
      sfr.clip_max_data1_scalar_register_element2, sfr.clip_min_control_op_mode,
      sfr.clip_min_control_arg_mode, sfr.clip_min_control_reg0_cmp_op,
      sfr.clip_min_control_reg1_cmp_op, sfr.clip_min_control_reg2_cmp_op,
      sfr.clip_min_control_rf_cmp_op, sfr.clip_min_control_reg0_execution_id,
      sfr.clip_min_control_reg0_execution_id_mask,
      sfr.clip_min_control_reg1_execution_id,
      sfr.clip_min_control_reg1_execution_id_mask,
      sfr.clip_min_control_reg2_execution_id,
      sfr.clip_min_control_reg2_execution_id_mask,
      sfr.clip_min_control_rf_execution_id,
      sfr.clip_min_control_rf_execution_id_mask,
      sfr.clip_min_data0_scalar_register_element0,
      sfr.clip_min_data0_scalar_register_element1,
      sfr.clip_min_data1_scalar_register_element2,
      sfr.alloc_indexer_read_indexer0_module,
      sfr.alloc_indexer_read_indexer1_module,
      sfr.alloc_indexer_read_indexer2_module,
      sfr.alloc_indexer_read_indexer3_module,
      sfr.alloc_indexer_operand_indexer_module,
      sfr.alloc_indexer_write_indexer_module, sfr.read_indexer0_operation,
      sfr.operation_read_indexer0_proceed,
      sfr.operation_read_indexer0_element_size, sfr.read_indexer1_operation,
      sfr.operation_read_indexer1_proceed,
      sfr.operation_read_indexer1_element_size, sfr.read_indexer2_operation,
      sfr.operation_read_indexer2_proceed,
      sfr.operation_read_indexer2_element_size, sfr.read_indexer3_operation,
      sfr.operation_read_indexer3_proceed,
      sfr.operation_read_indexer3_element_size, sfr.operand_indexer_operation,
      sfr.operation_operand_indexer_proceed,
      sfr.operation_operand_indexer_update_mode,
      sfr.operation_operand_indexer_element_size, sfr.write_indexer_operation,
      sfr.indexer_base0_read_indexer0_base,
      sfr.indexer_base0_read_indexer1_base,
      sfr.indexer_base0_read_indexer2_base,
      sfr.indexer_base0_read_indexer3_base,
      sfr.indexer_base1_operand_indexer_base,
      sfr.indexer_base1_write_indexer_base, read_indexer0_limits,
      read_indexer0_strides, read_indexer1_limits, read_indexer1_strides,
      read_indexer2_limits, read_indexer2_strides, read_indexer3_limits,
      read_indexer3_strides, operand_indexer_limits, operand_indexer_strides,
      write_indexer_limits, write_indexer_strides);
}

task::sfr::SfrVectorRouteUnitOp createSfrVectorRouteUnitOp(
    PatternRewriter &rewriter, Location loc,
    task::sfr::slice::VectorRouteUnitMainContext<task::sfr_data_t> &sfr) {
  ArrayAttr indexer_limits = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.indexer_limit0_index_limit_element0),
       static_cast<std::int64_t>(sfr.indexer_limit0_index_limit_element1),
       static_cast<std::int64_t>(sfr.indexer_limit0_index_limit_element2),
       static_cast<std::int64_t>(sfr.indexer_limit0_index_limit_element3),
       static_cast<std::int64_t>(sfr.indexer_limit1_index_limit_element4),
       static_cast<std::int64_t>(sfr.indexer_limit1_index_limit_element5),
       static_cast<std::int64_t>(sfr.indexer_limit1_index_limit_element6),
       static_cast<std::int64_t>(sfr.indexer_limit1_index_limit_element7)});
  ArrayAttr indexer_strides = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.indexer_stride0_index_stride_element0),
       static_cast<std::int64_t>(sfr.indexer_stride0_index_stride_element1),
       static_cast<std::int64_t>(sfr.indexer_stride0_index_stride_element2),
       static_cast<std::int64_t>(sfr.indexer_stride0_index_stride_element3),
       static_cast<std::int64_t>(sfr.indexer_stride1_index_stride_element4),
       static_cast<std::int64_t>(sfr.indexer_stride1_index_stride_element5),
       static_cast<std::int64_t>(sfr.indexer_stride1_index_stride_element6),
       static_cast<std::int64_t>(sfr.indexer_stride1_index_stride_element7)});
  ArrayAttr valid_generator_lowered_limits = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(
           sfr.valid_generator_limit0_lowered_limit_element0),
       static_cast<std::int64_t>(
           sfr.valid_generator_limit0_lowered_limit_element1),
       static_cast<std::int64_t>(
           sfr.valid_generator_limit0_lowered_limit_element2),
       static_cast<std::int64_t>(
           sfr.valid_generator_limit0_lowered_limit_element3),
       static_cast<std::int64_t>(
           sfr.valid_generator_limit1_lowered_limit_element4),
       static_cast<std::int64_t>(
           sfr.valid_generator_limit1_lowered_limit_element5),
       static_cast<std::int64_t>(
           sfr.valid_generator_limit1_lowered_limit_element6),
       static_cast<std::int64_t>(
           sfr.valid_generator_limit1_lowered_limit_element7)});
  ArrayAttr valid_generator_lowered_strides = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(
           sfr.valid_generator_stride0_lowered_stride_element0),
       static_cast<std::int64_t>(
           sfr.valid_generator_stride0_lowered_stride_element1),
       static_cast<std::int64_t>(
           sfr.valid_generator_stride0_lowered_stride_element2),
       static_cast<std::int64_t>(
           sfr.valid_generator_stride0_lowered_stride_element3),
       static_cast<std::int64_t>(
           sfr.valid_generator_stride1_lowered_stride_element4),
       static_cast<std::int64_t>(
           sfr.valid_generator_stride1_lowered_stride_element5),
       static_cast<std::int64_t>(
           sfr.valid_generator_stride1_lowered_stride_element6),
       static_cast<std::int64_t>(
           sfr.valid_generator_stride1_lowered_stride_element7)});
  ArrayAttr valid_generator_allocated_original_dim = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(
           sfr.valid_generator_original_dim_allocated_original_dim0),
       static_cast<std::int64_t>(
           sfr.valid_generator_original_dim_allocated_original_dim1),
       static_cast<std::int64_t>(
           sfr.valid_generator_original_dim_allocated_original_dim2),
       static_cast<std::int64_t>(
           sfr.valid_generator_original_dim_allocated_original_dim3),
       static_cast<std::int64_t>(
           sfr.valid_generator_original_dim_allocated_original_dim4),
       static_cast<std::int64_t>(
           sfr.valid_generator_original_dim_allocated_original_dim5),
       static_cast<std::int64_t>(
           sfr.valid_generator_original_dim_allocated_original_dim6),
       static_cast<std::int64_t>(
           sfr.valid_generator_original_dim_allocated_original_dim7)});
  ArrayAttr valid_generator_original_dim_partition_config =
      rewriter.getI64ArrayAttr(
          {static_cast<std::int64_t>(
               sfr.valid_generator_original_dim_original_dim_partition_config0),
           static_cast<std::int64_t>(
               sfr.valid_generator_original_dim_original_dim_partition_config1),
           static_cast<std::int64_t>(
               sfr.valid_generator_original_dim_original_dim_partition_config2),
           static_cast<std::int64_t>(
               sfr.valid_generator_original_dim_original_dim_partition_config3)});
  ArrayAttr valid_generator_original_dim_valid_count = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(
           sfr.valid_generator_valid_count_original_dim_valid_count0),
       static_cast<std::int64_t>(
           sfr.valid_generator_valid_count_original_dim_valid_count1),
       static_cast<std::int64_t>(
           sfr.valid_generator_valid_count_original_dim_valid_count2),
       static_cast<std::int64_t>(
           sfr.valid_generator_valid_count_original_dim_valid_count3)});
  ArrayAttr valid_generator_slice_mask = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.valid_generator_slice_info_slice_mask0),
       static_cast<std::int64_t>(sfr.valid_generator_slice_info_slice_mask1),
       static_cast<std::int64_t>(sfr.valid_generator_slice_info_slice_mask2),
       static_cast<std::int64_t>(sfr.valid_generator_slice_info_slice_mask3)});
  ArrayAttr valid_generator_slice_id_match = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(
           sfr.valid_generator_slice_info_slice_id_match0),
       static_cast<std::int64_t>(
           sfr.valid_generator_slice_info_slice_id_match1),
       static_cast<std::int64_t>(
           sfr.valid_generator_slice_info_slice_id_match2),
       static_cast<std::int64_t>(
           sfr.valid_generator_slice_info_slice_id_match3)});

  return rewriter.create<task::sfr::SfrVectorRouteUnitOp>(
      loc, sfr.route_info_data_out_source,
      sfr.route_info_reduce_channel_out_source,
      sfr.route_info_reduce_unit_in_source,
      sfr.route_info_arithmetic_unit_in_source,
      sfr.route_info_valid_generator_mode, sfr.route_info_route_mask,
      sfr.route_info_route_group_size, sfr.route_info_index_base = 0,
      indexer_limits, indexer_strides, valid_generator_lowered_limits,
      valid_generator_lowered_strides, valid_generator_allocated_original_dim,
      valid_generator_original_dim_partition_config,
      valid_generator_original_dim_valid_count, valid_generator_slice_mask,
      valid_generator_slice_id_match, sfr.collect_compaction_mode,
      sfr.compaction_mode_collect_compaction_cmp_op,
      sfr.compaction_mode_collect_compaction_execution_id,
      sfr.compaction_mode_collect_compaction_execution_id_mask,
      sfr.cast_compaction_mode, sfr.compaction_mode_cast_compaction_count);
}

task::sfr::SfrMainCommitUnitOp createSfrMainCommitUnitOp(
    PatternRewriter &rewriter, Location loc,
    task::sfr::slice::CommitUnitMainContext<task::sfr_data_t> &sfr) {
  ArrayAttr limits = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.limits_element0),
       static_cast<std::int64_t>(sfr.limits_element1),
       static_cast<std::int64_t>(sfr.limits_element2),
       static_cast<std::int64_t>(sfr.limits_element3),
       static_cast<std::int64_t>(sfr.limits_element4),
       static_cast<std::int64_t>(sfr.limits_element5),
       static_cast<std::int64_t>(sfr.limits_element6),
       static_cast<std::int64_t>(sfr.limits_element7)});
  ArrayAttr strides = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.strides_element0),
       static_cast<std::int64_t>(sfr.strides_element1),
       static_cast<std::int64_t>(sfr.strides_element2),
       static_cast<std::int64_t>(sfr.strides_element3),
       static_cast<std::int64_t>(sfr.strides_element4),
       static_cast<std::int64_t>(sfr.strides_element5),
       static_cast<std::int64_t>(sfr.strides_element6),
       static_cast<std::int64_t>(sfr.strides_element7)});
  ArrayAttr slice_enable_bitmap_mask = rewriter.getI64ArrayAttr(
      {static_cast<std::int64_t>(sfr.slice_enable_bitmap_mask_element0),
       static_cast<std::int64_t>(sfr.slice_enable_bitmap_mask_element1),
       static_cast<std::int64_t>(sfr.slice_enable_bitmap_mask_element2),
       static_cast<std::int64_t>(sfr.slice_enable_bitmap_mask_element3)});

  return rewriter.create<task::sfr::SfrMainCommitUnitOp>(
      loc, sfr.type_conversion, sfr.base, sfr.commit_in_size, sfr.commit_size,
      limits, strides, slice_enable_bitmap_mask);
}

} // namespace mlir::furiosa
