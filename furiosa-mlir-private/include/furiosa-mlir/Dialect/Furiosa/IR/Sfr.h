#pragma once

#include "furiosa-mlir/Dialect/Furiosa/IR/SfrBase.h"

namespace mlir::furiosa {
namespace sfr {
namespace slice {
template <class DT> class Common : public sfr::Block<DT> {
public:
  Common() : sfr::Block<DT>::Block(0x0, 0x30, false) {}

public:
  sfr::Register<DT> common_slice_id{"common_slice_id", *this, 0x0, 0x0};
  sfr::Bitfield<DT> slice_id{"slice_id", common_slice_id, 6, 0};
  sfr::Bitfield<DT> invalid_slice{"invalid_slice", common_slice_id, 1, 6};
  sfr::Register<DT> common_fusion{"common_fusion", *this, 0x8, 0x0};
  sfr::Bitfield<DT> fusion_processing_element_id{"fusion_processing_element_id",
                                                 common_fusion, 2, 0};
  sfr::Bitfield<DT> fusion_enable{"fusion_enable", common_fusion, 1, 32};
  sfr::Register<DT> common_soft_reset{"common_soft_reset", *this, 0x10,
                                      0x7ff000007ff};
  sfr::Bitfield<DT> main_context_soft_reset{"main_context_soft_reset",
                                            common_soft_reset, 11, 0};
  sfr::Bitfield<DT> sub_context_soft_reset{"sub_context_soft_reset",
                                           common_soft_reset, 11, 32};
  sfr::Register<DT> common_clock_enable{"common_clock_enable", *this, 0x18,
                                        0x0};
  sfr::Bitfield<DT> main_context_clock_enable{"main_context_clock_enable",
                                              common_clock_enable, 11, 0};
  sfr::Bitfield<DT> sub_context_clock_enable{"sub_context_clock_enable",
                                             common_clock_enable, 11, 32};
  sfr::Register<DT> common_enable{"common_enable", *this, 0x20, 0x0};
  sfr::Bitfield<DT> main_context_enable{"main_context_enable", common_enable,
                                        11, 0};
  sfr::Bitfield<DT> sub_context_enable{"sub_context_enable", common_enable, 11,
                                       32};
  sfr::Register<DT> common_context_id{"common_context_id", *this, 0x28, 0x0};
  sfr::Bitfield<DT> main_context_id{"main_context_id", common_context_id, 1, 0};
  sfr::Bitfield<DT> sub_context_id{"sub_context_id", common_context_id, 1, 32};
};

template <class DT> class FetchUnitMainContext : public sfr::Block<DT> {
public:
  FetchUnitMainContext() : sfr::Block<DT>::Block(0x1000, 0x80, false) {}

public:
  sfr::Register<DT> fetch_unit_fetch_mode{"fetch_unit_fetch_mode", *this, 0x0,
                                          0x0};
  sfr::Bitfield<DT> fetch_mode{"fetch_mode", fetch_unit_fetch_mode, 2, 0};
  sfr::Bitfield<DT> num_zero_points{"num_zero_points", fetch_unit_fetch_mode, 2,
                                    4};
  sfr::Bitfield<DT> zero_point0{"zero_point0", fetch_unit_fetch_mode, 8, 8};
  sfr::Bitfield<DT> zero_point1{"zero_point1", fetch_unit_fetch_mode, 8, 16};
  sfr::Bitfield<DT> table_entry_size{"table_entry_size", fetch_unit_fetch_mode,
                                     3, 24};
  sfr::Bitfield<DT> tables{"tables", fetch_unit_fetch_mode, 4, 28};
  sfr::Bitfield<DT> indirect_base{"indirect_base", fetch_unit_fetch_mode, 22,
                                  32};
  sfr::Bitfield<DT> indirect_dim{"indirect_dim", fetch_unit_fetch_mode, 3, 56};
  sfr::Bitfield<DT> table_base_mode{"table_base_mode", fetch_unit_fetch_mode, 1,
                                    59};
  sfr::Bitfield<DT> indirect_pointer_size{"indirect_pointer_size",
                                          fetch_unit_fetch_mode, 2, 60};
  sfr::Bitfield<DT> zeropoint_tail_mode{"zeropoint_tail_mode",
                                        fetch_unit_fetch_mode, 1, 63};
  sfr::Register<DT> fetch_unit_pad{"fetch_unit_pad", *this, 0x8, 0x0};
  sfr::Bitfield<DT> last_dim_pad_value{"last_dim_pad_value", fetch_unit_pad, 32,
                                       0};
  sfr::Bitfield<DT> last_dim{"last_dim", fetch_unit_pad, 3, 32};
  sfr::Bitfield<DT> pad_order{"pad_order", fetch_unit_pad, 1, 35};
  sfr::Bitfield<DT> last_dim_rightmost_valid_count_dim{
      "last_dim_rightmost_valid_count_dim", fetch_unit_pad, 4, 36};
  sfr::Bitfield<DT> last_dim_left_pad_count{"last_dim_left_pad_count",
                                            fetch_unit_pad, 7, 40};
  sfr::Bitfield<DT> type_conversion{"type_conversion", fetch_unit_pad, 4, 48};
  sfr::Bitfield<DT> last_dim_left_pad_mode{"last_dim_left_pad_mode",
                                           fetch_unit_pad, 1, 54};
  sfr::Bitfield<DT> zeropoint_dims{"zeropoint_dims", fetch_unit_pad, 8, 56};
  sfr::Register<DT> fetch_unit_last_dimension_rightmost_valid_count{
      "fetch_unit_last_dimension_rightmost_valid_count", *this, 0x10, 0x0};
  sfr::Bitfield<DT> last_dim_rightmost_valid_count0{
      "last_dim_rightmost_valid_count0",
      fetch_unit_last_dimension_rightmost_valid_count, 8, 0};
  sfr::Bitfield<DT> last_dim_rightmost_valid_count1{
      "last_dim_rightmost_valid_count1",
      fetch_unit_last_dimension_rightmost_valid_count, 8, 8};
  sfr::Bitfield<DT> last_dim_rightmost_valid_count2{
      "last_dim_rightmost_valid_count2",
      fetch_unit_last_dimension_rightmost_valid_count, 8, 16};
  sfr::Bitfield<DT> last_dim_rightmost_valid_count3{
      "last_dim_rightmost_valid_count3",
      fetch_unit_last_dimension_rightmost_valid_count, 8, 24};
  sfr::Bitfield<DT> last_dim_rightmost_valid_count4{
      "last_dim_rightmost_valid_count4",
      fetch_unit_last_dimension_rightmost_valid_count, 8, 32};
  sfr::Bitfield<DT> last_dim_rightmost_valid_count5{
      "last_dim_rightmost_valid_count5",
      fetch_unit_last_dimension_rightmost_valid_count, 8, 40};
  sfr::Bitfield<DT> last_dim_rightmost_valid_count6{
      "last_dim_rightmost_valid_count6",
      fetch_unit_last_dimension_rightmost_valid_count, 8, 48};
  sfr::Bitfield<DT> last_dim_rightmost_valid_count7{
      "last_dim_rightmost_valid_count7",
      fetch_unit_last_dimension_rightmost_valid_count, 8, 56};
  sfr::Register<DT> fetch_unit_base{"fetch_unit_base", *this, 0x18, 0x0};
  sfr::Bitfield<DT> base{"base", fetch_unit_base, 22, 0};
  sfr::Bitfield<DT> fetch_size{"fetch_size", fetch_unit_base, 6, 32};
  sfr::Register<DT> fetch_unit_limit0{"fetch_unit_limit0", *this, 0x20, 0x0};
  sfr::Bitfield<DT> limits_element0{"limits_element0", fetch_unit_limit0, 16,
                                    0};
  sfr::Bitfield<DT> limits_element1{"limits_element1", fetch_unit_limit0, 16,
                                    16};
  sfr::Bitfield<DT> limits_element2{"limits_element2", fetch_unit_limit0, 16,
                                    32};
  sfr::Bitfield<DT> limits_element3{"limits_element3", fetch_unit_limit0, 16,
                                    48};
  sfr::Register<DT> fetch_unit_limit1{"fetch_unit_limit1", *this, 0x28, 0x0};
  sfr::Bitfield<DT> limits_element4{"limits_element4", fetch_unit_limit1, 16,
                                    0};
  sfr::Bitfield<DT> limits_element5{"limits_element5", fetch_unit_limit1, 16,
                                    16};
  sfr::Bitfield<DT> limits_element6{"limits_element6", fetch_unit_limit1, 16,
                                    32};
  sfr::Bitfield<DT> limits_element7{"limits_element7", fetch_unit_limit1, 16,
                                    48};
  sfr::Register<DT> fetch_unit_stride0{"fetch_unit_stride0", *this, 0x30, 0x0};
  sfr::Bitfield<DT> strides_element0{"strides_element0", fetch_unit_stride0, 22,
                                     0};
  sfr::Bitfield<DT> strides_element1{"strides_element1", fetch_unit_stride0, 22,
                                     32};
  sfr::Register<DT> fetch_unit_stride1{"fetch_unit_stride1", *this, 0x38, 0x0};
  sfr::Bitfield<DT> strides_element2{"strides_element2", fetch_unit_stride1, 22,
                                     0};
  sfr::Bitfield<DT> strides_element3{"strides_element3", fetch_unit_stride1, 22,
                                     32};
  sfr::Register<DT> fetch_unit_stride2{"fetch_unit_stride2", *this, 0x40, 0x0};
  sfr::Bitfield<DT> strides_element4{"strides_element4", fetch_unit_stride2, 22,
                                     0};
  sfr::Bitfield<DT> strides_element5{"strides_element5", fetch_unit_stride2, 22,
                                     32};
  sfr::Register<DT> fetch_unit_stride3{"fetch_unit_stride3", *this, 0x48, 0x0};
  sfr::Bitfield<DT> strides_element6{"strides_element6", fetch_unit_stride3, 22,
                                     0};
  sfr::Bitfield<DT> strides_element7{"strides_element7", fetch_unit_stride3, 22,
                                     32};
  sfr::Register<DT> fetch_unit_fetch{"fetch_unit_fetch", *this, 0x50, 0x0};
  sfr::Bitfield<DT> flit_count{"flit_count", fetch_unit_fetch, 24, 0};
  sfr::Bitfield<DT> words_per_packet{"words_per_packet", fetch_unit_fetch, 16,
                                     32};
  sfr::Bitfield<DT> zeropoint_fetch_limit{"zeropoint_fetch_limit",
                                          fetch_unit_fetch, 16, 48};
  sfr::Register<DT> fetch_unit_topology{"fetch_unit_topology", *this, 0x58,
                                        0x0};
  sfr::Bitfield<DT> topology{"topology", fetch_unit_topology, 4, 0};
  sfr::Bitfield<DT> channel_config{"channel_config", fetch_unit_topology, 1, 4};
  sfr::Bitfield<DT> outer_slice_log_size{"outer_slice_log_size",
                                         fetch_unit_topology, 4, 8};
  sfr::Bitfield<DT> outer_dim0_log_size{"outer_dim0_log_size",
                                        fetch_unit_topology, 4, 16};
  sfr::Bitfield<DT> outer_dim1_log_size{"outer_dim1_log_size",
                                        fetch_unit_topology, 4, 24};
  sfr::Bitfield<DT> outer_dim0_chunk_size{"outer_dim0_chunk_size",
                                          fetch_unit_topology, 16, 32};
  sfr::Bitfield<DT> outer_dim1_chunk_size{"outer_dim1_chunk_size",
                                          fetch_unit_topology, 16, 48};
  sfr::Register<DT> fetch_unit_custom_snoop_bitmap0{
      "fetch_unit_custom_snoop_bitmap0", *this, 0x60, 0x0};
  sfr::Bitfield<DT> custom_snoop_bitmap_mask_element0{
      "custom_snoop_bitmap_mask_element0", fetch_unit_custom_snoop_bitmap0, 64,
      0};
  sfr::Register<DT> fetch_unit_custom_snoop_bitmap1{
      "fetch_unit_custom_snoop_bitmap1", *this, 0x68, 0x0};
  sfr::Bitfield<DT> custom_snoop_bitmap_mask_element1{
      "custom_snoop_bitmap_mask_element1", fetch_unit_custom_snoop_bitmap1, 64,
      0};
  sfr::Register<DT> fetch_unit_custom_snoop_bitmap2{
      "fetch_unit_custom_snoop_bitmap2", *this, 0x70, 0x0};
  sfr::Bitfield<DT> custom_snoop_bitmap_mask_element2{
      "custom_snoop_bitmap_mask_element2", fetch_unit_custom_snoop_bitmap2, 64,
      0};
  sfr::Register<DT> fetch_unit_custom_snoop_bitmap3{
      "fetch_unit_custom_snoop_bitmap3", *this, 0x78, 0x0};
  sfr::Bitfield<DT> custom_snoop_bitmap_mask_element3{
      "custom_snoop_bitmap_mask_element3", fetch_unit_custom_snoop_bitmap3, 64,
      0};
};

template <class DT>
class DotProductEngineWarmUpControl : public sfr::Block<DT> {
public:
  DotProductEngineWarmUpControl() : sfr::Block<DT>::Block(0x40, 0x8, false) {}

public:
  sfr::Register<DT> dot_product_engine_warm_up_control{
      "dot_product_engine_warm_up_control", *this, 0x0, 0x0};
  sfr::Bitfield<DT> accumulate_unit_warm_up_period{
      "accumulate_unit_warm_up_period", dot_product_engine_warm_up_control, 8,
      0};
  sfr::Bitfield<DT> accumulate_unit_cool_down_period{
      "accumulate_unit_cool_down_period", dot_product_engine_warm_up_control, 8,
      8};
  sfr::Bitfield<DT> register_file_warm_up_period{
      "register_file_warm_up_period", dot_product_engine_warm_up_control, 8,
      16};
  sfr::Bitfield<DT> register_file_cool_down_period{
      "register_file_cool_down_period", dot_product_engine_warm_up_control, 8,
      24};
};

template <class DT> class DotProductEngineRegisterFile : public sfr::Block<DT> {
public:
  DotProductEngineRegisterFile() : sfr::Block<DT>::Block(0x178, 0x10, false) {}

public:
  sfr::Register<DT> dot_product_engine_register_file_write_interleaving{
      "dot_product_engine_register_file_write_interleaving", *this, 0x0, 0x0};
  sfr::Bitfield<DT> write_interleaving_flit_count{
      "write_interleaving_flit_count",
      dot_product_engine_register_file_write_interleaving, 10, 0};
  sfr::Register<DT> dot_product_engine_register_file_write{
      "dot_product_engine_register_file_write", *this, 0x8, 0x0};
  sfr::Bitfield<DT> write_mode{"write_mode",
                               dot_product_engine_register_file_write, 2, 0};
  sfr::Bitfield<DT> write_mac_rows{
      "write_mac_rows", dot_product_engine_register_file_write, 4, 2};
  sfr::Bitfield<DT> write_skip_flit_count{
      "write_skip_flit_count", dot_product_engine_register_file_write, 10, 6};
  sfr::Bitfield<DT> write_row_base{
      "write_row_base", dot_product_engine_register_file_write, 11, 16};
  sfr::Bitfield<DT> write_mac_row_interleaving{
      "write_mac_row_interleaving", dot_product_engine_register_file_write, 1,
      27};
  sfr::Bitfield<DT> write_row_count{
      "write_row_count", dot_product_engine_register_file_write, 12, 28};
  sfr::Bitfield<DT> write_flits_per_period{
      "write_flits_per_period", dot_product_engine_register_file_write, 12, 40};
  sfr::Bitfield<DT> write_valid_flits_per_period{
      "write_valid_flits_per_period", dot_product_engine_register_file_write,
      12, 52};
};

template <class DT> class DotProductEngineParityError : public sfr::Block<DT> {
public:
  DotProductEngineParityError() : sfr::Block<DT>::Block(0x408, 0x8, false) {}

public:
  sfr::Register<DT> dot_product_engine_parity_error{
      "dot_product_engine_parity_error", *this, 0x0, 0x0};
  sfr::Bitfield<DT> status_trf_parity_error_counter{
      "status_trf_parity_error_counter", dot_product_engine_parity_error, 4, 0};
};

template <class DT> class DotProductEngineMainContext : public sfr::Block<DT> {
public:
  DotProductEngineMainContext() : sfr::Block<DT>::Block(0x1088, 0x70, false) {}

public:
  sfr::Register<DT> dot_product_engine_base{"dot_product_engine_base", *this,
                                            0x0, 0x0};
  sfr::Bitfield<DT> reg_indexer_base{"reg_indexer_base",
                                     dot_product_engine_base, 17, 0};
  sfr::Bitfield<DT> acc_indexer_base{"acc_indexer_base",
                                     dot_product_engine_base, 8, 32};
  sfr::Register<DT> dot_product_engine_feed{"dot_product_engine_feed", *this,
                                            0x8, 0x0};
  sfr::Bitfield<DT> flits_per_input{"flits_per_input", dot_product_engine_feed,
                                    2, 0};
  sfr::Bitfield<DT> feed_input_transpose{"feed_input_transpose",
                                         dot_product_engine_feed, 1, 8};
  sfr::Bitfield<DT> initial_shift_dim{"initial_shift_dim",
                                      dot_product_engine_feed, 4, 16};
  sfr::Bitfield<DT> shift_stride{"shift_stride", dot_product_engine_feed, 5,
                                 24};
  sfr::Bitfield<DT> pop_dim{"pop_dim", dot_product_engine_feed, 3, 32};
  sfr::Bitfield<DT> shift_dim{"shift_dim", dot_product_engine_feed, 3, 40};
  sfr::Bitfield<DT> channel_config{"channel_config", dot_product_engine_feed, 1,
                                   48};
  sfr::Bitfield<DT> feed_data_type{"feed_data_type", dot_product_engine_feed, 2,
                                   56};
  sfr::Register<DT> dot_product_engine_initial_shift{
      "dot_product_engine_initial_shift", *this, 0x10, 0x0};
  sfr::Bitfield<DT> initial_shift_element0{
      "initial_shift_element0", dot_product_engine_initial_shift, 5, 0};
  sfr::Bitfield<DT> initial_shift_element1{
      "initial_shift_element1", dot_product_engine_initial_shift, 5, 8};
  sfr::Bitfield<DT> initial_shift_element2{
      "initial_shift_element2", dot_product_engine_initial_shift, 5, 16};
  sfr::Bitfield<DT> initial_shift_element3{
      "initial_shift_element3", dot_product_engine_initial_shift, 5, 24};
  sfr::Bitfield<DT> initial_shift_element4{
      "initial_shift_element4", dot_product_engine_initial_shift, 5, 32};
  sfr::Bitfield<DT> initial_shift_element5{
      "initial_shift_element5", dot_product_engine_initial_shift, 5, 40};
  sfr::Bitfield<DT> initial_shift_element6{
      "initial_shift_element6", dot_product_engine_initial_shift, 5, 48};
  sfr::Bitfield<DT> initial_shift_element7{
      "initial_shift_element7", dot_product_engine_initial_shift, 5, 56};
  sfr::Register<DT> dot_product_engine_limit0{"dot_product_engine_limit0",
                                              *this, 0x18, 0x0};
  sfr::Bitfield<DT> iter_seq_limits_element0{"iter_seq_limits_element0",
                                             dot_product_engine_limit0, 16, 0};
  sfr::Bitfield<DT> iter_seq_limits_element1{"iter_seq_limits_element1",
                                             dot_product_engine_limit0, 16, 16};
  sfr::Bitfield<DT> iter_seq_limits_element2{"iter_seq_limits_element2",
                                             dot_product_engine_limit0, 16, 32};
  sfr::Bitfield<DT> iter_seq_limits_element3{"iter_seq_limits_element3",
                                             dot_product_engine_limit0, 16, 48};
  sfr::Register<DT> dot_product_engine_limit1{"dot_product_engine_limit1",
                                              *this, 0x20, 0x0};
  sfr::Bitfield<DT> iter_seq_limits_element4{"iter_seq_limits_element4",
                                             dot_product_engine_limit1, 16, 0};
  sfr::Bitfield<DT> iter_seq_limits_element5{"iter_seq_limits_element5",
                                             dot_product_engine_limit1, 16, 16};
  sfr::Bitfield<DT> iter_seq_limits_element6{"iter_seq_limits_element6",
                                             dot_product_engine_limit1, 16, 32};
  sfr::Bitfield<DT> iter_seq_limits_element7{"iter_seq_limits_element7",
                                             dot_product_engine_limit1, 16, 48};
  sfr::Register<DT> dot_product_engine_register_stride0{
      "dot_product_engine_register_stride0", *this, 0x28, 0x0};
  sfr::Bitfield<DT> reg_indexer_strides_element0{
      "reg_indexer_strides_element0", dot_product_engine_register_stride0, 17,
      0};
  sfr::Bitfield<DT> reg_indexer_strides_element1{
      "reg_indexer_strides_element1", dot_product_engine_register_stride0, 17,
      32};
  sfr::Register<DT> dot_product_engine_register_stride1{
      "dot_product_engine_register_stride1", *this, 0x30, 0x0};
  sfr::Bitfield<DT> reg_indexer_strides_element2{
      "reg_indexer_strides_element2", dot_product_engine_register_stride1, 17,
      0};
  sfr::Bitfield<DT> reg_indexer_strides_element3{
      "reg_indexer_strides_element3", dot_product_engine_register_stride1, 17,
      32};
  sfr::Register<DT> dot_product_engine_register_stride2{
      "dot_product_engine_register_stride2", *this, 0x38, 0x0};
  sfr::Bitfield<DT> reg_indexer_strides_element4{
      "reg_indexer_strides_element4", dot_product_engine_register_stride2, 17,
      0};
  sfr::Bitfield<DT> reg_indexer_strides_element5{
      "reg_indexer_strides_element5", dot_product_engine_register_stride2, 17,
      32};
  sfr::Register<DT> dot_product_engine_register_stride3{
      "dot_product_engine_register_stride3", *this, 0x40, 0x0};
  sfr::Bitfield<DT> reg_indexer_strides_element6{
      "reg_indexer_strides_element6", dot_product_engine_register_stride3, 17,
      0};
  sfr::Bitfield<DT> reg_indexer_strides_element7{
      "reg_indexer_strides_element7", dot_product_engine_register_stride3, 17,
      32};
  sfr::Register<DT> dot_product_engine_accumulator_stride0{
      "dot_product_engine_accumulator_stride0", *this, 0x48, 0x0};
  sfr::Bitfield<DT> acc_indexer_strides_element0{
      "acc_indexer_strides_element0", dot_product_engine_accumulator_stride0, 8,
      0};
  sfr::Bitfield<DT> acc_indexer_strides_element1{
      "acc_indexer_strides_element1", dot_product_engine_accumulator_stride0, 8,
      16};
  sfr::Bitfield<DT> acc_indexer_strides_element2{
      "acc_indexer_strides_element2", dot_product_engine_accumulator_stride0, 8,
      32};
  sfr::Bitfield<DT> acc_indexer_strides_element3{
      "acc_indexer_strides_element3", dot_product_engine_accumulator_stride0, 8,
      48};
  sfr::Register<DT> dot_product_engine_accumulator_stride1{
      "dot_product_engine_accumulator_stride1", *this, 0x50, 0x0};
  sfr::Bitfield<DT> acc_indexer_strides_element4{
      "acc_indexer_strides_element4", dot_product_engine_accumulator_stride1, 8,
      0};
  sfr::Bitfield<DT> acc_indexer_strides_element5{
      "acc_indexer_strides_element5", dot_product_engine_accumulator_stride1, 8,
      16};
  sfr::Bitfield<DT> acc_indexer_strides_element6{
      "acc_indexer_strides_element6", dot_product_engine_accumulator_stride1, 8,
      32};
  sfr::Bitfield<DT> acc_indexer_strides_element7{
      "acc_indexer_strides_element7", dot_product_engine_accumulator_stride1, 8,
      48};
  sfr::Register<DT> dot_product_engine_accumulation{
      "dot_product_engine_accumulation", *this, 0x58, 0x0};
  sfr::Bitfield<DT> acc_limit{"acc_limit", dot_product_engine_accumulation, 16,
                              0};
  sfr::Bitfield<DT> acc_cols{"acc_cols", dot_product_engine_accumulation, 6,
                             16};
  sfr::Bitfield<DT> acc_reset{"acc_reset", dot_product_engine_accumulation, 1,
                              24};
  sfr::Bitfield<DT> output_major{"output_major",
                                 dot_product_engine_accumulation, 1, 28};
  sfr::Bitfield<DT> acc_init_value{"acc_init_value",
                                   dot_product_engine_accumulation, 32, 32};
  sfr::Register<DT> dot_product_engine_mac_tree{"dot_product_engine_mac_tree",
                                                *this, 0x60, 0x0};
  sfr::Bitfield<DT> mac_tree_operation{"mac_tree_operation",
                                       dot_product_engine_mac_tree, 1, 0};
  sfr::Bitfield<DT> mac_tree_depth{"mac_tree_depth",
                                   dot_product_engine_mac_tree, 3, 8};
  sfr::Bitfield<DT> mac_type{"mac_type", dot_product_engine_mac_tree, 3, 16};
  sfr::Bitfield<DT> mac_rows{"mac_rows", dot_product_engine_mac_tree, 4, 24};
  sfr::Bitfield<DT> fp_ieee_nan_multiplication{
      "fp_ieee_nan_multiplication", dot_product_engine_mac_tree, 1, 32};
  sfr::Bitfield<DT> fxp_shift_rounding_mode{"fxp_shift_rounding_mode",
                                            dot_product_engine_mac_tree, 2, 40};
  sfr::Register<DT> dot_product_engine_register_file_read{
      "dot_product_engine_register_file_read", *this, 0x68, 0x0};
  sfr::Bitfield<DT> data_type{"data_type",
                              dot_product_engine_register_file_read, 2, 0};
  sfr::Bitfield<DT> reg_read_log_size{
      "reg_read_log_size", dot_product_engine_register_file_read, 3, 8};
  sfr::Bitfield<DT> reg_read_mode{"reg_read_mode",
                                  dot_product_engine_register_file_read, 2, 16};
  sfr::Bitfield<DT> reg_read_cache_mode{
      "reg_read_cache_mode", dot_product_engine_register_file_read, 1, 24};
};

template <class DT> class VectorRouteUnitMainContext : public sfr::Block<DT> {
public:
  VectorRouteUnitMainContext() : sfr::Block<DT>::Block(0x1160, 0x68, false) {}

public:
  sfr::Register<DT> vector_route_unit_route_info{"vector_route_unit_route_info",
                                                 *this, 0x0, 0x0};
  sfr::Bitfield<DT> route_info_data_out_source{
      "route_info_data_out_source", vector_route_unit_route_info, 3, 0};
  sfr::Bitfield<DT> route_info_reduce_channel_out_source{
      "route_info_reduce_channel_out_source", vector_route_unit_route_info, 3,
      4};
  sfr::Bitfield<DT> route_info_reduce_unit_in_source{
      "route_info_reduce_unit_in_source", vector_route_unit_route_info, 3, 8};
  sfr::Bitfield<DT> route_info_arithmetic_unit_in_source{
      "route_info_arithmetic_unit_in_source", vector_route_unit_route_info, 3,
      12};
  sfr::Bitfield<DT> route_info_valid_generator_mode{
      "route_info_valid_generator_mode", vector_route_unit_route_info, 1, 24};
  sfr::Bitfield<DT> route_info_route_mask{"route_info_route_mask",
                                          vector_route_unit_route_info, 8, 32};
  sfr::Bitfield<DT> route_info_route_group_size{
      "route_info_route_group_size", vector_route_unit_route_info, 4, 40};
  sfr::Bitfield<DT> route_info_index_base{"route_info_index_base",
                                          vector_route_unit_route_info, 9, 48};
  sfr::Register<DT> vector_route_unit_indexer_limit0{
      "vector_route_unit_indexer_limit0", *this, 0x8, 0x0};
  sfr::Bitfield<DT> indexer_limit0_index_limit_element0{
      "indexer_limit0_index_limit_element0", vector_route_unit_indexer_limit0,
      16, 0};
  sfr::Bitfield<DT> indexer_limit0_index_limit_element1{
      "indexer_limit0_index_limit_element1", vector_route_unit_indexer_limit0,
      16, 16};
  sfr::Bitfield<DT> indexer_limit0_index_limit_element2{
      "indexer_limit0_index_limit_element2", vector_route_unit_indexer_limit0,
      16, 32};
  sfr::Bitfield<DT> indexer_limit0_index_limit_element3{
      "indexer_limit0_index_limit_element3", vector_route_unit_indexer_limit0,
      16, 48};
  sfr::Register<DT> vector_route_unit_indexer_limit1{
      "vector_route_unit_indexer_limit1", *this, 0x10, 0x0};
  sfr::Bitfield<DT> indexer_limit1_index_limit_element4{
      "indexer_limit1_index_limit_element4", vector_route_unit_indexer_limit1,
      16, 0};
  sfr::Bitfield<DT> indexer_limit1_index_limit_element5{
      "indexer_limit1_index_limit_element5", vector_route_unit_indexer_limit1,
      16, 16};
  sfr::Bitfield<DT> indexer_limit1_index_limit_element6{
      "indexer_limit1_index_limit_element6", vector_route_unit_indexer_limit1,
      16, 32};
  sfr::Bitfield<DT> indexer_limit1_index_limit_element7{
      "indexer_limit1_index_limit_element7", vector_route_unit_indexer_limit1,
      16, 48};
  sfr::Register<DT> vector_route_unit_indexer_stride0{
      "vector_route_unit_indexer_stride0", *this, 0x18, 0x0};
  sfr::Bitfield<DT> indexer_stride0_index_stride_element0{
      "indexer_stride0_index_stride_element0",
      vector_route_unit_indexer_stride0, 9, 0};
  sfr::Bitfield<DT> indexer_stride0_index_stride_element1{
      "indexer_stride0_index_stride_element1",
      vector_route_unit_indexer_stride0, 9, 16};
  sfr::Bitfield<DT> indexer_stride0_index_stride_element2{
      "indexer_stride0_index_stride_element2",
      vector_route_unit_indexer_stride0, 9, 32};
  sfr::Bitfield<DT> indexer_stride0_index_stride_element3{
      "indexer_stride0_index_stride_element3",
      vector_route_unit_indexer_stride0, 9, 48};
  sfr::Register<DT> vector_route_unit_indexer_stride1{
      "vector_route_unit_indexer_stride1", *this, 0x20, 0x0};
  sfr::Bitfield<DT> indexer_stride1_index_stride_element4{
      "indexer_stride1_index_stride_element4",
      vector_route_unit_indexer_stride1, 9, 0};
  sfr::Bitfield<DT> indexer_stride1_index_stride_element5{
      "indexer_stride1_index_stride_element5",
      vector_route_unit_indexer_stride1, 9, 16};
  sfr::Bitfield<DT> indexer_stride1_index_stride_element6{
      "indexer_stride1_index_stride_element6",
      vector_route_unit_indexer_stride1, 9, 32};
  sfr::Bitfield<DT> indexer_stride1_index_stride_element7{
      "indexer_stride1_index_stride_element7",
      vector_route_unit_indexer_stride1, 9, 48};
  sfr::Register<DT> vector_route_unit_valid_generator_limit0{
      "vector_route_unit_valid_generator_limit0", *this, 0x28, 0x0};
  sfr::Bitfield<DT> valid_generator_limit0_lowered_limit_element0{
      "valid_generator_limit0_lowered_limit_element0",
      vector_route_unit_valid_generator_limit0, 16, 0};
  sfr::Bitfield<DT> valid_generator_limit0_lowered_limit_element1{
      "valid_generator_limit0_lowered_limit_element1",
      vector_route_unit_valid_generator_limit0, 16, 16};
  sfr::Bitfield<DT> valid_generator_limit0_lowered_limit_element2{
      "valid_generator_limit0_lowered_limit_element2",
      vector_route_unit_valid_generator_limit0, 16, 32};
  sfr::Bitfield<DT> valid_generator_limit0_lowered_limit_element3{
      "valid_generator_limit0_lowered_limit_element3",
      vector_route_unit_valid_generator_limit0, 16, 48};
  sfr::Register<DT> vector_route_unit_valid_generator_limit1{
      "vector_route_unit_valid_generator_limit1", *this, 0x30, 0x0};
  sfr::Bitfield<DT> valid_generator_limit1_lowered_limit_element4{
      "valid_generator_limit1_lowered_limit_element4",
      vector_route_unit_valid_generator_limit1, 16, 0};
  sfr::Bitfield<DT> valid_generator_limit1_lowered_limit_element5{
      "valid_generator_limit1_lowered_limit_element5",
      vector_route_unit_valid_generator_limit1, 16, 16};
  sfr::Bitfield<DT> valid_generator_limit1_lowered_limit_element6{
      "valid_generator_limit1_lowered_limit_element6",
      vector_route_unit_valid_generator_limit1, 16, 32};
  sfr::Bitfield<DT> valid_generator_limit1_lowered_limit_element7{
      "valid_generator_limit1_lowered_limit_element7",
      vector_route_unit_valid_generator_limit1, 16, 48};
  sfr::Register<DT> vector_route_unit_valid_generator_stride0{
      "vector_route_unit_valid_generator_stride0", *this, 0x38, 0x0};
  sfr::Bitfield<DT> valid_generator_stride0_lowered_stride_element0{
      "valid_generator_stride0_lowered_stride_element0",
      vector_route_unit_valid_generator_stride0, 16, 0};
  sfr::Bitfield<DT> valid_generator_stride0_lowered_stride_element1{
      "valid_generator_stride0_lowered_stride_element1",
      vector_route_unit_valid_generator_stride0, 16, 16};
  sfr::Bitfield<DT> valid_generator_stride0_lowered_stride_element2{
      "valid_generator_stride0_lowered_stride_element2",
      vector_route_unit_valid_generator_stride0, 16, 32};
  sfr::Bitfield<DT> valid_generator_stride0_lowered_stride_element3{
      "valid_generator_stride0_lowered_stride_element3",
      vector_route_unit_valid_generator_stride0, 16, 48};
  sfr::Register<DT> vector_route_unit_valid_generator_stride1{
      "vector_route_unit_valid_generator_stride1", *this, 0x40, 0x0};
  sfr::Bitfield<DT> valid_generator_stride1_lowered_stride_element4{
      "valid_generator_stride1_lowered_stride_element4",
      vector_route_unit_valid_generator_stride1, 16, 0};
  sfr::Bitfield<DT> valid_generator_stride1_lowered_stride_element5{
      "valid_generator_stride1_lowered_stride_element5",
      vector_route_unit_valid_generator_stride1, 16, 16};
  sfr::Bitfield<DT> valid_generator_stride1_lowered_stride_element6{
      "valid_generator_stride1_lowered_stride_element6",
      vector_route_unit_valid_generator_stride1, 16, 32};
  sfr::Bitfield<DT> valid_generator_stride1_lowered_stride_element7{
      "valid_generator_stride1_lowered_stride_element7",
      vector_route_unit_valid_generator_stride1, 16, 48};
  sfr::Register<DT> vector_route_unit_valid_generator_original_dim{
      "vector_route_unit_valid_generator_original_dim", *this, 0x48, 0x0};
  sfr::Bitfield<DT> valid_generator_original_dim_allocated_original_dim0{
      "valid_generator_original_dim_allocated_original_dim0",
      vector_route_unit_valid_generator_original_dim, 4, 0};
  sfr::Bitfield<DT> valid_generator_original_dim_allocated_original_dim1{
      "valid_generator_original_dim_allocated_original_dim1",
      vector_route_unit_valid_generator_original_dim, 4, 4};
  sfr::Bitfield<DT> valid_generator_original_dim_allocated_original_dim2{
      "valid_generator_original_dim_allocated_original_dim2",
      vector_route_unit_valid_generator_original_dim, 4, 8};
  sfr::Bitfield<DT> valid_generator_original_dim_allocated_original_dim3{
      "valid_generator_original_dim_allocated_original_dim3",
      vector_route_unit_valid_generator_original_dim, 4, 12};
  sfr::Bitfield<DT> valid_generator_original_dim_allocated_original_dim4{
      "valid_generator_original_dim_allocated_original_dim4",
      vector_route_unit_valid_generator_original_dim, 4, 16};
  sfr::Bitfield<DT> valid_generator_original_dim_allocated_original_dim5{
      "valid_generator_original_dim_allocated_original_dim5",
      vector_route_unit_valid_generator_original_dim, 4, 20};
  sfr::Bitfield<DT> valid_generator_original_dim_allocated_original_dim6{
      "valid_generator_original_dim_allocated_original_dim6",
      vector_route_unit_valid_generator_original_dim, 4, 24};
  sfr::Bitfield<DT> valid_generator_original_dim_allocated_original_dim7{
      "valid_generator_original_dim_allocated_original_dim7",
      vector_route_unit_valid_generator_original_dim, 4, 28};
  sfr::Bitfield<DT> valid_generator_original_dim_original_dim_partition_config0{
      "valid_generator_original_dim_original_dim_partition_config0",
      vector_route_unit_valid_generator_original_dim, 1, 32};
  sfr::Bitfield<DT> valid_generator_original_dim_original_dim_partition_config1{
      "valid_generator_original_dim_original_dim_partition_config1",
      vector_route_unit_valid_generator_original_dim, 1, 34};
  sfr::Bitfield<DT> valid_generator_original_dim_original_dim_partition_config2{
      "valid_generator_original_dim_original_dim_partition_config2",
      vector_route_unit_valid_generator_original_dim, 1, 36};
  sfr::Bitfield<DT> valid_generator_original_dim_original_dim_partition_config3{
      "valid_generator_original_dim_original_dim_partition_config3",
      vector_route_unit_valid_generator_original_dim, 1, 38};
  sfr::Register<DT> vector_route_unit_valid_generator_valid_count{
      "vector_route_unit_valid_generator_valid_count", *this, 0x50, 0x0};
  sfr::Bitfield<DT> valid_generator_valid_count_original_dim_valid_count0{
      "valid_generator_valid_count_original_dim_valid_count0",
      vector_route_unit_valid_generator_valid_count, 16, 0};
  sfr::Bitfield<DT> valid_generator_valid_count_original_dim_valid_count1{
      "valid_generator_valid_count_original_dim_valid_count1",
      vector_route_unit_valid_generator_valid_count, 16, 16};
  sfr::Bitfield<DT> valid_generator_valid_count_original_dim_valid_count2{
      "valid_generator_valid_count_original_dim_valid_count2",
      vector_route_unit_valid_generator_valid_count, 16, 32};
  sfr::Bitfield<DT> valid_generator_valid_count_original_dim_valid_count3{
      "valid_generator_valid_count_original_dim_valid_count3",
      vector_route_unit_valid_generator_valid_count, 16, 48};
  sfr::Register<DT> vector_route_unit_valid_generator_slice_info{
      "vector_route_unit_valid_generator_slice_info", *this, 0x58, 0x0};
  sfr::Bitfield<DT> valid_generator_slice_info_slice_mask0{
      "valid_generator_slice_info_slice_mask0",
      vector_route_unit_valid_generator_slice_info, 8, 0};
  sfr::Bitfield<DT> valid_generator_slice_info_slice_mask1{
      "valid_generator_slice_info_slice_mask1",
      vector_route_unit_valid_generator_slice_info, 8, 8};
  sfr::Bitfield<DT> valid_generator_slice_info_slice_mask2{
      "valid_generator_slice_info_slice_mask2",
      vector_route_unit_valid_generator_slice_info, 8, 16};
  sfr::Bitfield<DT> valid_generator_slice_info_slice_mask3{
      "valid_generator_slice_info_slice_mask3",
      vector_route_unit_valid_generator_slice_info, 8, 24};
  sfr::Bitfield<DT> valid_generator_slice_info_slice_id_match0{
      "valid_generator_slice_info_slice_id_match0",
      vector_route_unit_valid_generator_slice_info, 8, 32};
  sfr::Bitfield<DT> valid_generator_slice_info_slice_id_match1{
      "valid_generator_slice_info_slice_id_match1",
      vector_route_unit_valid_generator_slice_info, 8, 40};
  sfr::Bitfield<DT> valid_generator_slice_info_slice_id_match2{
      "valid_generator_slice_info_slice_id_match2",
      vector_route_unit_valid_generator_slice_info, 8, 48};
  sfr::Bitfield<DT> valid_generator_slice_info_slice_id_match3{
      "valid_generator_slice_info_slice_id_match3",
      vector_route_unit_valid_generator_slice_info, 8, 56};
  sfr::Register<DT> vector_route_unit_compaction_mode{
      "vector_route_unit_compaction_mode", *this, 0x60, 0x0};
  sfr::Bitfield<DT> collect_compaction_mode{
      "collect_compaction_mode", vector_route_unit_compaction_mode, 2, 0};
  sfr::Bitfield<DT> compaction_mode_collect_compaction_cmp_op{
      "compaction_mode_collect_compaction_cmp_op",
      vector_route_unit_compaction_mode, 2, 4};
  sfr::Bitfield<DT> compaction_mode_collect_compaction_execution_id{
      "compaction_mode_collect_compaction_execution_id",
      vector_route_unit_compaction_mode, 4, 8};
  sfr::Bitfield<DT> compaction_mode_collect_compaction_execution_id_mask{
      "compaction_mode_collect_compaction_execution_id_mask",
      vector_route_unit_compaction_mode, 4, 12};
  sfr::Bitfield<DT> cast_compaction_mode{
      "cast_compaction_mode", vector_route_unit_compaction_mode, 3, 16};
  sfr::Bitfield<DT> compaction_mode_cast_compaction_count{
      "compaction_mode_cast_compaction_count",
      vector_route_unit_compaction_mode, 4, 20};
};

template <class DT> class VectorReduceUnitMainContext : public sfr::Block<DT> {
public:
  VectorReduceUnitMainContext() : sfr::Block<DT>::Block(0x11c8, 0x68, false) {}

public:
  sfr::Register<DT> vector_reduce_unit_cluster_route{
      "vector_reduce_unit_cluster_route", *this, 0x0, 0x50000};
  sfr::Bitfield<DT> cluster_route_add_source{
      "cluster_route_add_source", vector_reduce_unit_cluster_route, 3, 0};
  sfr::Bitfield<DT> cluster_route_max_source{
      "cluster_route_max_source", vector_reduce_unit_cluster_route, 3, 4};
  sfr::Bitfield<DT> cluster_route_min_source{
      "cluster_route_min_source", vector_reduce_unit_cluster_route, 3, 8};
  sfr::Bitfield<DT> cluster_route_mul_source{
      "cluster_route_mul_source", vector_reduce_unit_cluster_route, 3, 12};
  sfr::Bitfield<DT> cluster_route_cluster_source{
      "cluster_route_cluster_source", vector_reduce_unit_cluster_route, 3, 16};
  sfr::Register<DT> vector_reduce_unit_mul_control{
      "vector_reduce_unit_mul_control", *this, 0x8, 0x0};
  sfr::Bitfield<DT> mul_control_op_mode{"mul_control_op_mode",
                                        vector_reduce_unit_mul_control, 4, 0};
  sfr::Bitfield<DT> mul_control_arg_mode{"mul_control_arg_mode",
                                         vector_reduce_unit_mul_control, 3, 4};
  sfr::Bitfield<DT> mul_control_reg0_cmp_op{
      "mul_control_reg0_cmp_op", vector_reduce_unit_mul_control, 2, 8};
  sfr::Bitfield<DT> mul_control_reg1_cmp_op{
      "mul_control_reg1_cmp_op", vector_reduce_unit_mul_control, 2, 10};
  sfr::Bitfield<DT> mul_control_reg2_cmp_op{
      "mul_control_reg2_cmp_op", vector_reduce_unit_mul_control, 2, 12};
  sfr::Bitfield<DT> mul_control_rhs_cmp_op{
      "mul_control_rhs_cmp_op", vector_reduce_unit_mul_control, 2, 16};
  sfr::Bitfield<DT> mul_control_reg0_execution_id{
      "mul_control_reg0_execution_id", vector_reduce_unit_mul_control, 4, 24};
  sfr::Bitfield<DT> mul_control_reg0_execution_id_mask{
      "mul_control_reg0_execution_id_mask", vector_reduce_unit_mul_control, 4,
      28};
  sfr::Bitfield<DT> mul_control_reg1_execution_id{
      "mul_control_reg1_execution_id", vector_reduce_unit_mul_control, 4, 32};
  sfr::Bitfield<DT> mul_control_reg1_execution_id_mask{
      "mul_control_reg1_execution_id_mask", vector_reduce_unit_mul_control, 4,
      36};
  sfr::Bitfield<DT> mul_control_reg2_execution_id{
      "mul_control_reg2_execution_id", vector_reduce_unit_mul_control, 4, 40};
  sfr::Bitfield<DT> mul_control_reg2_execution_id_mask{
      "mul_control_reg2_execution_id_mask", vector_reduce_unit_mul_control, 4,
      44};
  sfr::Bitfield<DT> mul_control_io_execution_id{
      "mul_control_io_execution_id", vector_reduce_unit_mul_control, 4, 48};
  sfr::Bitfield<DT> mul_control_io_execution_id_mask{
      "mul_control_io_execution_id_mask", vector_reduce_unit_mul_control, 4,
      52};
  sfr::Register<DT> vector_reduce_unit_mul_data0{"vector_reduce_unit_mul_data0",
                                                 *this, 0x10, 0x0};
  sfr::Bitfield<DT> mul_data0_scalar_register_element0{
      "mul_data0_scalar_register_element0", vector_reduce_unit_mul_data0, 32,
      0};
  sfr::Bitfield<DT> mul_data0_scalar_register_element1{
      "mul_data0_scalar_register_element1", vector_reduce_unit_mul_data0, 32,
      32};
  sfr::Register<DT> vector_reduce_unit_mul_data1{"vector_reduce_unit_mul_data1",
                                                 *this, 0x18, 0x0};
  sfr::Bitfield<DT> mul_data1_scalar_register_element2{
      "mul_data1_scalar_register_element2", vector_reduce_unit_mul_data1, 32,
      0};
  sfr::Register<DT> vector_reduce_unit_add_control{
      "vector_reduce_unit_add_control", *this, 0x20, 0x0};
  sfr::Bitfield<DT> add_control_op_mode{"add_control_op_mode",
                                        vector_reduce_unit_add_control, 4, 0};
  sfr::Bitfield<DT> add_control_arg_mode{"add_control_arg_mode",
                                         vector_reduce_unit_add_control, 3, 4};
  sfr::Bitfield<DT> add_control_reg0_cmp_op{
      "add_control_reg0_cmp_op", vector_reduce_unit_add_control, 2, 8};
  sfr::Bitfield<DT> add_control_reg1_cmp_op{
      "add_control_reg1_cmp_op", vector_reduce_unit_add_control, 2, 10};
  sfr::Bitfield<DT> add_control_reg2_cmp_op{
      "add_control_reg2_cmp_op", vector_reduce_unit_add_control, 2, 12};
  sfr::Bitfield<DT> add_control_rhs_cmp_op{
      "add_control_rhs_cmp_op", vector_reduce_unit_add_control, 2, 16};
  sfr::Bitfield<DT> add_control_reg0_execution_id{
      "add_control_reg0_execution_id", vector_reduce_unit_add_control, 4, 24};
  sfr::Bitfield<DT> add_control_reg0_execution_id_mask{
      "add_control_reg0_execution_id_mask", vector_reduce_unit_add_control, 4,
      28};
  sfr::Bitfield<DT> add_control_reg1_execution_id{
      "add_control_reg1_execution_id", vector_reduce_unit_add_control, 4, 32};
  sfr::Bitfield<DT> add_control_reg1_execution_id_mask{
      "add_control_reg1_execution_id_mask", vector_reduce_unit_add_control, 4,
      36};
  sfr::Bitfield<DT> add_control_reg2_execution_id{
      "add_control_reg2_execution_id", vector_reduce_unit_add_control, 4, 40};
  sfr::Bitfield<DT> add_control_reg2_execution_id_mask{
      "add_control_reg2_execution_id_mask", vector_reduce_unit_add_control, 4,
      44};
  sfr::Bitfield<DT> add_control_io_execution_id{
      "add_control_io_execution_id", vector_reduce_unit_add_control, 4, 48};
  sfr::Bitfield<DT> add_control_io_execution_id_mask{
      "add_control_io_execution_id_mask", vector_reduce_unit_add_control, 4,
      52};
  sfr::Register<DT> vector_reduce_unit_add_data0{"vector_reduce_unit_add_data0",
                                                 *this, 0x28, 0x0};
  sfr::Bitfield<DT> add_data0_scalar_register_element0{
      "add_data0_scalar_register_element0", vector_reduce_unit_add_data0, 32,
      0};
  sfr::Bitfield<DT> add_data0_scalar_register_element1{
      "add_data0_scalar_register_element1", vector_reduce_unit_add_data0, 32,
      32};
  sfr::Register<DT> vector_reduce_unit_add_data1{"vector_reduce_unit_add_data1",
                                                 *this, 0x30, 0x0};
  sfr::Bitfield<DT> add_data1_scalar_register_element2{
      "add_data1_scalar_register_element2", vector_reduce_unit_add_data1, 32,
      0};
  sfr::Register<DT> vector_reduce_unit_max_control{
      "vector_reduce_unit_max_control", *this, 0x38, 0x0};
  sfr::Bitfield<DT> max_control_op_mode{"max_control_op_mode",
                                        vector_reduce_unit_max_control, 4, 0};
  sfr::Bitfield<DT> max_control_arg_mode{"max_control_arg_mode",
                                         vector_reduce_unit_max_control, 3, 4};
  sfr::Bitfield<DT> max_control_reg0_cmp_op{
      "max_control_reg0_cmp_op", vector_reduce_unit_max_control, 2, 8};
  sfr::Bitfield<DT> max_control_reg1_cmp_op{
      "max_control_reg1_cmp_op", vector_reduce_unit_max_control, 2, 10};
  sfr::Bitfield<DT> max_control_reg2_cmp_op{
      "max_control_reg2_cmp_op", vector_reduce_unit_max_control, 2, 12};
  sfr::Bitfield<DT> max_control_rhs_cmp_op{
      "max_control_rhs_cmp_op", vector_reduce_unit_max_control, 2, 16};
  sfr::Bitfield<DT> max_control_reg0_execution_id{
      "max_control_reg0_execution_id", vector_reduce_unit_max_control, 4, 24};
  sfr::Bitfield<DT> max_control_reg0_execution_id_mask{
      "max_control_reg0_execution_id_mask", vector_reduce_unit_max_control, 4,
      28};
  sfr::Bitfield<DT> max_control_reg1_execution_id{
      "max_control_reg1_execution_id", vector_reduce_unit_max_control, 4, 32};
  sfr::Bitfield<DT> max_control_reg1_execution_id_mask{
      "max_control_reg1_execution_id_mask", vector_reduce_unit_max_control, 4,
      36};
  sfr::Bitfield<DT> max_control_reg2_execution_id{
      "max_control_reg2_execution_id", vector_reduce_unit_max_control, 4, 40};
  sfr::Bitfield<DT> max_control_reg2_execution_id_mask{
      "max_control_reg2_execution_id_mask", vector_reduce_unit_max_control, 4,
      44};
  sfr::Bitfield<DT> max_control_io_execution_id{
      "max_control_io_execution_id", vector_reduce_unit_max_control, 4, 48};
  sfr::Bitfield<DT> max_control_io_execution_id_mask{
      "max_control_io_execution_id_mask", vector_reduce_unit_max_control, 4,
      52};
  sfr::Register<DT> vector_reduce_unit_max_data0{"vector_reduce_unit_max_data0",
                                                 *this, 0x40, 0x0};
  sfr::Bitfield<DT> max_data0_scalar_register_element0{
      "max_data0_scalar_register_element0", vector_reduce_unit_max_data0, 32,
      0};
  sfr::Bitfield<DT> max_data0_scalar_register_element1{
      "max_data0_scalar_register_element1", vector_reduce_unit_max_data0, 32,
      32};
  sfr::Register<DT> vector_reduce_unit_max_data1{"vector_reduce_unit_max_data1",
                                                 *this, 0x48, 0x0};
  sfr::Bitfield<DT> max_data1_scalar_register_element2{
      "max_data1_scalar_register_element2", vector_reduce_unit_max_data1, 32,
      0};
  sfr::Register<DT> vector_reduce_unit_min_control{
      "vector_reduce_unit_min_control", *this, 0x50, 0x0};
  sfr::Bitfield<DT> min_control_op_mode{"min_control_op_mode",
                                        vector_reduce_unit_min_control, 4, 0};
  sfr::Bitfield<DT> min_control_arg_mode{"min_control_arg_mode",
                                         vector_reduce_unit_min_control, 3, 4};
  sfr::Bitfield<DT> min_control_reg0_cmp_op{
      "min_control_reg0_cmp_op", vector_reduce_unit_min_control, 2, 8};
  sfr::Bitfield<DT> min_control_reg1_cmp_op{
      "min_control_reg1_cmp_op", vector_reduce_unit_min_control, 2, 10};
  sfr::Bitfield<DT> min_control_reg2_cmp_op{
      "min_control_reg2_cmp_op", vector_reduce_unit_min_control, 2, 12};
  sfr::Bitfield<DT> min_control_rhs_cmp_op{
      "min_control_rhs_cmp_op", vector_reduce_unit_min_control, 2, 16};
  sfr::Bitfield<DT> min_control_reg0_execution_id{
      "min_control_reg0_execution_id", vector_reduce_unit_min_control, 4, 24};
  sfr::Bitfield<DT> min_control_reg0_execution_id_mask{
      "min_control_reg0_execution_id_mask", vector_reduce_unit_min_control, 4,
      28};
  sfr::Bitfield<DT> min_control_reg1_execution_id{
      "min_control_reg1_execution_id", vector_reduce_unit_min_control, 4, 32};
  sfr::Bitfield<DT> min_control_reg1_execution_id_mask{
      "min_control_reg1_execution_id_mask", vector_reduce_unit_min_control, 4,
      36};
  sfr::Bitfield<DT> min_control_reg2_execution_id{
      "min_control_reg2_execution_id", vector_reduce_unit_min_control, 4, 40};
  sfr::Bitfield<DT> min_control_reg2_execution_id_mask{
      "min_control_reg2_execution_id_mask", vector_reduce_unit_min_control, 4,
      44};
  sfr::Bitfield<DT> min_control_io_execution_id{
      "min_control_io_execution_id", vector_reduce_unit_min_control, 4, 48};
  sfr::Bitfield<DT> min_control_io_execution_id_mask{
      "min_control_io_execution_id_mask", vector_reduce_unit_min_control, 4,
      52};
  sfr::Register<DT> vector_reduce_unit_min_data0{"vector_reduce_unit_min_data0",
                                                 *this, 0x58, 0x0};
  sfr::Bitfield<DT> min_data0_scalar_register_element0{
      "min_data0_scalar_register_element0", vector_reduce_unit_min_data0, 32,
      0};
  sfr::Bitfield<DT> min_data0_scalar_register_element1{
      "min_data0_scalar_register_element1", vector_reduce_unit_min_data0, 32,
      32};
  sfr::Register<DT> vector_reduce_unit_min_data1{"vector_reduce_unit_min_data1",
                                                 *this, 0x60, 0x0};
  sfr::Bitfield<DT> min_data1_scalar_register_element2{
      "min_data1_scalar_register_element2", vector_reduce_unit_min_data1, 32,
      0};
};

template <class DT> class VectorRegisterFile : public sfr::Block<DT> {
public:
  VectorRegisterFile() : sfr::Block<DT>::Block(0x190, 0x8, false) {}

public:
  sfr::Register<DT> vector_register_file_write{"vector_register_file_write",
                                               *this, 0x0, 0x0};
  sfr::Bitfield<DT> write_row_base{"write_row_base", vector_register_file_write,
                                   8, 0};
  sfr::Bitfield<DT> write_row_count{"write_row_count",
                                    vector_register_file_write, 9, 16};
  sfr::Bitfield<DT> write_skip_flit_count{"write_skip_flit_count",
                                          vector_register_file_write, 16, 32};
  sfr::Bitfield<DT> write_row_stride{"write_row_stride",
                                     vector_register_file_write, 8, 48};
};

template <class DT>
class VectorArithmeticUnitMainContext : public sfr::Block<DT> {
public:
  VectorArithmeticUnitMainContext()
      : sfr::Block<DT>::Block(0x1230, 0x2f8, false) {}

public:
  sfr::Register<DT> vector_arithmetic_unit_branch_mode{
      "vector_arithmetic_unit_branch_mode", *this, 0x0, 0x0};
  sfr::Bitfield<DT> branch_mode_mode{"branch_mode_mode",
                                     vector_arithmetic_unit_branch_mode, 3, 0};
  sfr::Bitfield<DT> branch_mode_format{
      "branch_mode_format", vector_arithmetic_unit_branch_mode, 2, 4};
  sfr::Bitfield<DT> branch_mode_compare_operation0{
      "branch_mode_compare_operation0", vector_arithmetic_unit_branch_mode, 3,
      8};
  sfr::Bitfield<DT> branch_mode_compare_operation1{
      "branch_mode_compare_operation1", vector_arithmetic_unit_branch_mode, 3,
      12};
  sfr::Bitfield<DT> branch_mode_compare_operation2{
      "branch_mode_compare_operation2", vector_arithmetic_unit_branch_mode, 3,
      16};
  sfr::Bitfield<DT> branch_mode_compare_operation3{
      "branch_mode_compare_operation3", vector_arithmetic_unit_branch_mode, 3,
      20};
  sfr::Bitfield<DT> branch_mode_group_size{
      "branch_mode_group_size", vector_arithmetic_unit_branch_mode, 8, 24};
  sfr::Bitfield<DT> branch_mode_branch_read_base{
      "branch_mode_branch_read_base", vector_arithmetic_unit_branch_mode, 13,
      32};
  sfr::Bitfield<DT> branch_mode_branch_read_limit{
      "branch_mode_branch_read_limit", vector_arithmetic_unit_branch_mode, 16,
      48};
  sfr::Register<DT> vector_arithmetic_unit_branch_data0{
      "vector_arithmetic_unit_branch_data0", *this, 0x8, 0x0};
  sfr::Bitfield<DT> branch_data0_scalar_register_element0{
      "branch_data0_scalar_register_element0",
      vector_arithmetic_unit_branch_data0, 32, 0};
  sfr::Bitfield<DT> branch_data0_scalar_register_element1{
      "branch_data0_scalar_register_element1",
      vector_arithmetic_unit_branch_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_branch_data1{
      "vector_arithmetic_unit_branch_data1", *this, 0x10, 0x0};
  sfr::Bitfield<DT> branch_data1_scalar_register_element2{
      "branch_data1_scalar_register_element2",
      vector_arithmetic_unit_branch_data1, 32, 0};
  sfr::Bitfield<DT> branch_data1_scalar_register_element3{
      "branch_data1_scalar_register_element3",
      vector_arithmetic_unit_branch_data1, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_register_file_write_mode{
      "vector_arithmetic_unit_register_file_write_mode", *this, 0x18, 0x0};
  sfr::Bitfield<DT> register_file_write_mode_branch_write_mode{
      "register_file_write_mode_branch_write_mode",
      vector_arithmetic_unit_register_file_write_mode, 1, 0};
  sfr::Bitfield<DT> register_file_write_mode_branch_write_base{
      "register_file_write_mode_branch_write_base",
      vector_arithmetic_unit_register_file_write_mode, 13, 16};
  sfr::Bitfield<DT> register_file_write_mode_branch_write_limit{
      "register_file_write_mode_branch_write_limit",
      vector_arithmetic_unit_register_file_write_mode, 16, 32};
  sfr::Bitfield<DT> register_file_write_mode_write_cmp_op{
      "register_file_write_mode_write_cmp_op",
      vector_arithmetic_unit_register_file_write_mode, 2, 48};
  sfr::Bitfield<DT> register_file_write_mode_write_execution_id{
      "register_file_write_mode_write_execution_id",
      vector_arithmetic_unit_register_file_write_mode, 4, 56};
  sfr::Bitfield<DT> register_file_write_mode_write_execution_id_mask{
      "register_file_write_mode_write_execution_id_mask",
      vector_arithmetic_unit_register_file_write_mode, 4, 60};
  sfr::Register<DT> vector_arithmetic_unit_logic_cluster_route{
      "vector_arithmetic_unit_logic_cluster_route", *this, 0x20, 0x600000};
  sfr::Bitfield<DT> logic_cluster_route_logic_and_source{
      "logic_cluster_route_logic_and_source",
      vector_arithmetic_unit_logic_cluster_route, 3, 0};
  sfr::Bitfield<DT> logic_cluster_route_logic_or_source{
      "logic_cluster_route_logic_or_source",
      vector_arithmetic_unit_logic_cluster_route, 3, 4};
  sfr::Bitfield<DT> logic_cluster_route_logic_xor_source{
      "logic_cluster_route_logic_xor_source",
      vector_arithmetic_unit_logic_cluster_route, 3, 8};
  sfr::Bitfield<DT> logic_cluster_route_logic_left_shift_source{
      "logic_cluster_route_logic_left_shift_source",
      vector_arithmetic_unit_logic_cluster_route, 3, 12};
  sfr::Bitfield<DT> logic_cluster_route_logic_right_shift_source{
      "logic_cluster_route_logic_right_shift_source",
      vector_arithmetic_unit_logic_cluster_route, 3, 16};
  sfr::Bitfield<DT> logic_cluster_route_logic_cluster_source{
      "logic_cluster_route_logic_cluster_source",
      vector_arithmetic_unit_logic_cluster_route, 3, 20};
  sfr::Register<DT> vector_arithmetic_unit_logic_and_control{
      "vector_arithmetic_unit_logic_and_control", *this, 0x28, 0x0};
  sfr::Bitfield<DT> logic_and_control_op_mode{
      "logic_and_control_op_mode", vector_arithmetic_unit_logic_and_control, 4,
      0};
  sfr::Bitfield<DT> logic_and_control_arg_mode{
      "logic_and_control_arg_mode", vector_arithmetic_unit_logic_and_control, 3,
      4};
  sfr::Bitfield<DT> logic_and_control_reg0_cmp_op{
      "logic_and_control_reg0_cmp_op", vector_arithmetic_unit_logic_and_control,
      2, 8};
  sfr::Bitfield<DT> logic_and_control_reg1_cmp_op{
      "logic_and_control_reg1_cmp_op", vector_arithmetic_unit_logic_and_control,
      2, 10};
  sfr::Bitfield<DT> logic_and_control_reg2_cmp_op{
      "logic_and_control_reg2_cmp_op", vector_arithmetic_unit_logic_and_control,
      2, 12};
  sfr::Bitfield<DT> logic_and_control_rf_cmp_op{
      "logic_and_control_rf_cmp_op", vector_arithmetic_unit_logic_and_control,
      2, 16};
  sfr::Bitfield<DT> logic_and_control_reg0_execution_id{
      "logic_and_control_reg0_execution_id",
      vector_arithmetic_unit_logic_and_control, 4, 24};
  sfr::Bitfield<DT> logic_and_control_reg0_execution_id_mask{
      "logic_and_control_reg0_execution_id_mask",
      vector_arithmetic_unit_logic_and_control, 4, 28};
  sfr::Bitfield<DT> logic_and_control_reg1_execution_id{
      "logic_and_control_reg1_execution_id",
      vector_arithmetic_unit_logic_and_control, 4, 32};
  sfr::Bitfield<DT> logic_and_control_reg1_execution_id_mask{
      "logic_and_control_reg1_execution_id_mask",
      vector_arithmetic_unit_logic_and_control, 4, 36};
  sfr::Bitfield<DT> logic_and_control_reg2_execution_id{
      "logic_and_control_reg2_execution_id",
      vector_arithmetic_unit_logic_and_control, 4, 40};
  sfr::Bitfield<DT> logic_and_control_reg2_execution_id_mask{
      "logic_and_control_reg2_execution_id_mask",
      vector_arithmetic_unit_logic_and_control, 4, 44};
  sfr::Bitfield<DT> logic_and_control_rf_execution_id{
      "logic_and_control_rf_execution_id",
      vector_arithmetic_unit_logic_and_control, 4, 48};
  sfr::Bitfield<DT> logic_and_control_rf_execution_id_mask{
      "logic_and_control_rf_execution_id_mask",
      vector_arithmetic_unit_logic_and_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_logic_and_data0{
      "vector_arithmetic_unit_logic_and_data0", *this, 0x30, 0x0};
  sfr::Bitfield<DT> logic_and_data0_scalar_register_element0{
      "logic_and_data0_scalar_register_element0",
      vector_arithmetic_unit_logic_and_data0, 32, 0};
  sfr::Bitfield<DT> logic_and_data0_scalar_register_element1{
      "logic_and_data0_scalar_register_element1",
      vector_arithmetic_unit_logic_and_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_logic_and_data1{
      "vector_arithmetic_unit_logic_and_data1", *this, 0x38, 0x0};
  sfr::Bitfield<DT> logic_and_data1_scalar_register_element2{
      "logic_and_data1_scalar_register_element2",
      vector_arithmetic_unit_logic_and_data1, 32, 0};
  sfr::Register<DT> vector_arithmetic_unit_logic_or_control{
      "vector_arithmetic_unit_logic_or_control", *this, 0x40, 0x0};
  sfr::Bitfield<DT> logic_or_control_op_mode{
      "logic_or_control_op_mode", vector_arithmetic_unit_logic_or_control, 4,
      0};
  sfr::Bitfield<DT> logic_or_control_arg_mode{
      "logic_or_control_arg_mode", vector_arithmetic_unit_logic_or_control, 3,
      4};
  sfr::Bitfield<DT> logic_or_control_reg0_cmp_op{
      "logic_or_control_reg0_cmp_op", vector_arithmetic_unit_logic_or_control,
      2, 8};
  sfr::Bitfield<DT> logic_or_control_reg1_cmp_op{
      "logic_or_control_reg1_cmp_op", vector_arithmetic_unit_logic_or_control,
      2, 10};
  sfr::Bitfield<DT> logic_or_control_reg2_cmp_op{
      "logic_or_control_reg2_cmp_op", vector_arithmetic_unit_logic_or_control,
      2, 12};
  sfr::Bitfield<DT> logic_or_control_rf_cmp_op{
      "logic_or_control_rf_cmp_op", vector_arithmetic_unit_logic_or_control, 2,
      16};
  sfr::Bitfield<DT> logic_or_control_reg0_execution_id{
      "logic_or_control_reg0_execution_id",
      vector_arithmetic_unit_logic_or_control, 4, 24};
  sfr::Bitfield<DT> logic_or_control_reg0_execution_id_mask{
      "logic_or_control_reg0_execution_id_mask",
      vector_arithmetic_unit_logic_or_control, 4, 28};
  sfr::Bitfield<DT> logic_or_control_reg1_execution_id{
      "logic_or_control_reg1_execution_id",
      vector_arithmetic_unit_logic_or_control, 4, 32};
  sfr::Bitfield<DT> logic_or_control_reg1_execution_id_mask{
      "logic_or_control_reg1_execution_id_mask",
      vector_arithmetic_unit_logic_or_control, 4, 36};
  sfr::Bitfield<DT> logic_or_control_reg2_execution_id{
      "logic_or_control_reg2_execution_id",
      vector_arithmetic_unit_logic_or_control, 4, 40};
  sfr::Bitfield<DT> logic_or_control_reg2_execution_id_mask{
      "logic_or_control_reg2_execution_id_mask",
      vector_arithmetic_unit_logic_or_control, 4, 44};
  sfr::Bitfield<DT> logic_or_control_rf_execution_id{
      "logic_or_control_rf_execution_id",
      vector_arithmetic_unit_logic_or_control, 4, 48};
  sfr::Bitfield<DT> logic_or_control_rf_execution_id_mask{
      "logic_or_control_rf_execution_id_mask",
      vector_arithmetic_unit_logic_or_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_logic_or_data0{
      "vector_arithmetic_unit_logic_or_data0", *this, 0x48, 0x0};
  sfr::Bitfield<DT> logic_or_data0_scalar_register_element0{
      "logic_or_data0_scalar_register_element0",
      vector_arithmetic_unit_logic_or_data0, 32, 0};
  sfr::Bitfield<DT> logic_or_data0_scalar_register_element1{
      "logic_or_data0_scalar_register_element1",
      vector_arithmetic_unit_logic_or_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_logic_or_data1{
      "vector_arithmetic_unit_logic_or_data1", *this, 0x50, 0x0};
  sfr::Bitfield<DT> logic_or_data1_scalar_register_element2{
      "logic_or_data1_scalar_register_element2",
      vector_arithmetic_unit_logic_or_data1, 32, 0};
  sfr::Register<DT> vector_arithmetic_unit_logic_xor_control{
      "vector_arithmetic_unit_logic_xor_control", *this, 0x58, 0x0};
  sfr::Bitfield<DT> logic_xor_control_op_mode{
      "logic_xor_control_op_mode", vector_arithmetic_unit_logic_xor_control, 4,
      0};
  sfr::Bitfield<DT> logic_xor_control_arg_mode{
      "logic_xor_control_arg_mode", vector_arithmetic_unit_logic_xor_control, 3,
      4};
  sfr::Bitfield<DT> logic_xor_control_reg0_cmp_op{
      "logic_xor_control_reg0_cmp_op", vector_arithmetic_unit_logic_xor_control,
      2, 8};
  sfr::Bitfield<DT> logic_xor_control_reg1_cmp_op{
      "logic_xor_control_reg1_cmp_op", vector_arithmetic_unit_logic_xor_control,
      2, 10};
  sfr::Bitfield<DT> logic_xor_control_reg2_cmp_op{
      "logic_xor_control_reg2_cmp_op", vector_arithmetic_unit_logic_xor_control,
      2, 12};
  sfr::Bitfield<DT> logic_xor_control_rf_cmp_op{
      "logic_xor_control_rf_cmp_op", vector_arithmetic_unit_logic_xor_control,
      2, 16};
  sfr::Bitfield<DT> logic_xor_control_reg0_execution_id{
      "logic_xor_control_reg0_execution_id",
      vector_arithmetic_unit_logic_xor_control, 4, 24};
  sfr::Bitfield<DT> logic_xor_control_reg0_execution_id_mask{
      "logic_xor_control_reg0_execution_id_mask",
      vector_arithmetic_unit_logic_xor_control, 4, 28};
  sfr::Bitfield<DT> logic_xor_control_reg1_execution_id{
      "logic_xor_control_reg1_execution_id",
      vector_arithmetic_unit_logic_xor_control, 4, 32};
  sfr::Bitfield<DT> logic_xor_control_reg1_execution_id_mask{
      "logic_xor_control_reg1_execution_id_mask",
      vector_arithmetic_unit_logic_xor_control, 4, 36};
  sfr::Bitfield<DT> logic_xor_control_reg2_execution_id{
      "logic_xor_control_reg2_execution_id",
      vector_arithmetic_unit_logic_xor_control, 4, 40};
  sfr::Bitfield<DT> logic_xor_control_reg2_execution_id_mask{
      "logic_xor_control_reg2_execution_id_mask",
      vector_arithmetic_unit_logic_xor_control, 4, 44};
  sfr::Bitfield<DT> logic_xor_control_rf_execution_id{
      "logic_xor_control_rf_execution_id",
      vector_arithmetic_unit_logic_xor_control, 4, 48};
  sfr::Bitfield<DT> logic_xor_control_rf_execution_id_mask{
      "logic_xor_control_rf_execution_id_mask",
      vector_arithmetic_unit_logic_xor_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_logic_xor_data0{
      "vector_arithmetic_unit_logic_xor_data0", *this, 0x60, 0x0};
  sfr::Bitfield<DT> logic_xor_data0_scalar_register_element0{
      "logic_xor_data0_scalar_register_element0",
      vector_arithmetic_unit_logic_xor_data0, 32, 0};
  sfr::Bitfield<DT> logic_xor_data0_scalar_register_element1{
      "logic_xor_data0_scalar_register_element1",
      vector_arithmetic_unit_logic_xor_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_logic_xor_data1{
      "vector_arithmetic_unit_logic_xor_data1", *this, 0x68, 0x0};
  sfr::Bitfield<DT> logic_xor_data1_scalar_register_element2{
      "logic_xor_data1_scalar_register_element2",
      vector_arithmetic_unit_logic_xor_data1, 32, 0};
  sfr::Register<DT> vector_arithmetic_unit_logic_left_shift_control{
      "vector_arithmetic_unit_logic_left_shift_control", *this, 0x70, 0x0};
  sfr::Bitfield<DT> logic_left_shift_control_op_mode{
      "logic_left_shift_control_op_mode",
      vector_arithmetic_unit_logic_left_shift_control, 4, 0};
  sfr::Bitfield<DT> logic_left_shift_control_arg_mode{
      "logic_left_shift_control_arg_mode",
      vector_arithmetic_unit_logic_left_shift_control, 3, 4};
  sfr::Bitfield<DT> logic_left_shift_control_reg0_cmp_op{
      "logic_left_shift_control_reg0_cmp_op",
      vector_arithmetic_unit_logic_left_shift_control, 2, 8};
  sfr::Bitfield<DT> logic_left_shift_control_reg1_cmp_op{
      "logic_left_shift_control_reg1_cmp_op",
      vector_arithmetic_unit_logic_left_shift_control, 2, 10};
  sfr::Bitfield<DT> logic_left_shift_control_reg2_cmp_op{
      "logic_left_shift_control_reg2_cmp_op",
      vector_arithmetic_unit_logic_left_shift_control, 2, 12};
  sfr::Bitfield<DT> logic_left_shift_control_rf_cmp_op{
      "logic_left_shift_control_rf_cmp_op",
      vector_arithmetic_unit_logic_left_shift_control, 2, 16};
  sfr::Bitfield<DT> logic_left_shift_control_reg0_execution_id{
      "logic_left_shift_control_reg0_execution_id",
      vector_arithmetic_unit_logic_left_shift_control, 4, 24};
  sfr::Bitfield<DT> logic_left_shift_control_reg0_execution_id_mask{
      "logic_left_shift_control_reg0_execution_id_mask",
      vector_arithmetic_unit_logic_left_shift_control, 4, 28};
  sfr::Bitfield<DT> logic_left_shift_control_reg1_execution_id{
      "logic_left_shift_control_reg1_execution_id",
      vector_arithmetic_unit_logic_left_shift_control, 4, 32};
  sfr::Bitfield<DT> logic_left_shift_control_reg1_execution_id_mask{
      "logic_left_shift_control_reg1_execution_id_mask",
      vector_arithmetic_unit_logic_left_shift_control, 4, 36};
  sfr::Bitfield<DT> logic_left_shift_control_reg2_execution_id{
      "logic_left_shift_control_reg2_execution_id",
      vector_arithmetic_unit_logic_left_shift_control, 4, 40};
  sfr::Bitfield<DT> logic_left_shift_control_reg2_execution_id_mask{
      "logic_left_shift_control_reg2_execution_id_mask",
      vector_arithmetic_unit_logic_left_shift_control, 4, 44};
  sfr::Bitfield<DT> logic_left_shift_control_rf_execution_id{
      "logic_left_shift_control_rf_execution_id",
      vector_arithmetic_unit_logic_left_shift_control, 4, 48};
  sfr::Bitfield<DT> logic_left_shift_control_rf_execution_id_mask{
      "logic_left_shift_control_rf_execution_id_mask",
      vector_arithmetic_unit_logic_left_shift_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_logic_left_shift_data0{
      "vector_arithmetic_unit_logic_left_shift_data0", *this, 0x78, 0x0};
  sfr::Bitfield<DT> logic_left_shift_data0_scalar_register_element0{
      "logic_left_shift_data0_scalar_register_element0",
      vector_arithmetic_unit_logic_left_shift_data0, 32, 0};
  sfr::Bitfield<DT> logic_left_shift_data0_scalar_register_element1{
      "logic_left_shift_data0_scalar_register_element1",
      vector_arithmetic_unit_logic_left_shift_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_logic_left_shift_data1{
      "vector_arithmetic_unit_logic_left_shift_data1", *this, 0x80, 0x0};
  sfr::Bitfield<DT> logic_left_shift_data1_scalar_register_element2{
      "logic_left_shift_data1_scalar_register_element2",
      vector_arithmetic_unit_logic_left_shift_data1, 32, 0};
  sfr::Register<DT> vector_arithmetic_unit_logic_right_shift_control{
      "vector_arithmetic_unit_logic_right_shift_control", *this, 0x88, 0x0};
  sfr::Bitfield<DT> logic_right_shift_control_op_mode{
      "logic_right_shift_control_op_mode",
      vector_arithmetic_unit_logic_right_shift_control, 4, 0};
  sfr::Bitfield<DT> logic_right_shift_control_arg_mode{
      "logic_right_shift_control_arg_mode",
      vector_arithmetic_unit_logic_right_shift_control, 3, 4};
  sfr::Bitfield<DT> logic_right_shift_control_reg0_cmp_op{
      "logic_right_shift_control_reg0_cmp_op",
      vector_arithmetic_unit_logic_right_shift_control, 2, 8};
  sfr::Bitfield<DT> logic_right_shift_control_reg1_cmp_op{
      "logic_right_shift_control_reg1_cmp_op",
      vector_arithmetic_unit_logic_right_shift_control, 2, 10};
  sfr::Bitfield<DT> logic_right_shift_control_reg2_cmp_op{
      "logic_right_shift_control_reg2_cmp_op",
      vector_arithmetic_unit_logic_right_shift_control, 2, 12};
  sfr::Bitfield<DT> logic_right_shift_control_rf_cmp_op{
      "logic_right_shift_control_rf_cmp_op",
      vector_arithmetic_unit_logic_right_shift_control, 2, 16};
  sfr::Bitfield<DT> logic_right_shift_control_reg0_execution_id{
      "logic_right_shift_control_reg0_execution_id",
      vector_arithmetic_unit_logic_right_shift_control, 4, 24};
  sfr::Bitfield<DT> logic_right_shift_control_reg0_execution_id_mask{
      "logic_right_shift_control_reg0_execution_id_mask",
      vector_arithmetic_unit_logic_right_shift_control, 4, 28};
  sfr::Bitfield<DT> logic_right_shift_control_reg1_execution_id{
      "logic_right_shift_control_reg1_execution_id",
      vector_arithmetic_unit_logic_right_shift_control, 4, 32};
  sfr::Bitfield<DT> logic_right_shift_control_reg1_execution_id_mask{
      "logic_right_shift_control_reg1_execution_id_mask",
      vector_arithmetic_unit_logic_right_shift_control, 4, 36};
  sfr::Bitfield<DT> logic_right_shift_control_reg2_execution_id{
      "logic_right_shift_control_reg2_execution_id",
      vector_arithmetic_unit_logic_right_shift_control, 4, 40};
  sfr::Bitfield<DT> logic_right_shift_control_reg2_execution_id_mask{
      "logic_right_shift_control_reg2_execution_id_mask",
      vector_arithmetic_unit_logic_right_shift_control, 4, 44};
  sfr::Bitfield<DT> logic_right_shift_control_rf_execution_id{
      "logic_right_shift_control_rf_execution_id",
      vector_arithmetic_unit_logic_right_shift_control, 4, 48};
  sfr::Bitfield<DT> logic_right_shift_control_rf_execution_id_mask{
      "logic_right_shift_control_rf_execution_id_mask",
      vector_arithmetic_unit_logic_right_shift_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_logic_right_shift_data0{
      "vector_arithmetic_unit_logic_right_shift_data0", *this, 0x90, 0x0};
  sfr::Bitfield<DT> logic_right_shift_data0_scalar_register_element0{
      "logic_right_shift_data0_scalar_register_element0",
      vector_arithmetic_unit_logic_right_shift_data0, 32, 0};
  sfr::Bitfield<DT> logic_right_shift_data0_scalar_register_element1{
      "logic_right_shift_data0_scalar_register_element1",
      vector_arithmetic_unit_logic_right_shift_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_logic_right_shift_data1{
      "vector_arithmetic_unit_logic_right_shift_data1", *this, 0x98, 0x0};
  sfr::Bitfield<DT> logic_right_shift_data1_scalar_register_element2{
      "logic_right_shift_data1_scalar_register_element2",
      vector_arithmetic_unit_logic_right_shift_data1, 32, 0};
  sfr::Register<DT> vector_arithmetic_unit_fxp_cluster_route{
      "vector_arithmetic_unit_fxp_cluster_route", *this, 0xa0, 0x50000};
  sfr::Bitfield<DT> fxp_cluster_route_fxp_add_source{
      "fxp_cluster_route_fxp_add_source",
      vector_arithmetic_unit_fxp_cluster_route, 3, 0};
  sfr::Bitfield<DT> fxp_cluster_route_fxp_left_shift_source{
      "fxp_cluster_route_fxp_left_shift_source",
      vector_arithmetic_unit_fxp_cluster_route, 3, 4};
  sfr::Bitfield<DT> fxp_cluster_route_fxp_mul_source{
      "fxp_cluster_route_fxp_mul_source",
      vector_arithmetic_unit_fxp_cluster_route, 3, 8};
  sfr::Bitfield<DT> fxp_cluster_route_fxp_right_shift_source{
      "fxp_cluster_route_fxp_right_shift_source",
      vector_arithmetic_unit_fxp_cluster_route, 3, 12};
  sfr::Bitfield<DT> fxp_cluster_route_fxp_cluster_source{
      "fxp_cluster_route_fxp_cluster_source",
      vector_arithmetic_unit_fxp_cluster_route, 3, 16};
  sfr::Register<DT> vector_arithmetic_unit_fxp_add_control{
      "vector_arithmetic_unit_fxp_add_control", *this, 0xa8, 0x0};
  sfr::Bitfield<DT> fxp_add_control_op_mode{
      "fxp_add_control_op_mode", vector_arithmetic_unit_fxp_add_control, 4, 0};
  sfr::Bitfield<DT> fxp_add_control_arg_mode{
      "fxp_add_control_arg_mode", vector_arithmetic_unit_fxp_add_control, 3, 4};
  sfr::Bitfield<DT> fxp_add_control_reg0_cmp_op{
      "fxp_add_control_reg0_cmp_op", vector_arithmetic_unit_fxp_add_control, 2,
      8};
  sfr::Bitfield<DT> fxp_add_control_reg1_cmp_op{
      "fxp_add_control_reg1_cmp_op", vector_arithmetic_unit_fxp_add_control, 2,
      10};
  sfr::Bitfield<DT> fxp_add_control_reg2_cmp_op{
      "fxp_add_control_reg2_cmp_op", vector_arithmetic_unit_fxp_add_control, 2,
      12};
  sfr::Bitfield<DT> fxp_add_control_rf_cmp_op{
      "fxp_add_control_rf_cmp_op", vector_arithmetic_unit_fxp_add_control, 2,
      16};
  sfr::Bitfield<DT> fxp_add_control_reg0_execution_id{
      "fxp_add_control_reg0_execution_id",
      vector_arithmetic_unit_fxp_add_control, 4, 24};
  sfr::Bitfield<DT> fxp_add_control_reg0_execution_id_mask{
      "fxp_add_control_reg0_execution_id_mask",
      vector_arithmetic_unit_fxp_add_control, 4, 28};
  sfr::Bitfield<DT> fxp_add_control_reg1_execution_id{
      "fxp_add_control_reg1_execution_id",
      vector_arithmetic_unit_fxp_add_control, 4, 32};
  sfr::Bitfield<DT> fxp_add_control_reg1_execution_id_mask{
      "fxp_add_control_reg1_execution_id_mask",
      vector_arithmetic_unit_fxp_add_control, 4, 36};
  sfr::Bitfield<DT> fxp_add_control_reg2_execution_id{
      "fxp_add_control_reg2_execution_id",
      vector_arithmetic_unit_fxp_add_control, 4, 40};
  sfr::Bitfield<DT> fxp_add_control_reg2_execution_id_mask{
      "fxp_add_control_reg2_execution_id_mask",
      vector_arithmetic_unit_fxp_add_control, 4, 44};
  sfr::Bitfield<DT> fxp_add_control_rf_execution_id{
      "fxp_add_control_rf_execution_id", vector_arithmetic_unit_fxp_add_control,
      4, 48};
  sfr::Bitfield<DT> fxp_add_control_rf_execution_id_mask{
      "fxp_add_control_rf_execution_id_mask",
      vector_arithmetic_unit_fxp_add_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_fxp_add_data0{
      "vector_arithmetic_unit_fxp_add_data0", *this, 0xb0, 0x0};
  sfr::Bitfield<DT> fxp_add_data0_scalar_register_element0{
      "fxp_add_data0_scalar_register_element0",
      vector_arithmetic_unit_fxp_add_data0, 32, 0};
  sfr::Bitfield<DT> fxp_add_data0_scalar_register_element1{
      "fxp_add_data0_scalar_register_element1",
      vector_arithmetic_unit_fxp_add_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_fxp_add_data1{
      "vector_arithmetic_unit_fxp_add_data1", *this, 0xb8, 0x0};
  sfr::Bitfield<DT> fxp_add_data1_scalar_register_element2{
      "fxp_add_data1_scalar_register_element2",
      vector_arithmetic_unit_fxp_add_data1, 32, 0};
  sfr::Register<DT> vector_arithmetic_unit_fxp_left_shift_control{
      "vector_arithmetic_unit_fxp_left_shift_control", *this, 0xc0, 0x0};
  sfr::Bitfield<DT> fxp_left_shift_control_op_mode{
      "fxp_left_shift_control_op_mode",
      vector_arithmetic_unit_fxp_left_shift_control, 4, 0};
  sfr::Bitfield<DT> fxp_left_shift_control_arg_mode{
      "fxp_left_shift_control_arg_mode",
      vector_arithmetic_unit_fxp_left_shift_control, 3, 4};
  sfr::Bitfield<DT> fxp_left_shift_control_reg0_cmp_op{
      "fxp_left_shift_control_reg0_cmp_op",
      vector_arithmetic_unit_fxp_left_shift_control, 2, 8};
  sfr::Bitfield<DT> fxp_left_shift_control_reg1_cmp_op{
      "fxp_left_shift_control_reg1_cmp_op",
      vector_arithmetic_unit_fxp_left_shift_control, 2, 10};
  sfr::Bitfield<DT> fxp_left_shift_control_reg2_cmp_op{
      "fxp_left_shift_control_reg2_cmp_op",
      vector_arithmetic_unit_fxp_left_shift_control, 2, 12};
  sfr::Bitfield<DT> fxp_left_shift_control_rf_cmp_op{
      "fxp_left_shift_control_rf_cmp_op",
      vector_arithmetic_unit_fxp_left_shift_control, 2, 16};
  sfr::Bitfield<DT> fxp_left_shift_control_reg0_execution_id{
      "fxp_left_shift_control_reg0_execution_id",
      vector_arithmetic_unit_fxp_left_shift_control, 4, 24};
  sfr::Bitfield<DT> fxp_left_shift_control_reg0_execution_id_mask{
      "fxp_left_shift_control_reg0_execution_id_mask",
      vector_arithmetic_unit_fxp_left_shift_control, 4, 28};
  sfr::Bitfield<DT> fxp_left_shift_control_reg1_execution_id{
      "fxp_left_shift_control_reg1_execution_id",
      vector_arithmetic_unit_fxp_left_shift_control, 4, 32};
  sfr::Bitfield<DT> fxp_left_shift_control_reg1_execution_id_mask{
      "fxp_left_shift_control_reg1_execution_id_mask",
      vector_arithmetic_unit_fxp_left_shift_control, 4, 36};
  sfr::Bitfield<DT> fxp_left_shift_control_reg2_execution_id{
      "fxp_left_shift_control_reg2_execution_id",
      vector_arithmetic_unit_fxp_left_shift_control, 4, 40};
  sfr::Bitfield<DT> fxp_left_shift_control_reg2_execution_id_mask{
      "fxp_left_shift_control_reg2_execution_id_mask",
      vector_arithmetic_unit_fxp_left_shift_control, 4, 44};
  sfr::Bitfield<DT> fxp_left_shift_control_rf_execution_id{
      "fxp_left_shift_control_rf_execution_id",
      vector_arithmetic_unit_fxp_left_shift_control, 4, 48};
  sfr::Bitfield<DT> fxp_left_shift_control_rf_execution_id_mask{
      "fxp_left_shift_control_rf_execution_id_mask",
      vector_arithmetic_unit_fxp_left_shift_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_fxp_left_shift_data0{
      "vector_arithmetic_unit_fxp_left_shift_data0", *this, 0xc8, 0x0};
  sfr::Bitfield<DT> fxp_left_shift_data0_scalar_register_element0{
      "fxp_left_shift_data0_scalar_register_element0",
      vector_arithmetic_unit_fxp_left_shift_data0, 32, 0};
  sfr::Bitfield<DT> fxp_left_shift_data0_scalar_register_element1{
      "fxp_left_shift_data0_scalar_register_element1",
      vector_arithmetic_unit_fxp_left_shift_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_fxp_left_shift_data1{
      "vector_arithmetic_unit_fxp_left_shift_data1", *this, 0xd0, 0x0};
  sfr::Bitfield<DT> fxp_left_shift_data1_scalar_register_element2{
      "fxp_left_shift_data1_scalar_register_element2",
      vector_arithmetic_unit_fxp_left_shift_data1, 32, 0};
  sfr::Register<DT> vector_arithmetic_unit_fxp_mul_control{
      "vector_arithmetic_unit_fxp_mul_control", *this, 0xd8, 0x0};
  sfr::Bitfield<DT> fxp_mul_control_op_mode{
      "fxp_mul_control_op_mode", vector_arithmetic_unit_fxp_mul_control, 4, 0};
  sfr::Bitfield<DT> fxp_mul_control_arg_mode{
      "fxp_mul_control_arg_mode", vector_arithmetic_unit_fxp_mul_control, 3, 4};
  sfr::Bitfield<DT> fxp_mul_control_reg0_cmp_op{
      "fxp_mul_control_reg0_cmp_op", vector_arithmetic_unit_fxp_mul_control, 2,
      8};
  sfr::Bitfield<DT> fxp_mul_control_reg1_cmp_op{
      "fxp_mul_control_reg1_cmp_op", vector_arithmetic_unit_fxp_mul_control, 2,
      10};
  sfr::Bitfield<DT> fxp_mul_control_reg2_cmp_op{
      "fxp_mul_control_reg2_cmp_op", vector_arithmetic_unit_fxp_mul_control, 2,
      12};
  sfr::Bitfield<DT> fxp_mul_control_rf_cmp_op{
      "fxp_mul_control_rf_cmp_op", vector_arithmetic_unit_fxp_mul_control, 2,
      16};
  sfr::Bitfield<DT> fxp_mul_control_reg0_execution_id{
      "fxp_mul_control_reg0_execution_id",
      vector_arithmetic_unit_fxp_mul_control, 4, 24};
  sfr::Bitfield<DT> fxp_mul_control_reg0_execution_id_mask{
      "fxp_mul_control_reg0_execution_id_mask",
      vector_arithmetic_unit_fxp_mul_control, 4, 28};
  sfr::Bitfield<DT> fxp_mul_control_reg1_execution_id{
      "fxp_mul_control_reg1_execution_id",
      vector_arithmetic_unit_fxp_mul_control, 4, 32};
  sfr::Bitfield<DT> fxp_mul_control_reg1_execution_id_mask{
      "fxp_mul_control_reg1_execution_id_mask",
      vector_arithmetic_unit_fxp_mul_control, 4, 36};
  sfr::Bitfield<DT> fxp_mul_control_reg2_execution_id{
      "fxp_mul_control_reg2_execution_id",
      vector_arithmetic_unit_fxp_mul_control, 4, 40};
  sfr::Bitfield<DT> fxp_mul_control_reg2_execution_id_mask{
      "fxp_mul_control_reg2_execution_id_mask",
      vector_arithmetic_unit_fxp_mul_control, 4, 44};
  sfr::Bitfield<DT> fxp_mul_control_rf_execution_id{
      "fxp_mul_control_rf_execution_id", vector_arithmetic_unit_fxp_mul_control,
      4, 48};
  sfr::Bitfield<DT> fxp_mul_control_rf_execution_id_mask{
      "fxp_mul_control_rf_execution_id_mask",
      vector_arithmetic_unit_fxp_mul_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_fxp_mul_data0{
      "vector_arithmetic_unit_fxp_mul_data0", *this, 0xe0, 0x0};
  sfr::Bitfield<DT> fxp_mul_data0_scalar_register_element0{
      "fxp_mul_data0_scalar_register_element0",
      vector_arithmetic_unit_fxp_mul_data0, 32, 0};
  sfr::Bitfield<DT> fxp_mul_data0_scalar_register_element1{
      "fxp_mul_data0_scalar_register_element1",
      vector_arithmetic_unit_fxp_mul_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_fxp_mul_data1{
      "vector_arithmetic_unit_fxp_mul_data1", *this, 0xe8, 0x0};
  sfr::Bitfield<DT> fxp_mul_data1_scalar_register_element2{
      "fxp_mul_data1_scalar_register_element2",
      vector_arithmetic_unit_fxp_mul_data1, 32, 0};
  sfr::Register<DT> vector_arithmetic_unit_fxp_right_shift_control{
      "vector_arithmetic_unit_fxp_right_shift_control", *this, 0xf0, 0x0};
  sfr::Bitfield<DT> fxp_right_shift_control_op_mode{
      "fxp_right_shift_control_op_mode",
      vector_arithmetic_unit_fxp_right_shift_control, 4, 0};
  sfr::Bitfield<DT> fxp_right_shift_control_arg_mode{
      "fxp_right_shift_control_arg_mode",
      vector_arithmetic_unit_fxp_right_shift_control, 3, 4};
  sfr::Bitfield<DT> fxp_right_shift_control_reg0_cmp_op{
      "fxp_right_shift_control_reg0_cmp_op",
      vector_arithmetic_unit_fxp_right_shift_control, 2, 8};
  sfr::Bitfield<DT> fxp_right_shift_control_reg1_cmp_op{
      "fxp_right_shift_control_reg1_cmp_op",
      vector_arithmetic_unit_fxp_right_shift_control, 2, 10};
  sfr::Bitfield<DT> fxp_right_shift_control_reg2_cmp_op{
      "fxp_right_shift_control_reg2_cmp_op",
      vector_arithmetic_unit_fxp_right_shift_control, 2, 12};
  sfr::Bitfield<DT> fxp_right_shift_control_rf_cmp_op{
      "fxp_right_shift_control_rf_cmp_op",
      vector_arithmetic_unit_fxp_right_shift_control, 2, 16};
  sfr::Bitfield<DT> fxp_right_shift_control_reg0_execution_id{
      "fxp_right_shift_control_reg0_execution_id",
      vector_arithmetic_unit_fxp_right_shift_control, 4, 24};
  sfr::Bitfield<DT> fxp_right_shift_control_reg0_execution_id_mask{
      "fxp_right_shift_control_reg0_execution_id_mask",
      vector_arithmetic_unit_fxp_right_shift_control, 4, 28};
  sfr::Bitfield<DT> fxp_right_shift_control_reg1_execution_id{
      "fxp_right_shift_control_reg1_execution_id",
      vector_arithmetic_unit_fxp_right_shift_control, 4, 32};
  sfr::Bitfield<DT> fxp_right_shift_control_reg1_execution_id_mask{
      "fxp_right_shift_control_reg1_execution_id_mask",
      vector_arithmetic_unit_fxp_right_shift_control, 4, 36};
  sfr::Bitfield<DT> fxp_right_shift_control_reg2_execution_id{
      "fxp_right_shift_control_reg2_execution_id",
      vector_arithmetic_unit_fxp_right_shift_control, 4, 40};
  sfr::Bitfield<DT> fxp_right_shift_control_reg2_execution_id_mask{
      "fxp_right_shift_control_reg2_execution_id_mask",
      vector_arithmetic_unit_fxp_right_shift_control, 4, 44};
  sfr::Bitfield<DT> fxp_right_shift_control_rf_execution_id{
      "fxp_right_shift_control_rf_execution_id",
      vector_arithmetic_unit_fxp_right_shift_control, 4, 48};
  sfr::Bitfield<DT> fxp_right_shift_control_rf_execution_id_mask{
      "fxp_right_shift_control_rf_execution_id_mask",
      vector_arithmetic_unit_fxp_right_shift_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_fxp_right_shift_data0{
      "vector_arithmetic_unit_fxp_right_shift_data0", *this, 0xf8, 0x0};
  sfr::Bitfield<DT> fxp_right_shift_data0_scalar_register_element0{
      "fxp_right_shift_data0_scalar_register_element0",
      vector_arithmetic_unit_fxp_right_shift_data0, 32, 0};
  sfr::Bitfield<DT> fxp_right_shift_data0_scalar_register_element1{
      "fxp_right_shift_data0_scalar_register_element1",
      vector_arithmetic_unit_fxp_right_shift_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_fxp_right_shift_data1{
      "vector_arithmetic_unit_fxp_right_shift_data1", *this, 0x100, 0x0};
  sfr::Bitfield<DT> fxp_right_shift_data1_scalar_register_element2{
      "fxp_right_shift_data1_scalar_register_element2",
      vector_arithmetic_unit_fxp_right_shift_data1, 32, 0};
  sfr::Register<DT> vector_arithmetic_unit_fp_cluster_route{
      "vector_arithmetic_unit_fp_cluster_route", *this, 0x108, 0x600000};
  sfr::Bitfield<DT> fp_cluster_route_fp_fma_source{
      "fp_cluster_route_fp_fma_source", vector_arithmetic_unit_fp_cluster_route,
      3, 0};
  sfr::Bitfield<DT> fp_cluster_route_fp_fpu_source{
      "fp_cluster_route_fp_fpu_source", vector_arithmetic_unit_fp_cluster_route,
      3, 4};
  sfr::Bitfield<DT> fp_cluster_route_fp_exp_source{
      "fp_cluster_route_fp_exp_source", vector_arithmetic_unit_fp_cluster_route,
      3, 8};
  sfr::Bitfield<DT> fp_cluster_route_fp_mul0_source{
      "fp_cluster_route_fp_mul0_source",
      vector_arithmetic_unit_fp_cluster_route, 3, 12};
  sfr::Bitfield<DT> fp_cluster_route_fp_mul1_source{
      "fp_cluster_route_fp_mul1_source",
      vector_arithmetic_unit_fp_cluster_route, 3, 16};
  sfr::Bitfield<DT> fp_cluster_route_fp_cluster_source{
      "fp_cluster_route_fp_cluster_source",
      vector_arithmetic_unit_fp_cluster_route, 3, 20};
  sfr::Register<DT> vector_arithmetic_unit_fp_fma_control{
      "vector_arithmetic_unit_fp_fma_control", *this, 0x110, 0x0};
  sfr::Bitfield<DT> fp_fma_control_op_mode{
      "fp_fma_control_op_mode", vector_arithmetic_unit_fp_fma_control, 4, 0};
  sfr::Bitfield<DT> fp_fma_control_arg_mode{
      "fp_fma_control_arg_mode", vector_arithmetic_unit_fp_fma_control, 3, 4};
  sfr::Bitfield<DT> fp_fma_control_reg0_cmp_op{
      "fp_fma_control_reg0_cmp_op", vector_arithmetic_unit_fp_fma_control, 2,
      8};
  sfr::Bitfield<DT> fp_fma_control_reg1_cmp_op{
      "fp_fma_control_reg1_cmp_op", vector_arithmetic_unit_fp_fma_control, 2,
      10};
  sfr::Bitfield<DT> fp_fma_control_reg2_cmp_op{
      "fp_fma_control_reg2_cmp_op", vector_arithmetic_unit_fp_fma_control, 2,
      12};
  sfr::Bitfield<DT> fp_fma_control_rf_cmp_op{
      "fp_fma_control_rf_cmp_op", vector_arithmetic_unit_fp_fma_control, 2, 16};
  sfr::Bitfield<DT> fp_fma_control_reg0_execution_id{
      "fp_fma_control_reg0_execution_id", vector_arithmetic_unit_fp_fma_control,
      4, 24};
  sfr::Bitfield<DT> fp_fma_control_reg0_execution_id_mask{
      "fp_fma_control_reg0_execution_id_mask",
      vector_arithmetic_unit_fp_fma_control, 4, 28};
  sfr::Bitfield<DT> fp_fma_control_reg1_execution_id{
      "fp_fma_control_reg1_execution_id", vector_arithmetic_unit_fp_fma_control,
      4, 32};
  sfr::Bitfield<DT> fp_fma_control_reg1_execution_id_mask{
      "fp_fma_control_reg1_execution_id_mask",
      vector_arithmetic_unit_fp_fma_control, 4, 36};
  sfr::Bitfield<DT> fp_fma_control_reg2_execution_id{
      "fp_fma_control_reg2_execution_id", vector_arithmetic_unit_fp_fma_control,
      4, 40};
  sfr::Bitfield<DT> fp_fma_control_reg2_execution_id_mask{
      "fp_fma_control_reg2_execution_id_mask",
      vector_arithmetic_unit_fp_fma_control, 4, 44};
  sfr::Bitfield<DT> fp_fma_control_rf_execution_id{
      "fp_fma_control_rf_execution_id", vector_arithmetic_unit_fp_fma_control,
      4, 48};
  sfr::Bitfield<DT> fp_fma_control_rf_execution_id_mask{
      "fp_fma_control_rf_execution_id_mask",
      vector_arithmetic_unit_fp_fma_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_fp_fma_data0{
      "vector_arithmetic_unit_fp_fma_data0", *this, 0x118, 0x0};
  sfr::Bitfield<DT> fp_fma_data0_scalar_register_element0{
      "fp_fma_data0_scalar_register_element0",
      vector_arithmetic_unit_fp_fma_data0, 32, 0};
  sfr::Bitfield<DT> fp_fma_data0_scalar_register_element1{
      "fp_fma_data0_scalar_register_element1",
      vector_arithmetic_unit_fp_fma_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_fp_fma_data1{
      "vector_arithmetic_unit_fp_fma_data1", *this, 0x120, 0x0};
  sfr::Bitfield<DT> fp_fma_data1_scalar_register_element2{
      "fp_fma_data1_scalar_register_element2",
      vector_arithmetic_unit_fp_fma_data1, 32, 0};
  sfr::Bitfield<DT> fp_fma_data1_secondary_scalar_register_element0{
      "fp_fma_data1_secondary_scalar_register_element0",
      vector_arithmetic_unit_fp_fma_data1, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_fp_fma_data2{
      "vector_arithmetic_unit_fp_fma_data2", *this, 0x128, 0x0};
  sfr::Bitfield<DT> fp_fma_data2_secondary_scalar_register_element1{
      "fp_fma_data2_secondary_scalar_register_element1",
      vector_arithmetic_unit_fp_fma_data2, 32, 0};
  sfr::Bitfield<DT> fp_fma_data2_secondary_scalar_register_element2{
      "fp_fma_data2_secondary_scalar_register_element2",
      vector_arithmetic_unit_fp_fma_data2, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_fp_fpu_control{
      "vector_arithmetic_unit_fp_fpu_control", *this, 0x130, 0x0};
  sfr::Bitfield<DT> fp_fpu_control_op_mode{
      "fp_fpu_control_op_mode", vector_arithmetic_unit_fp_fpu_control, 4, 0};
  sfr::Bitfield<DT> fp_fpu_control_arg_mode{
      "fp_fpu_control_arg_mode", vector_arithmetic_unit_fp_fpu_control, 3, 4};
  sfr::Bitfield<DT> fp_fpu_control_reg0_cmp_op{
      "fp_fpu_control_reg0_cmp_op", vector_arithmetic_unit_fp_fpu_control, 2,
      8};
  sfr::Bitfield<DT> fp_fpu_control_reg1_cmp_op{
      "fp_fpu_control_reg1_cmp_op", vector_arithmetic_unit_fp_fpu_control, 2,
      10};
  sfr::Bitfield<DT> fp_fpu_control_reg2_cmp_op{
      "fp_fpu_control_reg2_cmp_op", vector_arithmetic_unit_fp_fpu_control, 2,
      12};
  sfr::Bitfield<DT> fp_fpu_control_rf_cmp_op{
      "fp_fpu_control_rf_cmp_op", vector_arithmetic_unit_fp_fpu_control, 2, 16};
  sfr::Bitfield<DT> fp_fpu_control_reg0_execution_id{
      "fp_fpu_control_reg0_execution_id", vector_arithmetic_unit_fp_fpu_control,
      4, 24};
  sfr::Bitfield<DT> fp_fpu_control_reg0_execution_id_mask{
      "fp_fpu_control_reg0_execution_id_mask",
      vector_arithmetic_unit_fp_fpu_control, 4, 28};
  sfr::Bitfield<DT> fp_fpu_control_reg1_execution_id{
      "fp_fpu_control_reg1_execution_id", vector_arithmetic_unit_fp_fpu_control,
      4, 32};
  sfr::Bitfield<DT> fp_fpu_control_reg1_execution_id_mask{
      "fp_fpu_control_reg1_execution_id_mask",
      vector_arithmetic_unit_fp_fpu_control, 4, 36};
  sfr::Bitfield<DT> fp_fpu_control_reg2_execution_id{
      "fp_fpu_control_reg2_execution_id", vector_arithmetic_unit_fp_fpu_control,
      4, 40};
  sfr::Bitfield<DT> fp_fpu_control_reg2_execution_id_mask{
      "fp_fpu_control_reg2_execution_id_mask",
      vector_arithmetic_unit_fp_fpu_control, 4, 44};
  sfr::Bitfield<DT> fp_fpu_control_rf_execution_id{
      "fp_fpu_control_rf_execution_id", vector_arithmetic_unit_fp_fpu_control,
      4, 48};
  sfr::Bitfield<DT> fp_fpu_control_rf_execution_id_mask{
      "fp_fpu_control_rf_execution_id_mask",
      vector_arithmetic_unit_fp_fpu_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_fp_fpu_data0{
      "vector_arithmetic_unit_fp_fpu_data0", *this, 0x138, 0x0};
  sfr::Bitfield<DT> fp_fpu_data0_scalar_register_element0{
      "fp_fpu_data0_scalar_register_element0",
      vector_arithmetic_unit_fp_fpu_data0, 32, 0};
  sfr::Bitfield<DT> fp_fpu_data0_scalar_register_element1{
      "fp_fpu_data0_scalar_register_element1",
      vector_arithmetic_unit_fp_fpu_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_fp_fpu_data1{
      "vector_arithmetic_unit_fp_fpu_data1", *this, 0x140, 0x0};
  sfr::Bitfield<DT> fp_fpu_data1_scalar_register_element2{
      "fp_fpu_data1_scalar_register_element2",
      vector_arithmetic_unit_fp_fpu_data1, 32, 0};
  sfr::Register<DT> vector_arithmetic_unit_fp_exp_control{
      "vector_arithmetic_unit_fp_exp_control", *this, 0x148, 0x0};
  sfr::Bitfield<DT> fp_exp_control_op_mode{
      "fp_exp_control_op_mode", vector_arithmetic_unit_fp_exp_control, 4, 0};
  sfr::Bitfield<DT> fp_exp_control_arg_mode{
      "fp_exp_control_arg_mode", vector_arithmetic_unit_fp_exp_control, 3, 4};
  sfr::Bitfield<DT> fp_exp_control_reg0_cmp_op{
      "fp_exp_control_reg0_cmp_op", vector_arithmetic_unit_fp_exp_control, 2,
      8};
  sfr::Bitfield<DT> fp_exp_control_reg1_cmp_op{
      "fp_exp_control_reg1_cmp_op", vector_arithmetic_unit_fp_exp_control, 2,
      10};
  sfr::Bitfield<DT> fp_exp_control_reg2_cmp_op{
      "fp_exp_control_reg2_cmp_op", vector_arithmetic_unit_fp_exp_control, 2,
      12};
  sfr::Bitfield<DT> fp_exp_control_rf_cmp_op{
      "fp_exp_control_rf_cmp_op", vector_arithmetic_unit_fp_exp_control, 2, 16};
  sfr::Bitfield<DT> fp_exp_control_reg0_execution_id{
      "fp_exp_control_reg0_execution_id", vector_arithmetic_unit_fp_exp_control,
      4, 24};
  sfr::Bitfield<DT> fp_exp_control_reg0_execution_id_mask{
      "fp_exp_control_reg0_execution_id_mask",
      vector_arithmetic_unit_fp_exp_control, 4, 28};
  sfr::Bitfield<DT> fp_exp_control_reg1_execution_id{
      "fp_exp_control_reg1_execution_id", vector_arithmetic_unit_fp_exp_control,
      4, 32};
  sfr::Bitfield<DT> fp_exp_control_reg1_execution_id_mask{
      "fp_exp_control_reg1_execution_id_mask",
      vector_arithmetic_unit_fp_exp_control, 4, 36};
  sfr::Bitfield<DT> fp_exp_control_reg2_execution_id{
      "fp_exp_control_reg2_execution_id", vector_arithmetic_unit_fp_exp_control,
      4, 40};
  sfr::Bitfield<DT> fp_exp_control_reg2_execution_id_mask{
      "fp_exp_control_reg2_execution_id_mask",
      vector_arithmetic_unit_fp_exp_control, 4, 44};
  sfr::Bitfield<DT> fp_exp_control_rf_execution_id{
      "fp_exp_control_rf_execution_id", vector_arithmetic_unit_fp_exp_control,
      4, 48};
  sfr::Bitfield<DT> fp_exp_control_rf_execution_id_mask{
      "fp_exp_control_rf_execution_id_mask",
      vector_arithmetic_unit_fp_exp_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_fp_mul0_control{
      "vector_arithmetic_unit_fp_mul0_control", *this, 0x150, 0x0};
  sfr::Bitfield<DT> fp_mul0_control_op_mode{
      "fp_mul0_control_op_mode", vector_arithmetic_unit_fp_mul0_control, 4, 0};
  sfr::Bitfield<DT> fp_mul0_control_arg_mode{
      "fp_mul0_control_arg_mode", vector_arithmetic_unit_fp_mul0_control, 3, 4};
  sfr::Bitfield<DT> fp_mul0_control_reg0_cmp_op{
      "fp_mul0_control_reg0_cmp_op", vector_arithmetic_unit_fp_mul0_control, 2,
      8};
  sfr::Bitfield<DT> fp_mul0_control_reg1_cmp_op{
      "fp_mul0_control_reg1_cmp_op", vector_arithmetic_unit_fp_mul0_control, 2,
      10};
  sfr::Bitfield<DT> fp_mul0_control_reg2_cmp_op{
      "fp_mul0_control_reg2_cmp_op", vector_arithmetic_unit_fp_mul0_control, 2,
      12};
  sfr::Bitfield<DT> fp_mul0_control_rf_cmp_op{
      "fp_mul0_control_rf_cmp_op", vector_arithmetic_unit_fp_mul0_control, 2,
      16};
  sfr::Bitfield<DT> fp_mul0_control_reg0_execution_id{
      "fp_mul0_control_reg0_execution_id",
      vector_arithmetic_unit_fp_mul0_control, 4, 24};
  sfr::Bitfield<DT> fp_mul0_control_reg0_execution_id_mask{
      "fp_mul0_control_reg0_execution_id_mask",
      vector_arithmetic_unit_fp_mul0_control, 4, 28};
  sfr::Bitfield<DT> fp_mul0_control_reg1_execution_id{
      "fp_mul0_control_reg1_execution_id",
      vector_arithmetic_unit_fp_mul0_control, 4, 32};
  sfr::Bitfield<DT> fp_mul0_control_reg1_execution_id_mask{
      "fp_mul0_control_reg1_execution_id_mask",
      vector_arithmetic_unit_fp_mul0_control, 4, 36};
  sfr::Bitfield<DT> fp_mul0_control_reg2_execution_id{
      "fp_mul0_control_reg2_execution_id",
      vector_arithmetic_unit_fp_mul0_control, 4, 40};
  sfr::Bitfield<DT> fp_mul0_control_reg2_execution_id_mask{
      "fp_mul0_control_reg2_execution_id_mask",
      vector_arithmetic_unit_fp_mul0_control, 4, 44};
  sfr::Bitfield<DT> fp_mul0_control_rf_execution_id{
      "fp_mul0_control_rf_execution_id", vector_arithmetic_unit_fp_mul0_control,
      4, 48};
  sfr::Bitfield<DT> fp_mul0_control_rf_execution_id_mask{
      "fp_mul0_control_rf_execution_id_mask",
      vector_arithmetic_unit_fp_mul0_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_fp_mul0_data0{
      "vector_arithmetic_unit_fp_mul0_data0", *this, 0x158, 0x0};
  sfr::Bitfield<DT> fp_mul0_data0_scalar_register_element0{
      "fp_mul0_data0_scalar_register_element0",
      vector_arithmetic_unit_fp_mul0_data0, 32, 0};
  sfr::Bitfield<DT> fp_mul0_data0_scalar_register_element1{
      "fp_mul0_data0_scalar_register_element1",
      vector_arithmetic_unit_fp_mul0_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_fp_mul0_data1{
      "vector_arithmetic_unit_fp_mul0_data1", *this, 0x160, 0x0};
  sfr::Bitfield<DT> fp_mul0_data1_scalar_register_element2{
      "fp_mul0_data1_scalar_register_element2",
      vector_arithmetic_unit_fp_mul0_data1, 32, 0};
  sfr::Register<DT> vector_arithmetic_unit_fp_mul1_control{
      "vector_arithmetic_unit_fp_mul1_control", *this, 0x168, 0x0};
  sfr::Bitfield<DT> fp_mul1_control_op_mode{
      "fp_mul1_control_op_mode", vector_arithmetic_unit_fp_mul1_control, 4, 0};
  sfr::Bitfield<DT> fp_mul1_control_arg_mode{
      "fp_mul1_control_arg_mode", vector_arithmetic_unit_fp_mul1_control, 3, 4};
  sfr::Bitfield<DT> fp_mul1_control_reg0_cmp_op{
      "fp_mul1_control_reg0_cmp_op", vector_arithmetic_unit_fp_mul1_control, 2,
      8};
  sfr::Bitfield<DT> fp_mul1_control_reg1_cmp_op{
      "fp_mul1_control_reg1_cmp_op", vector_arithmetic_unit_fp_mul1_control, 2,
      10};
  sfr::Bitfield<DT> fp_mul1_control_reg2_cmp_op{
      "fp_mul1_control_reg2_cmp_op", vector_arithmetic_unit_fp_mul1_control, 2,
      12};
  sfr::Bitfield<DT> fp_mul1_control_rf_cmp_op{
      "fp_mul1_control_rf_cmp_op", vector_arithmetic_unit_fp_mul1_control, 2,
      16};
  sfr::Bitfield<DT> fp_mul1_control_reg0_execution_id{
      "fp_mul1_control_reg0_execution_id",
      vector_arithmetic_unit_fp_mul1_control, 4, 24};
  sfr::Bitfield<DT> fp_mul1_control_reg0_execution_id_mask{
      "fp_mul1_control_reg0_execution_id_mask",
      vector_arithmetic_unit_fp_mul1_control, 4, 28};
  sfr::Bitfield<DT> fp_mul1_control_reg1_execution_id{
      "fp_mul1_control_reg1_execution_id",
      vector_arithmetic_unit_fp_mul1_control, 4, 32};
  sfr::Bitfield<DT> fp_mul1_control_reg1_execution_id_mask{
      "fp_mul1_control_reg1_execution_id_mask",
      vector_arithmetic_unit_fp_mul1_control, 4, 36};
  sfr::Bitfield<DT> fp_mul1_control_reg2_execution_id{
      "fp_mul1_control_reg2_execution_id",
      vector_arithmetic_unit_fp_mul1_control, 4, 40};
  sfr::Bitfield<DT> fp_mul1_control_reg2_execution_id_mask{
      "fp_mul1_control_reg2_execution_id_mask",
      vector_arithmetic_unit_fp_mul1_control, 4, 44};
  sfr::Bitfield<DT> fp_mul1_control_rf_execution_id{
      "fp_mul1_control_rf_execution_id", vector_arithmetic_unit_fp_mul1_control,
      4, 48};
  sfr::Bitfield<DT> fp_mul1_control_rf_execution_id_mask{
      "fp_mul1_control_rf_execution_id_mask",
      vector_arithmetic_unit_fp_mul1_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_fp_mul1_data0{
      "vector_arithmetic_unit_fp_mul1_data0", *this, 0x170, 0x0};
  sfr::Bitfield<DT> fp_mul1_data0_scalar_register_element0{
      "fp_mul1_data0_scalar_register_element0",
      vector_arithmetic_unit_fp_mul1_data0, 32, 0};
  sfr::Bitfield<DT> fp_mul1_data0_scalar_register_element1{
      "fp_mul1_data0_scalar_register_element1",
      vector_arithmetic_unit_fp_mul1_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_fp_mul1_data1{
      "vector_arithmetic_unit_fp_mul1_data1", *this, 0x178, 0x0};
  sfr::Bitfield<DT> fp_mul1_data1_scalar_register_element2{
      "fp_mul1_data1_scalar_register_element2",
      vector_arithmetic_unit_fp_mul1_data1, 32, 0};
  sfr::Register<DT> vector_arithmetic_unit_reduce_layer_mode{
      "vector_arithmetic_unit_reduce_layer_mode", *this, 0x180, 0x0};
  sfr::Bitfield<DT> reduce_layer_mode_reduce_data_path{
      "reduce_layer_mode_reduce_data_path",
      vector_arithmetic_unit_reduce_layer_mode, 1, 0};
  sfr::Bitfield<DT> reduce_layer_mode_reduce_rows{
      "reduce_layer_mode_reduce_rows", vector_arithmetic_unit_reduce_layer_mode,
      2, 2};
  sfr::Bitfield<DT> reduce_layer_mode_reduce_tree_depth{
      "reduce_layer_mode_reduce_tree_depth",
      vector_arithmetic_unit_reduce_layer_mode, 2, 4};
  sfr::Bitfield<DT> reduce_layer_mode_acc_mode{
      "reduce_layer_mode_acc_mode", vector_arithmetic_unit_reduce_layer_mode, 3,
      8};
  sfr::Bitfield<DT> reduce_layer_mode_acc_indexer_proceed{
      "reduce_layer_mode_acc_indexer_proceed",
      vector_arithmetic_unit_reduce_layer_mode, 1, 12};
  sfr::Bitfield<DT> reduce_layer_mode_fxp_shift_rounding_mode{
      "reduce_layer_mode_fxp_shift_rounding_mode",
      vector_arithmetic_unit_reduce_layer_mode, 2, 14};
  sfr::Bitfield<DT> reduce_layer_mode_reduce_row0_op_mode{
      "reduce_layer_mode_reduce_row0_op_mode",
      vector_arithmetic_unit_reduce_layer_mode, 3, 16};
  sfr::Bitfield<DT> reduce_layer_mode_reduce_row1_op_mode{
      "reduce_layer_mode_reduce_row1_op_mode",
      vector_arithmetic_unit_reduce_layer_mode, 3, 20};
  sfr::Bitfield<DT> reduce_layer_mode_accumulation_limit{
      "reduce_layer_mode_accumulation_limit",
      vector_arithmetic_unit_reduce_layer_mode, 16, 24};
  sfr::Bitfield<DT> reduce_layer_mode_acc_indexer_base{
      "reduce_layer_mode_acc_indexer_base",
      vector_arithmetic_unit_reduce_layer_mode, 4, 40};
  sfr::Register<DT> vector_arithmetic_unit_reduce_layer_acc_init{
      "vector_arithmetic_unit_reduce_layer_acc_init", *this, 0x188, 0x0};
  sfr::Bitfield<DT> reduce_layer_acc_init_reduce_row0_acc_init{
      "reduce_layer_acc_init_reduce_row0_acc_init",
      vector_arithmetic_unit_reduce_layer_acc_init, 32, 0};
  sfr::Bitfield<DT> reduce_layer_acc_init_reduce_row1_acc_init{
      "reduce_layer_acc_init_reduce_row1_acc_init",
      vector_arithmetic_unit_reduce_layer_acc_init, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_reduce_layer_acc_limit0{
      "vector_arithmetic_unit_reduce_layer_acc_limit0", *this, 0x190, 0x0};
  sfr::Bitfield<DT> reduce_layer_acc_limit0_acc_indexer_limit_element0{
      "reduce_layer_acc_limit0_acc_indexer_limit_element0",
      vector_arithmetic_unit_reduce_layer_acc_limit0, 16, 0};
  sfr::Bitfield<DT> reduce_layer_acc_limit0_acc_indexer_limit_element1{
      "reduce_layer_acc_limit0_acc_indexer_limit_element1",
      vector_arithmetic_unit_reduce_layer_acc_limit0, 16, 16};
  sfr::Bitfield<DT> reduce_layer_acc_limit0_acc_indexer_limit_element2{
      "reduce_layer_acc_limit0_acc_indexer_limit_element2",
      vector_arithmetic_unit_reduce_layer_acc_limit0, 16, 32};
  sfr::Bitfield<DT> reduce_layer_acc_limit0_acc_indexer_limit_element3{
      "reduce_layer_acc_limit0_acc_indexer_limit_element3",
      vector_arithmetic_unit_reduce_layer_acc_limit0, 16, 48};
  sfr::Register<DT> vector_arithmetic_unit_reduce_layer_acc_limit1{
      "vector_arithmetic_unit_reduce_layer_acc_limit1", *this, 0x198, 0x0};
  sfr::Bitfield<DT> reduce_layer_acc_limit1_acc_indexer_limit_element4{
      "reduce_layer_acc_limit1_acc_indexer_limit_element4",
      vector_arithmetic_unit_reduce_layer_acc_limit1, 16, 0};
  sfr::Bitfield<DT> reduce_layer_acc_limit1_acc_indexer_limit_element5{
      "reduce_layer_acc_limit1_acc_indexer_limit_element5",
      vector_arithmetic_unit_reduce_layer_acc_limit1, 16, 16};
  sfr::Bitfield<DT> reduce_layer_acc_limit1_acc_indexer_limit_element6{
      "reduce_layer_acc_limit1_acc_indexer_limit_element6",
      vector_arithmetic_unit_reduce_layer_acc_limit1, 16, 32};
  sfr::Bitfield<DT> reduce_layer_acc_limit1_acc_indexer_limit_element7{
      "reduce_layer_acc_limit1_acc_indexer_limit_element7",
      vector_arithmetic_unit_reduce_layer_acc_limit1, 16, 48};
  sfr::Register<DT> vector_arithmetic_unit_reduce_layer_acc_stride{
      "vector_arithmetic_unit_reduce_layer_acc_stride", *this, 0x1a0, 0x0};
  sfr::Bitfield<DT> reduce_layer_acc_stride_acc_indexer_stride_element0{
      "reduce_layer_acc_stride_acc_indexer_stride_element0",
      vector_arithmetic_unit_reduce_layer_acc_stride, 4, 0};
  sfr::Bitfield<DT> reduce_layer_acc_stride_acc_indexer_stride_element1{
      "reduce_layer_acc_stride_acc_indexer_stride_element1",
      vector_arithmetic_unit_reduce_layer_acc_stride, 4, 8};
  sfr::Bitfield<DT> reduce_layer_acc_stride_acc_indexer_stride_element2{
      "reduce_layer_acc_stride_acc_indexer_stride_element2",
      vector_arithmetic_unit_reduce_layer_acc_stride, 4, 16};
  sfr::Bitfield<DT> reduce_layer_acc_stride_acc_indexer_stride_element3{
      "reduce_layer_acc_stride_acc_indexer_stride_element3",
      vector_arithmetic_unit_reduce_layer_acc_stride, 4, 24};
  sfr::Bitfield<DT> reduce_layer_acc_stride_acc_indexer_stride_element4{
      "reduce_layer_acc_stride_acc_indexer_stride_element4",
      vector_arithmetic_unit_reduce_layer_acc_stride, 4, 32};
  sfr::Bitfield<DT> reduce_layer_acc_stride_acc_indexer_stride_element5{
      "reduce_layer_acc_stride_acc_indexer_stride_element5",
      vector_arithmetic_unit_reduce_layer_acc_stride, 4, 40};
  sfr::Bitfield<DT> reduce_layer_acc_stride_acc_indexer_stride_element6{
      "reduce_layer_acc_stride_acc_indexer_stride_element6",
      vector_arithmetic_unit_reduce_layer_acc_stride, 4, 48};
  sfr::Bitfield<DT> reduce_layer_acc_stride_acc_indexer_stride_element7{
      "reduce_layer_acc_stride_acc_indexer_stride_element7",
      vector_arithmetic_unit_reduce_layer_acc_stride, 4, 56};
  sfr::Register<DT> vector_arithmetic_unit_fp_div_control{
      "vector_arithmetic_unit_fp_div_control", *this, 0x1a8, 0x0};
  sfr::Bitfield<DT> fp_div_control_op_mode{
      "fp_div_control_op_mode", vector_arithmetic_unit_fp_div_control, 4, 0};
  sfr::Bitfield<DT> fp_div_control_arg_mode{
      "fp_div_control_arg_mode", vector_arithmetic_unit_fp_div_control, 3, 4};
  sfr::Bitfield<DT> fp_div_control_reg0_cmp_op{
      "fp_div_control_reg0_cmp_op", vector_arithmetic_unit_fp_div_control, 2,
      8};
  sfr::Bitfield<DT> fp_div_control_reg1_cmp_op{
      "fp_div_control_reg1_cmp_op", vector_arithmetic_unit_fp_div_control, 2,
      10};
  sfr::Bitfield<DT> fp_div_control_reg2_cmp_op{
      "fp_div_control_reg2_cmp_op", vector_arithmetic_unit_fp_div_control, 2,
      12};
  sfr::Bitfield<DT> fp_div_control_rf_cmp_op{
      "fp_div_control_rf_cmp_op", vector_arithmetic_unit_fp_div_control, 2, 16};
  sfr::Bitfield<DT> fp_div_control_acc_cmp_op{
      "fp_div_control_acc_cmp_op", vector_arithmetic_unit_fp_div_control, 2,
      18};
  sfr::Bitfield<DT> fp_div_control_reg0_execution_id{
      "fp_div_control_reg0_execution_id", vector_arithmetic_unit_fp_div_control,
      4, 24};
  sfr::Bitfield<DT> fp_div_control_reg0_execution_id_mask{
      "fp_div_control_reg0_execution_id_mask",
      vector_arithmetic_unit_fp_div_control, 4, 28};
  sfr::Bitfield<DT> fp_div_control_reg1_execution_id{
      "fp_div_control_reg1_execution_id", vector_arithmetic_unit_fp_div_control,
      4, 32};
  sfr::Bitfield<DT> fp_div_control_reg1_execution_id_mask{
      "fp_div_control_reg1_execution_id_mask",
      vector_arithmetic_unit_fp_div_control, 4, 36};
  sfr::Bitfield<DT> fp_div_control_reg2_execution_id{
      "fp_div_control_reg2_execution_id", vector_arithmetic_unit_fp_div_control,
      4, 40};
  sfr::Bitfield<DT> fp_div_control_reg2_execution_id_mask{
      "fp_div_control_reg2_execution_id_mask",
      vector_arithmetic_unit_fp_div_control, 4, 44};
  sfr::Bitfield<DT> fp_div_control_rf_execution_id{
      "fp_div_control_rf_execution_id", vector_arithmetic_unit_fp_div_control,
      4, 48};
  sfr::Bitfield<DT> fp_div_control_rf_execution_id_mask{
      "fp_div_control_rf_execution_id_mask",
      vector_arithmetic_unit_fp_div_control, 4, 52};
  sfr::Bitfield<DT> fp_div_control_acc_execution_id{
      "fp_div_control_acc_execution_id", vector_arithmetic_unit_fp_div_control,
      4, 56};
  sfr::Bitfield<DT> fp_div_control_acc_execution_id_mask{
      "fp_div_control_acc_execution_id_mask",
      vector_arithmetic_unit_fp_div_control, 4, 60};
  sfr::Register<DT> vector_arithmetic_unit_fp_div_data0{
      "vector_arithmetic_unit_fp_div_data0", *this, 0x1b0, 0x0};
  sfr::Bitfield<DT> fp_div_data0_scalar_register_element0{
      "fp_div_data0_scalar_register_element0",
      vector_arithmetic_unit_fp_div_data0, 32, 0};
  sfr::Bitfield<DT> fp_div_data0_scalar_register_element1{
      "fp_div_data0_scalar_register_element1",
      vector_arithmetic_unit_fp_div_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_fp_div_data1{
      "vector_arithmetic_unit_fp_div_data1", *this, 0x1b8, 0x0};
  sfr::Bitfield<DT> fp_div_data1_scalar_register_element2{
      "fp_div_data1_scalar_register_element2",
      vector_arithmetic_unit_fp_div_data1, 32, 0};
  sfr::Register<DT> vector_arithmetic_unit_float_adapter{
      "vector_arithmetic_unit_float_adapter", *this, 0x1c0, 0x0};
  sfr::Bitfield<DT> float_adapter_fxp_to_fp_mode{
      "float_adapter_fxp_to_fp_mode", vector_arithmetic_unit_float_adapter, 1,
      0};
  sfr::Bitfield<DT> float_adapter_fxp_to_fp_int_width{
      "float_adapter_fxp_to_fp_int_width", vector_arithmetic_unit_float_adapter,
      5, 1};
  sfr::Bitfield<DT> float_adapter_fxp_to_fp_round_mode{
      "float_adapter_fxp_to_fp_round_mode",
      vector_arithmetic_unit_float_adapter, 1, 6};
  sfr::Bitfield<DT> float_adapter_fp_to_fxp_mode{
      "float_adapter_fp_to_fxp_mode", vector_arithmetic_unit_float_adapter, 1,
      8};
  sfr::Bitfield<DT> float_adapter_fp_to_fxp_int_width{
      "float_adapter_fp_to_fxp_int_width", vector_arithmetic_unit_float_adapter,
      5, 9};
  sfr::Bitfield<DT> float_adapter_fp_to_fxp_round_mode{
      "float_adapter_fp_to_fxp_round_mode",
      vector_arithmetic_unit_float_adapter, 1, 14};
  sfr::Bitfield<DT> float_adapter_split_layer_mode{
      "float_adapter_split_layer_mode", vector_arithmetic_unit_float_adapter, 1,
      16};
  sfr::Bitfield<DT> float_adapter_concat_layer_mode{
      "float_adapter_concat_layer_mode", vector_arithmetic_unit_float_adapter,
      1, 24};
  sfr::Register<DT> vector_arithmetic_unit_clip_cluster_route{
      "vector_arithmetic_unit_clip_cluster_route", *this, 0x1c8, 0x4000};
  sfr::Bitfield<DT> clip_cluster_route_clip_add_source{
      "clip_cluster_route_clip_add_source",
      vector_arithmetic_unit_clip_cluster_route, 3, 0};
  sfr::Bitfield<DT> clip_cluster_route_clip_max_source{
      "clip_cluster_route_clip_max_source",
      vector_arithmetic_unit_clip_cluster_route, 3, 4};
  sfr::Bitfield<DT> clip_cluster_route_clip_min_source{
      "clip_cluster_route_clip_min_source",
      vector_arithmetic_unit_clip_cluster_route, 3, 8};
  sfr::Bitfield<DT> clip_cluster_route_clip_cluster_source{
      "clip_cluster_route_clip_cluster_source",
      vector_arithmetic_unit_clip_cluster_route, 3, 12};
  sfr::Register<DT> vector_arithmetic_unit_clip_add_control{
      "vector_arithmetic_unit_clip_add_control", *this, 0x1d0, 0x0};
  sfr::Bitfield<DT> clip_add_control_op_mode{
      "clip_add_control_op_mode", vector_arithmetic_unit_clip_add_control, 4,
      0};
  sfr::Bitfield<DT> clip_add_control_arg_mode{
      "clip_add_control_arg_mode", vector_arithmetic_unit_clip_add_control, 3,
      4};
  sfr::Bitfield<DT> clip_add_control_reg0_cmp_op{
      "clip_add_control_reg0_cmp_op", vector_arithmetic_unit_clip_add_control,
      2, 8};
  sfr::Bitfield<DT> clip_add_control_reg1_cmp_op{
      "clip_add_control_reg1_cmp_op", vector_arithmetic_unit_clip_add_control,
      2, 10};
  sfr::Bitfield<DT> clip_add_control_reg2_cmp_op{
      "clip_add_control_reg2_cmp_op", vector_arithmetic_unit_clip_add_control,
      2, 12};
  sfr::Bitfield<DT> clip_add_control_rf_cmp_op{
      "clip_add_control_rf_cmp_op", vector_arithmetic_unit_clip_add_control, 2,
      16};
  sfr::Bitfield<DT> clip_add_control_reg0_execution_id{
      "clip_add_control_reg0_execution_id",
      vector_arithmetic_unit_clip_add_control, 4, 24};
  sfr::Bitfield<DT> clip_add_control_reg0_execution_id_mask{
      "clip_add_control_reg0_execution_id_mask",
      vector_arithmetic_unit_clip_add_control, 4, 28};
  sfr::Bitfield<DT> clip_add_control_reg1_execution_id{
      "clip_add_control_reg1_execution_id",
      vector_arithmetic_unit_clip_add_control, 4, 32};
  sfr::Bitfield<DT> clip_add_control_reg1_execution_id_mask{
      "clip_add_control_reg1_execution_id_mask",
      vector_arithmetic_unit_clip_add_control, 4, 36};
  sfr::Bitfield<DT> clip_add_control_reg2_execution_id{
      "clip_add_control_reg2_execution_id",
      vector_arithmetic_unit_clip_add_control, 4, 40};
  sfr::Bitfield<DT> clip_add_control_reg2_execution_id_mask{
      "clip_add_control_reg2_execution_id_mask",
      vector_arithmetic_unit_clip_add_control, 4, 44};
  sfr::Bitfield<DT> clip_add_control_rf_execution_id{
      "clip_add_control_rf_execution_id",
      vector_arithmetic_unit_clip_add_control, 4, 48};
  sfr::Bitfield<DT> clip_add_control_rf_execution_id_mask{
      "clip_add_control_rf_execution_id_mask",
      vector_arithmetic_unit_clip_add_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_clip_add_data0{
      "vector_arithmetic_unit_clip_add_data0", *this, 0x1d8, 0x0};
  sfr::Bitfield<DT> clip_add_data0_scalar_register_element0{
      "clip_add_data0_scalar_register_element0",
      vector_arithmetic_unit_clip_add_data0, 32, 0};
  sfr::Bitfield<DT> clip_add_data0_scalar_register_element1{
      "clip_add_data0_scalar_register_element1",
      vector_arithmetic_unit_clip_add_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_clip_add_data1{
      "vector_arithmetic_unit_clip_add_data1", *this, 0x1e0, 0x0};
  sfr::Bitfield<DT> clip_add_data1_scalar_register_element2{
      "clip_add_data1_scalar_register_element2",
      vector_arithmetic_unit_clip_add_data1, 32, 0};
  sfr::Register<DT> vector_arithmetic_unit_clip_max_control{
      "vector_arithmetic_unit_clip_max_control", *this, 0x1e8, 0x0};
  sfr::Bitfield<DT> clip_max_control_op_mode{
      "clip_max_control_op_mode", vector_arithmetic_unit_clip_max_control, 4,
      0};
  sfr::Bitfield<DT> clip_max_control_arg_mode{
      "clip_max_control_arg_mode", vector_arithmetic_unit_clip_max_control, 3,
      4};
  sfr::Bitfield<DT> clip_max_control_reg0_cmp_op{
      "clip_max_control_reg0_cmp_op", vector_arithmetic_unit_clip_max_control,
      2, 8};
  sfr::Bitfield<DT> clip_max_control_reg1_cmp_op{
      "clip_max_control_reg1_cmp_op", vector_arithmetic_unit_clip_max_control,
      2, 10};
  sfr::Bitfield<DT> clip_max_control_reg2_cmp_op{
      "clip_max_control_reg2_cmp_op", vector_arithmetic_unit_clip_max_control,
      2, 12};
  sfr::Bitfield<DT> clip_max_control_rf_cmp_op{
      "clip_max_control_rf_cmp_op", vector_arithmetic_unit_clip_max_control, 2,
      16};
  sfr::Bitfield<DT> clip_max_control_reg0_execution_id{
      "clip_max_control_reg0_execution_id",
      vector_arithmetic_unit_clip_max_control, 4, 24};
  sfr::Bitfield<DT> clip_max_control_reg0_execution_id_mask{
      "clip_max_control_reg0_execution_id_mask",
      vector_arithmetic_unit_clip_max_control, 4, 28};
  sfr::Bitfield<DT> clip_max_control_reg1_execution_id{
      "clip_max_control_reg1_execution_id",
      vector_arithmetic_unit_clip_max_control, 4, 32};
  sfr::Bitfield<DT> clip_max_control_reg1_execution_id_mask{
      "clip_max_control_reg1_execution_id_mask",
      vector_arithmetic_unit_clip_max_control, 4, 36};
  sfr::Bitfield<DT> clip_max_control_reg2_execution_id{
      "clip_max_control_reg2_execution_id",
      vector_arithmetic_unit_clip_max_control, 4, 40};
  sfr::Bitfield<DT> clip_max_control_reg2_execution_id_mask{
      "clip_max_control_reg2_execution_id_mask",
      vector_arithmetic_unit_clip_max_control, 4, 44};
  sfr::Bitfield<DT> clip_max_control_rf_execution_id{
      "clip_max_control_rf_execution_id",
      vector_arithmetic_unit_clip_max_control, 4, 48};
  sfr::Bitfield<DT> clip_max_control_rf_execution_id_mask{
      "clip_max_control_rf_execution_id_mask",
      vector_arithmetic_unit_clip_max_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_clip_max_data0{
      "vector_arithmetic_unit_clip_max_data0", *this, 0x1f0, 0x0};
  sfr::Bitfield<DT> clip_max_data0_scalar_register_element0{
      "clip_max_data0_scalar_register_element0",
      vector_arithmetic_unit_clip_max_data0, 32, 0};
  sfr::Bitfield<DT> clip_max_data0_scalar_register_element1{
      "clip_max_data0_scalar_register_element1",
      vector_arithmetic_unit_clip_max_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_clip_max_data1{
      "vector_arithmetic_unit_clip_max_data1", *this, 0x1f8, 0x0};
  sfr::Bitfield<DT> clip_max_data1_scalar_register_element2{
      "clip_max_data1_scalar_register_element2",
      vector_arithmetic_unit_clip_max_data1, 32, 0};
  sfr::Register<DT> vector_arithmetic_unit_clip_min_control{
      "vector_arithmetic_unit_clip_min_control", *this, 0x200, 0x0};
  sfr::Bitfield<DT> clip_min_control_op_mode{
      "clip_min_control_op_mode", vector_arithmetic_unit_clip_min_control, 4,
      0};
  sfr::Bitfield<DT> clip_min_control_arg_mode{
      "clip_min_control_arg_mode", vector_arithmetic_unit_clip_min_control, 3,
      4};
  sfr::Bitfield<DT> clip_min_control_reg0_cmp_op{
      "clip_min_control_reg0_cmp_op", vector_arithmetic_unit_clip_min_control,
      2, 8};
  sfr::Bitfield<DT> clip_min_control_reg1_cmp_op{
      "clip_min_control_reg1_cmp_op", vector_arithmetic_unit_clip_min_control,
      2, 10};
  sfr::Bitfield<DT> clip_min_control_reg2_cmp_op{
      "clip_min_control_reg2_cmp_op", vector_arithmetic_unit_clip_min_control,
      2, 12};
  sfr::Bitfield<DT> clip_min_control_rf_cmp_op{
      "clip_min_control_rf_cmp_op", vector_arithmetic_unit_clip_min_control, 2,
      16};
  sfr::Bitfield<DT> clip_min_control_reg0_execution_id{
      "clip_min_control_reg0_execution_id",
      vector_arithmetic_unit_clip_min_control, 4, 24};
  sfr::Bitfield<DT> clip_min_control_reg0_execution_id_mask{
      "clip_min_control_reg0_execution_id_mask",
      vector_arithmetic_unit_clip_min_control, 4, 28};
  sfr::Bitfield<DT> clip_min_control_reg1_execution_id{
      "clip_min_control_reg1_execution_id",
      vector_arithmetic_unit_clip_min_control, 4, 32};
  sfr::Bitfield<DT> clip_min_control_reg1_execution_id_mask{
      "clip_min_control_reg1_execution_id_mask",
      vector_arithmetic_unit_clip_min_control, 4, 36};
  sfr::Bitfield<DT> clip_min_control_reg2_execution_id{
      "clip_min_control_reg2_execution_id",
      vector_arithmetic_unit_clip_min_control, 4, 40};
  sfr::Bitfield<DT> clip_min_control_reg2_execution_id_mask{
      "clip_min_control_reg2_execution_id_mask",
      vector_arithmetic_unit_clip_min_control, 4, 44};
  sfr::Bitfield<DT> clip_min_control_rf_execution_id{
      "clip_min_control_rf_execution_id",
      vector_arithmetic_unit_clip_min_control, 4, 48};
  sfr::Bitfield<DT> clip_min_control_rf_execution_id_mask{
      "clip_min_control_rf_execution_id_mask",
      vector_arithmetic_unit_clip_min_control, 4, 52};
  sfr::Register<DT> vector_arithmetic_unit_clip_min_data0{
      "vector_arithmetic_unit_clip_min_data0", *this, 0x208, 0x0};
  sfr::Bitfield<DT> clip_min_data0_scalar_register_element0{
      "clip_min_data0_scalar_register_element0",
      vector_arithmetic_unit_clip_min_data0, 32, 0};
  sfr::Bitfield<DT> clip_min_data0_scalar_register_element1{
      "clip_min_data0_scalar_register_element1",
      vector_arithmetic_unit_clip_min_data0, 32, 32};
  sfr::Register<DT> vector_arithmetic_unit_clip_min_data1{
      "vector_arithmetic_unit_clip_min_data1", *this, 0x210, 0x0};
  sfr::Bitfield<DT> clip_min_data1_scalar_register_element2{
      "clip_min_data1_scalar_register_element2",
      vector_arithmetic_unit_clip_min_data1, 32, 0};
  sfr::Register<DT> vector_arithmetic_unit_alloc_indexer{
      "vector_arithmetic_unit_alloc_indexer", *this, 0x218, 0x0};
  sfr::Bitfield<DT> alloc_indexer_read_indexer0_module{
      "alloc_indexer_read_indexer0_module",
      vector_arithmetic_unit_alloc_indexer, 5, 0};
  sfr::Bitfield<DT> alloc_indexer_read_indexer1_module{
      "alloc_indexer_read_indexer1_module",
      vector_arithmetic_unit_alloc_indexer, 5, 8};
  sfr::Bitfield<DT> alloc_indexer_read_indexer2_module{
      "alloc_indexer_read_indexer2_module",
      vector_arithmetic_unit_alloc_indexer, 5, 16};
  sfr::Bitfield<DT> alloc_indexer_read_indexer3_module{
      "alloc_indexer_read_indexer3_module",
      vector_arithmetic_unit_alloc_indexer, 5, 24};
  sfr::Bitfield<DT> alloc_indexer_operand_indexer_module{
      "alloc_indexer_operand_indexer_module",
      vector_arithmetic_unit_alloc_indexer, 5, 32};
  sfr::Bitfield<DT> alloc_indexer_write_indexer_module{
      "alloc_indexer_write_indexer_module",
      vector_arithmetic_unit_alloc_indexer, 5, 40};
  sfr::Register<DT> vector_arithmetic_unit_operation{
      "vector_arithmetic_unit_operation", *this, 0x220, 0x0};
  sfr::Bitfield<DT> read_indexer0_operation{
      "read_indexer0_operation", vector_arithmetic_unit_operation, 2, 0};
  sfr::Bitfield<DT> operation_read_indexer0_proceed{
      "operation_read_indexer0_proceed", vector_arithmetic_unit_operation, 1,
      2};
  sfr::Bitfield<DT> operation_read_indexer0_element_size{
      "operation_read_indexer0_element_size", vector_arithmetic_unit_operation,
      2, 4};
  sfr::Bitfield<DT> read_indexer1_operation{
      "read_indexer1_operation", vector_arithmetic_unit_operation, 2, 8};
  sfr::Bitfield<DT> operation_read_indexer1_proceed{
      "operation_read_indexer1_proceed", vector_arithmetic_unit_operation, 1,
      10};
  sfr::Bitfield<DT> operation_read_indexer1_element_size{
      "operation_read_indexer1_element_size", vector_arithmetic_unit_operation,
      2, 12};
  sfr::Bitfield<DT> read_indexer2_operation{
      "read_indexer2_operation", vector_arithmetic_unit_operation, 2, 16};
  sfr::Bitfield<DT> operation_read_indexer2_proceed{
      "operation_read_indexer2_proceed", vector_arithmetic_unit_operation, 1,
      18};
  sfr::Bitfield<DT> operation_read_indexer2_element_size{
      "operation_read_indexer2_element_size", vector_arithmetic_unit_operation,
      2, 20};
  sfr::Bitfield<DT> read_indexer3_operation{
      "read_indexer3_operation", vector_arithmetic_unit_operation, 2, 24};
  sfr::Bitfield<DT> operation_read_indexer3_proceed{
      "operation_read_indexer3_proceed", vector_arithmetic_unit_operation, 1,
      26};
  sfr::Bitfield<DT> operation_read_indexer3_element_size{
      "operation_read_indexer3_element_size", vector_arithmetic_unit_operation,
      2, 28};
  sfr::Bitfield<DT> operand_indexer_operation{
      "operand_indexer_operation", vector_arithmetic_unit_operation, 2, 32};
  sfr::Bitfield<DT> operation_operand_indexer_proceed{
      "operation_operand_indexer_proceed", vector_arithmetic_unit_operation, 1,
      34};
  sfr::Bitfield<DT> operation_operand_indexer_update_mode{
      "operation_operand_indexer_update_mode", vector_arithmetic_unit_operation,
      1, 35};
  sfr::Bitfield<DT> operation_operand_indexer_element_size{
      "operation_operand_indexer_element_size",
      vector_arithmetic_unit_operation, 2, 36};
  sfr::Bitfield<DT> write_indexer_operation{
      "write_indexer_operation", vector_arithmetic_unit_operation, 2, 40};
  sfr::Register<DT> vector_arithmetic_unit_indexer_base0{
      "vector_arithmetic_unit_indexer_base0", *this, 0x228, 0x0};
  sfr::Bitfield<DT> indexer_base0_read_indexer0_base{
      "indexer_base0_read_indexer0_base", vector_arithmetic_unit_indexer_base0,
      13, 0};
  sfr::Bitfield<DT> indexer_base0_read_indexer1_base{
      "indexer_base0_read_indexer1_base", vector_arithmetic_unit_indexer_base0,
      13, 16};
  sfr::Bitfield<DT> indexer_base0_read_indexer2_base{
      "indexer_base0_read_indexer2_base", vector_arithmetic_unit_indexer_base0,
      13, 32};
  sfr::Bitfield<DT> indexer_base0_read_indexer3_base{
      "indexer_base0_read_indexer3_base", vector_arithmetic_unit_indexer_base0,
      13, 48};
  sfr::Register<DT> vector_arithmetic_unit_indexer_base1{
      "vector_arithmetic_unit_indexer_base1", *this, 0x230, 0x0};
  sfr::Bitfield<DT> indexer_base1_operand_indexer_base{
      "indexer_base1_operand_indexer_base",
      vector_arithmetic_unit_indexer_base1, 13, 0};
  sfr::Bitfield<DT> indexer_base1_write_indexer_base{
      "indexer_base1_write_indexer_base", vector_arithmetic_unit_indexer_base1,
      13, 16};
  sfr::Register<DT> vector_arithmetic_unit_read_indexer0_limit_info0{
      "vector_arithmetic_unit_read_indexer0_limit_info0", *this, 0x238, 0x0};
  sfr::Bitfield<DT> read_indexer0_limit_info0_read_indexer0_limit_element0{
      "read_indexer0_limit_info0_read_indexer0_limit_element0",
      vector_arithmetic_unit_read_indexer0_limit_info0, 16, 0};
  sfr::Bitfield<DT> read_indexer0_limit_info0_read_indexer0_limit_element1{
      "read_indexer0_limit_info0_read_indexer0_limit_element1",
      vector_arithmetic_unit_read_indexer0_limit_info0, 16, 16};
  sfr::Bitfield<DT> read_indexer0_limit_info0_read_indexer0_limit_element2{
      "read_indexer0_limit_info0_read_indexer0_limit_element2",
      vector_arithmetic_unit_read_indexer0_limit_info0, 16, 32};
  sfr::Bitfield<DT> read_indexer0_limit_info0_read_indexer0_limit_element3{
      "read_indexer0_limit_info0_read_indexer0_limit_element3",
      vector_arithmetic_unit_read_indexer0_limit_info0, 16, 48};
  sfr::Register<DT> vector_arithmetic_unit_read_indexer0_limit_info1{
      "vector_arithmetic_unit_read_indexer0_limit_info1", *this, 0x240, 0x0};
  sfr::Bitfield<DT> read_indexer0_limit_info1_read_indexer0_limit_element4{
      "read_indexer0_limit_info1_read_indexer0_limit_element4",
      vector_arithmetic_unit_read_indexer0_limit_info1, 16, 0};
  sfr::Bitfield<DT> read_indexer0_limit_info1_read_indexer0_limit_element5{
      "read_indexer0_limit_info1_read_indexer0_limit_element5",
      vector_arithmetic_unit_read_indexer0_limit_info1, 16, 16};
  sfr::Bitfield<DT> read_indexer0_limit_info1_read_indexer0_limit_element6{
      "read_indexer0_limit_info1_read_indexer0_limit_element6",
      vector_arithmetic_unit_read_indexer0_limit_info1, 16, 32};
  sfr::Bitfield<DT> read_indexer0_limit_info1_read_indexer0_limit_element7{
      "read_indexer0_limit_info1_read_indexer0_limit_element7",
      vector_arithmetic_unit_read_indexer0_limit_info1, 16, 48};
  sfr::Register<DT> vector_arithmetic_unit_read_indexer0_stride_info0{
      "vector_arithmetic_unit_read_indexer0_stride_info0", *this, 0x248, 0x0};
  sfr::Bitfield<DT> read_indexer0_stride_info0_read_indexer0_stride_element0{
      "read_indexer0_stride_info0_read_indexer0_stride_element0",
      vector_arithmetic_unit_read_indexer0_stride_info0, 13, 0};
  sfr::Bitfield<DT> read_indexer0_stride_info0_read_indexer0_stride_element1{
      "read_indexer0_stride_info0_read_indexer0_stride_element1",
      vector_arithmetic_unit_read_indexer0_stride_info0, 13, 16};
  sfr::Bitfield<DT> read_indexer0_stride_info0_read_indexer0_stride_element2{
      "read_indexer0_stride_info0_read_indexer0_stride_element2",
      vector_arithmetic_unit_read_indexer0_stride_info0, 13, 32};
  sfr::Bitfield<DT> read_indexer0_stride_info0_read_indexer0_stride_element3{
      "read_indexer0_stride_info0_read_indexer0_stride_element3",
      vector_arithmetic_unit_read_indexer0_stride_info0, 13, 48};
  sfr::Register<DT> vector_arithmetic_unit_read_indexer0_stride_info1{
      "vector_arithmetic_unit_read_indexer0_stride_info1", *this, 0x250, 0x0};
  sfr::Bitfield<DT> read_indexer0_stride_info1_read_indexer0_stride_element4{
      "read_indexer0_stride_info1_read_indexer0_stride_element4",
      vector_arithmetic_unit_read_indexer0_stride_info1, 13, 0};
  sfr::Bitfield<DT> read_indexer0_stride_info1_read_indexer0_stride_element5{
      "read_indexer0_stride_info1_read_indexer0_stride_element5",
      vector_arithmetic_unit_read_indexer0_stride_info1, 13, 16};
  sfr::Bitfield<DT> read_indexer0_stride_info1_read_indexer0_stride_element6{
      "read_indexer0_stride_info1_read_indexer0_stride_element6",
      vector_arithmetic_unit_read_indexer0_stride_info1, 13, 32};
  sfr::Bitfield<DT> read_indexer0_stride_info1_read_indexer0_stride_element7{
      "read_indexer0_stride_info1_read_indexer0_stride_element7",
      vector_arithmetic_unit_read_indexer0_stride_info1, 13, 48};
  sfr::Register<DT> vector_arithmetic_unit_read_indexer1_limit_info0{
      "vector_arithmetic_unit_read_indexer1_limit_info0", *this, 0x258, 0x0};
  sfr::Bitfield<DT> read_indexer1_limit_info0_read_indexer1_limit_element0{
      "read_indexer1_limit_info0_read_indexer1_limit_element0",
      vector_arithmetic_unit_read_indexer1_limit_info0, 16, 0};
  sfr::Bitfield<DT> read_indexer1_limit_info0_read_indexer1_limit_element1{
      "read_indexer1_limit_info0_read_indexer1_limit_element1",
      vector_arithmetic_unit_read_indexer1_limit_info0, 16, 16};
  sfr::Bitfield<DT> read_indexer1_limit_info0_read_indexer1_limit_element2{
      "read_indexer1_limit_info0_read_indexer1_limit_element2",
      vector_arithmetic_unit_read_indexer1_limit_info0, 16, 32};
  sfr::Bitfield<DT> read_indexer1_limit_info0_read_indexer1_limit_element3{
      "read_indexer1_limit_info0_read_indexer1_limit_element3",
      vector_arithmetic_unit_read_indexer1_limit_info0, 16, 48};
  sfr::Register<DT> vector_arithmetic_unit_read_indexer1_limit_info1{
      "vector_arithmetic_unit_read_indexer1_limit_info1", *this, 0x260, 0x0};
  sfr::Bitfield<DT> read_indexer1_limit_info1_read_indexer1_limit_element4{
      "read_indexer1_limit_info1_read_indexer1_limit_element4",
      vector_arithmetic_unit_read_indexer1_limit_info1, 16, 0};
  sfr::Bitfield<DT> read_indexer1_limit_info1_read_indexer1_limit_element5{
      "read_indexer1_limit_info1_read_indexer1_limit_element5",
      vector_arithmetic_unit_read_indexer1_limit_info1, 16, 16};
  sfr::Bitfield<DT> read_indexer1_limit_info1_read_indexer1_limit_element6{
      "read_indexer1_limit_info1_read_indexer1_limit_element6",
      vector_arithmetic_unit_read_indexer1_limit_info1, 16, 32};
  sfr::Bitfield<DT> read_indexer1_limit_info1_read_indexer1_limit_element7{
      "read_indexer1_limit_info1_read_indexer1_limit_element7",
      vector_arithmetic_unit_read_indexer1_limit_info1, 16, 48};
  sfr::Register<DT> vector_arithmetic_unit_read_indexer1_stride_info0{
      "vector_arithmetic_unit_read_indexer1_stride_info0", *this, 0x268, 0x0};
  sfr::Bitfield<DT> read_indexer1_stride_info0_read_indexer1_stride_element0{
      "read_indexer1_stride_info0_read_indexer1_stride_element0",
      vector_arithmetic_unit_read_indexer1_stride_info0, 13, 0};
  sfr::Bitfield<DT> read_indexer1_stride_info0_read_indexer1_stride_element1{
      "read_indexer1_stride_info0_read_indexer1_stride_element1",
      vector_arithmetic_unit_read_indexer1_stride_info0, 13, 16};
  sfr::Bitfield<DT> read_indexer1_stride_info0_read_indexer1_stride_element2{
      "read_indexer1_stride_info0_read_indexer1_stride_element2",
      vector_arithmetic_unit_read_indexer1_stride_info0, 13, 32};
  sfr::Bitfield<DT> read_indexer1_stride_info0_read_indexer1_stride_element3{
      "read_indexer1_stride_info0_read_indexer1_stride_element3",
      vector_arithmetic_unit_read_indexer1_stride_info0, 13, 48};
  sfr::Register<DT> vector_arithmetic_unit_read_indexer1_stride_info1{
      "vector_arithmetic_unit_read_indexer1_stride_info1", *this, 0x270, 0x0};
  sfr::Bitfield<DT> read_indexer1_stride_info1_read_indexer1_stride_element4{
      "read_indexer1_stride_info1_read_indexer1_stride_element4",
      vector_arithmetic_unit_read_indexer1_stride_info1, 13, 0};
  sfr::Bitfield<DT> read_indexer1_stride_info1_read_indexer1_stride_element5{
      "read_indexer1_stride_info1_read_indexer1_stride_element5",
      vector_arithmetic_unit_read_indexer1_stride_info1, 13, 16};
  sfr::Bitfield<DT> read_indexer1_stride_info1_read_indexer1_stride_element6{
      "read_indexer1_stride_info1_read_indexer1_stride_element6",
      vector_arithmetic_unit_read_indexer1_stride_info1, 13, 32};
  sfr::Bitfield<DT> read_indexer1_stride_info1_read_indexer1_stride_element7{
      "read_indexer1_stride_info1_read_indexer1_stride_element7",
      vector_arithmetic_unit_read_indexer1_stride_info1, 13, 48};
  sfr::Register<DT> vector_arithmetic_unit_read_indexer2_limit_info0{
      "vector_arithmetic_unit_read_indexer2_limit_info0", *this, 0x278, 0x0};
  sfr::Bitfield<DT> read_indexer2_limit_info0_read_indexer2_limit_element0{
      "read_indexer2_limit_info0_read_indexer2_limit_element0",
      vector_arithmetic_unit_read_indexer2_limit_info0, 16, 0};
  sfr::Bitfield<DT> read_indexer2_limit_info0_read_indexer2_limit_element1{
      "read_indexer2_limit_info0_read_indexer2_limit_element1",
      vector_arithmetic_unit_read_indexer2_limit_info0, 16, 16};
  sfr::Bitfield<DT> read_indexer2_limit_info0_read_indexer2_limit_element2{
      "read_indexer2_limit_info0_read_indexer2_limit_element2",
      vector_arithmetic_unit_read_indexer2_limit_info0, 16, 32};
  sfr::Bitfield<DT> read_indexer2_limit_info0_read_indexer2_limit_element3{
      "read_indexer2_limit_info0_read_indexer2_limit_element3",
      vector_arithmetic_unit_read_indexer2_limit_info0, 16, 48};
  sfr::Register<DT> vector_arithmetic_unit_read_indexer2_limit_info1{
      "vector_arithmetic_unit_read_indexer2_limit_info1", *this, 0x280, 0x0};
  sfr::Bitfield<DT> read_indexer2_limit_info1_read_indexer2_limit_element4{
      "read_indexer2_limit_info1_read_indexer2_limit_element4",
      vector_arithmetic_unit_read_indexer2_limit_info1, 16, 0};
  sfr::Bitfield<DT> read_indexer2_limit_info1_read_indexer2_limit_element5{
      "read_indexer2_limit_info1_read_indexer2_limit_element5",
      vector_arithmetic_unit_read_indexer2_limit_info1, 16, 16};
  sfr::Bitfield<DT> read_indexer2_limit_info1_read_indexer2_limit_element6{
      "read_indexer2_limit_info1_read_indexer2_limit_element6",
      vector_arithmetic_unit_read_indexer2_limit_info1, 16, 32};
  sfr::Bitfield<DT> read_indexer2_limit_info1_read_indexer2_limit_element7{
      "read_indexer2_limit_info1_read_indexer2_limit_element7",
      vector_arithmetic_unit_read_indexer2_limit_info1, 16, 48};
  sfr::Register<DT> vector_arithmetic_unit_read_indexer2_stride_info0{
      "vector_arithmetic_unit_read_indexer2_stride_info0", *this, 0x288, 0x0};
  sfr::Bitfield<DT> read_indexer2_stride_info0_read_indexer2_stride_element0{
      "read_indexer2_stride_info0_read_indexer2_stride_element0",
      vector_arithmetic_unit_read_indexer2_stride_info0, 13, 0};
  sfr::Bitfield<DT> read_indexer2_stride_info0_read_indexer2_stride_element1{
      "read_indexer2_stride_info0_read_indexer2_stride_element1",
      vector_arithmetic_unit_read_indexer2_stride_info0, 13, 16};
  sfr::Bitfield<DT> read_indexer2_stride_info0_read_indexer2_stride_element2{
      "read_indexer2_stride_info0_read_indexer2_stride_element2",
      vector_arithmetic_unit_read_indexer2_stride_info0, 13, 32};
  sfr::Bitfield<DT> read_indexer2_stride_info0_read_indexer2_stride_element3{
      "read_indexer2_stride_info0_read_indexer2_stride_element3",
      vector_arithmetic_unit_read_indexer2_stride_info0, 13, 48};
  sfr::Register<DT> vector_arithmetic_unit_read_indexer2_stride_info1{
      "vector_arithmetic_unit_read_indexer2_stride_info1", *this, 0x290, 0x0};
  sfr::Bitfield<DT> read_indexer2_stride_info1_read_indexer2_stride_element4{
      "read_indexer2_stride_info1_read_indexer2_stride_element4",
      vector_arithmetic_unit_read_indexer2_stride_info1, 13, 0};
  sfr::Bitfield<DT> read_indexer2_stride_info1_read_indexer2_stride_element5{
      "read_indexer2_stride_info1_read_indexer2_stride_element5",
      vector_arithmetic_unit_read_indexer2_stride_info1, 13, 16};
  sfr::Bitfield<DT> read_indexer2_stride_info1_read_indexer2_stride_element6{
      "read_indexer2_stride_info1_read_indexer2_stride_element6",
      vector_arithmetic_unit_read_indexer2_stride_info1, 13, 32};
  sfr::Bitfield<DT> read_indexer2_stride_info1_read_indexer2_stride_element7{
      "read_indexer2_stride_info1_read_indexer2_stride_element7",
      vector_arithmetic_unit_read_indexer2_stride_info1, 13, 48};
  sfr::Register<DT> vector_arithmetic_unit_read_indexer3_limit_info0{
      "vector_arithmetic_unit_read_indexer3_limit_info0", *this, 0x298, 0x0};
  sfr::Bitfield<DT> read_indexer3_limit_info0_read_indexer3_limit_element0{
      "read_indexer3_limit_info0_read_indexer3_limit_element0",
      vector_arithmetic_unit_read_indexer3_limit_info0, 16, 0};
  sfr::Bitfield<DT> read_indexer3_limit_info0_read_indexer3_limit_element1{
      "read_indexer3_limit_info0_read_indexer3_limit_element1",
      vector_arithmetic_unit_read_indexer3_limit_info0, 16, 16};
  sfr::Bitfield<DT> read_indexer3_limit_info0_read_indexer3_limit_element2{
      "read_indexer3_limit_info0_read_indexer3_limit_element2",
      vector_arithmetic_unit_read_indexer3_limit_info0, 16, 32};
  sfr::Bitfield<DT> read_indexer3_limit_info0_read_indexer3_limit_element3{
      "read_indexer3_limit_info0_read_indexer3_limit_element3",
      vector_arithmetic_unit_read_indexer3_limit_info0, 16, 48};
  sfr::Register<DT> vector_arithmetic_unit_read_indexer3_limit_info1{
      "vector_arithmetic_unit_read_indexer3_limit_info1", *this, 0x2a0, 0x0};
  sfr::Bitfield<DT> read_indexer3_limit_info1_read_indexer3_limit_element4{
      "read_indexer3_limit_info1_read_indexer3_limit_element4",
      vector_arithmetic_unit_read_indexer3_limit_info1, 16, 0};
  sfr::Bitfield<DT> read_indexer3_limit_info1_read_indexer3_limit_element5{
      "read_indexer3_limit_info1_read_indexer3_limit_element5",
      vector_arithmetic_unit_read_indexer3_limit_info1, 16, 16};
  sfr::Bitfield<DT> read_indexer3_limit_info1_read_indexer3_limit_element6{
      "read_indexer3_limit_info1_read_indexer3_limit_element6",
      vector_arithmetic_unit_read_indexer3_limit_info1, 16, 32};
  sfr::Bitfield<DT> read_indexer3_limit_info1_read_indexer3_limit_element7{
      "read_indexer3_limit_info1_read_indexer3_limit_element7",
      vector_arithmetic_unit_read_indexer3_limit_info1, 16, 48};
  sfr::Register<DT> vector_arithmetic_unit_read_indexer3_stride_info0{
      "vector_arithmetic_unit_read_indexer3_stride_info0", *this, 0x2a8, 0x0};
  sfr::Bitfield<DT> read_indexer3_stride_info0_read_indexer3_stride_element0{
      "read_indexer3_stride_info0_read_indexer3_stride_element0",
      vector_arithmetic_unit_read_indexer3_stride_info0, 13, 0};
  sfr::Bitfield<DT> read_indexer3_stride_info0_read_indexer3_stride_element1{
      "read_indexer3_stride_info0_read_indexer3_stride_element1",
      vector_arithmetic_unit_read_indexer3_stride_info0, 13, 16};
  sfr::Bitfield<DT> read_indexer3_stride_info0_read_indexer3_stride_element2{
      "read_indexer3_stride_info0_read_indexer3_stride_element2",
      vector_arithmetic_unit_read_indexer3_stride_info0, 13, 32};
  sfr::Bitfield<DT> read_indexer3_stride_info0_read_indexer3_stride_element3{
      "read_indexer3_stride_info0_read_indexer3_stride_element3",
      vector_arithmetic_unit_read_indexer3_stride_info0, 13, 48};
  sfr::Register<DT> vector_arithmetic_unit_read_indexer3_stride_info1{
      "vector_arithmetic_unit_read_indexer3_stride_info1", *this, 0x2b0, 0x0};
  sfr::Bitfield<DT> read_indexer3_stride_info1_read_indexer3_stride_element4{
      "read_indexer3_stride_info1_read_indexer3_stride_element4",
      vector_arithmetic_unit_read_indexer3_stride_info1, 13, 0};
  sfr::Bitfield<DT> read_indexer3_stride_info1_read_indexer3_stride_element5{
      "read_indexer3_stride_info1_read_indexer3_stride_element5",
      vector_arithmetic_unit_read_indexer3_stride_info1, 13, 16};
  sfr::Bitfield<DT> read_indexer3_stride_info1_read_indexer3_stride_element6{
      "read_indexer3_stride_info1_read_indexer3_stride_element6",
      vector_arithmetic_unit_read_indexer3_stride_info1, 13, 32};
  sfr::Bitfield<DT> read_indexer3_stride_info1_read_indexer3_stride_element7{
      "read_indexer3_stride_info1_read_indexer3_stride_element7",
      vector_arithmetic_unit_read_indexer3_stride_info1, 13, 48};
  sfr::Register<DT> vector_arithmetic_unit_operand_indexer_limit_info0{
      "vector_arithmetic_unit_operand_indexer_limit_info0", *this, 0x2b8, 0x0};
  sfr::Bitfield<DT> operand_indexer_limit_info0_operand_indexer_limit_element0{
      "operand_indexer_limit_info0_operand_indexer_limit_element0",
      vector_arithmetic_unit_operand_indexer_limit_info0, 16, 0};
  sfr::Bitfield<DT> operand_indexer_limit_info0_operand_indexer_limit_element1{
      "operand_indexer_limit_info0_operand_indexer_limit_element1",
      vector_arithmetic_unit_operand_indexer_limit_info0, 16, 16};
  sfr::Bitfield<DT> operand_indexer_limit_info0_operand_indexer_limit_element2{
      "operand_indexer_limit_info0_operand_indexer_limit_element2",
      vector_arithmetic_unit_operand_indexer_limit_info0, 16, 32};
  sfr::Bitfield<DT> operand_indexer_limit_info0_operand_indexer_limit_element3{
      "operand_indexer_limit_info0_operand_indexer_limit_element3",
      vector_arithmetic_unit_operand_indexer_limit_info0, 16, 48};
  sfr::Register<DT> vector_arithmetic_unit_operand_indexer_limit_info1{
      "vector_arithmetic_unit_operand_indexer_limit_info1", *this, 0x2c0, 0x0};
  sfr::Bitfield<DT> operand_indexer_limit_info1_operand_indexer_limit_element4{
      "operand_indexer_limit_info1_operand_indexer_limit_element4",
      vector_arithmetic_unit_operand_indexer_limit_info1, 16, 0};
  sfr::Bitfield<DT> operand_indexer_limit_info1_operand_indexer_limit_element5{
      "operand_indexer_limit_info1_operand_indexer_limit_element5",
      vector_arithmetic_unit_operand_indexer_limit_info1, 16, 16};
  sfr::Bitfield<DT> operand_indexer_limit_info1_operand_indexer_limit_element6{
      "operand_indexer_limit_info1_operand_indexer_limit_element6",
      vector_arithmetic_unit_operand_indexer_limit_info1, 16, 32};
  sfr::Bitfield<DT> operand_indexer_limit_info1_operand_indexer_limit_element7{
      "operand_indexer_limit_info1_operand_indexer_limit_element7",
      vector_arithmetic_unit_operand_indexer_limit_info1, 16, 48};
  sfr::Register<DT> vector_arithmetic_unit_operand_indexer_stride_info0{
      "vector_arithmetic_unit_operand_indexer_stride_info0", *this, 0x2c8, 0x0};
  sfr::Bitfield<DT>
      operand_indexer_stride_info0_operand_indexer_stride_element0{
          "operand_indexer_stride_info0_operand_indexer_stride_element0",
          vector_arithmetic_unit_operand_indexer_stride_info0, 13, 0};
  sfr::Bitfield<DT>
      operand_indexer_stride_info0_operand_indexer_stride_element1{
          "operand_indexer_stride_info0_operand_indexer_stride_element1",
          vector_arithmetic_unit_operand_indexer_stride_info0, 13, 16};
  sfr::Bitfield<DT>
      operand_indexer_stride_info0_operand_indexer_stride_element2{
          "operand_indexer_stride_info0_operand_indexer_stride_element2",
          vector_arithmetic_unit_operand_indexer_stride_info0, 13, 32};
  sfr::Bitfield<DT>
      operand_indexer_stride_info0_operand_indexer_stride_element3{
          "operand_indexer_stride_info0_operand_indexer_stride_element3",
          vector_arithmetic_unit_operand_indexer_stride_info0, 13, 48};
  sfr::Register<DT> vector_arithmetic_unit_operand_indexer_stride_info1{
      "vector_arithmetic_unit_operand_indexer_stride_info1", *this, 0x2d0, 0x0};
  sfr::Bitfield<DT>
      operand_indexer_stride_info1_operand_indexer_stride_element4{
          "operand_indexer_stride_info1_operand_indexer_stride_element4",
          vector_arithmetic_unit_operand_indexer_stride_info1, 13, 0};
  sfr::Bitfield<DT>
      operand_indexer_stride_info1_operand_indexer_stride_element5{
          "operand_indexer_stride_info1_operand_indexer_stride_element5",
          vector_arithmetic_unit_operand_indexer_stride_info1, 13, 16};
  sfr::Bitfield<DT>
      operand_indexer_stride_info1_operand_indexer_stride_element6{
          "operand_indexer_stride_info1_operand_indexer_stride_element6",
          vector_arithmetic_unit_operand_indexer_stride_info1, 13, 32};
  sfr::Bitfield<DT>
      operand_indexer_stride_info1_operand_indexer_stride_element7{
          "operand_indexer_stride_info1_operand_indexer_stride_element7",
          vector_arithmetic_unit_operand_indexer_stride_info1, 13, 48};
  sfr::Register<DT> vector_arithmetic_unit_write_indexer_limit_info0{
      "vector_arithmetic_unit_write_indexer_limit_info0", *this, 0x2d8, 0x0};
  sfr::Bitfield<DT> write_indexer_limit_info0_write_indexer_limit_element0{
      "write_indexer_limit_info0_write_indexer_limit_element0",
      vector_arithmetic_unit_write_indexer_limit_info0, 16, 0};
  sfr::Bitfield<DT> write_indexer_limit_info0_write_indexer_limit_element1{
      "write_indexer_limit_info0_write_indexer_limit_element1",
      vector_arithmetic_unit_write_indexer_limit_info0, 16, 16};
  sfr::Bitfield<DT> write_indexer_limit_info0_write_indexer_limit_element2{
      "write_indexer_limit_info0_write_indexer_limit_element2",
      vector_arithmetic_unit_write_indexer_limit_info0, 16, 32};
  sfr::Bitfield<DT> write_indexer_limit_info0_write_indexer_limit_element3{
      "write_indexer_limit_info0_write_indexer_limit_element3",
      vector_arithmetic_unit_write_indexer_limit_info0, 16, 48};
  sfr::Register<DT> vector_arithmetic_unit_write_indexer_limit_info1{
      "vector_arithmetic_unit_write_indexer_limit_info1", *this, 0x2e0, 0x0};
  sfr::Bitfield<DT> write_indexer_limit_info1_write_indexer_limit_element4{
      "write_indexer_limit_info1_write_indexer_limit_element4",
      vector_arithmetic_unit_write_indexer_limit_info1, 16, 0};
  sfr::Bitfield<DT> write_indexer_limit_info1_write_indexer_limit_element5{
      "write_indexer_limit_info1_write_indexer_limit_element5",
      vector_arithmetic_unit_write_indexer_limit_info1, 16, 16};
  sfr::Bitfield<DT> write_indexer_limit_info1_write_indexer_limit_element6{
      "write_indexer_limit_info1_write_indexer_limit_element6",
      vector_arithmetic_unit_write_indexer_limit_info1, 16, 32};
  sfr::Bitfield<DT> write_indexer_limit_info1_write_indexer_limit_element7{
      "write_indexer_limit_info1_write_indexer_limit_element7",
      vector_arithmetic_unit_write_indexer_limit_info1, 16, 48};
  sfr::Register<DT> vector_arithmetic_unit_write_indexer_stride_info0{
      "vector_arithmetic_unit_write_indexer_stride_info0", *this, 0x2e8, 0x0};
  sfr::Bitfield<DT> write_indexer_stride_info0_write_indexer_stride_element0{
      "write_indexer_stride_info0_write_indexer_stride_element0",
      vector_arithmetic_unit_write_indexer_stride_info0, 13, 0};
  sfr::Bitfield<DT> write_indexer_stride_info0_write_indexer_stride_element1{
      "write_indexer_stride_info0_write_indexer_stride_element1",
      vector_arithmetic_unit_write_indexer_stride_info0, 13, 16};
  sfr::Bitfield<DT> write_indexer_stride_info0_write_indexer_stride_element2{
      "write_indexer_stride_info0_write_indexer_stride_element2",
      vector_arithmetic_unit_write_indexer_stride_info0, 13, 32};
  sfr::Bitfield<DT> write_indexer_stride_info0_write_indexer_stride_element3{
      "write_indexer_stride_info0_write_indexer_stride_element3",
      vector_arithmetic_unit_write_indexer_stride_info0, 13, 48};
  sfr::Register<DT> vector_arithmetic_unit_write_indexer_stride_info1{
      "vector_arithmetic_unit_write_indexer_stride_info1", *this, 0x2f0, 0x0};
  sfr::Bitfield<DT> write_indexer_stride_info1_write_indexer_stride_element4{
      "write_indexer_stride_info1_write_indexer_stride_element4",
      vector_arithmetic_unit_write_indexer_stride_info1, 13, 0};
  sfr::Bitfield<DT> write_indexer_stride_info1_write_indexer_stride_element5{
      "write_indexer_stride_info1_write_indexer_stride_element5",
      vector_arithmetic_unit_write_indexer_stride_info1, 13, 16};
  sfr::Bitfield<DT> write_indexer_stride_info1_write_indexer_stride_element6{
      "write_indexer_stride_info1_write_indexer_stride_element6",
      vector_arithmetic_unit_write_indexer_stride_info1, 13, 32};
  sfr::Bitfield<DT> write_indexer_stride_info1_write_indexer_stride_element7{
      "write_indexer_stride_info1_write_indexer_stride_element7",
      vector_arithmetic_unit_write_indexer_stride_info1, 13, 48};
};

template <class DT>
class VectorRegisterFileParityErrorCounter : public sfr::Block<DT> {
public:
  VectorRegisterFileParityErrorCounter()
      : sfr::Block<DT>::Block(0x1f08, 0x8, false) {}

public:
  sfr::Register<DT> vector_register_file_parity_error_counter{
      "vector_register_file_parity_error_counter", *this, 0x0, 0x0};
  sfr::Bitfield<DT> vrf_parity_error_counter{
      "vrf_parity_error_counter", vector_register_file_parity_error_counter, 4,
      0};
};

template <class DT> class TransposeEngineMainContext : public sfr::Block<DT> {
public:
  TransposeEngineMainContext() : sfr::Block<DT>::Block(0x10f8, 0x8, false) {}

public:
  sfr::Register<DT> transpose_engine_shape{"transpose_engine_shape", *this, 0x0,
                                           0x0};
  sfr::Bitfield<DT> fetch_in_cols{"fetch_in_cols", transpose_engine_shape, 6,
                                  0};
  sfr::Bitfield<DT> fetch_in_rows{"fetch_in_rows", transpose_engine_shape, 5,
                                  8};
  sfr::Bitfield<DT> fetch_out_rows{"fetch_out_rows", transpose_engine_shape, 6,
                                   16};
  sfr::Bitfield<DT> data_type{"data_type", transpose_engine_shape, 2, 24};
  sfr::Bitfield<DT> fetch_in_width_shift{"fetch_in_width_shift",
                                         transpose_engine_shape, 1, 32};
};

template <class DT> class OperationDataPath : public sfr::Block<DT> {
public:
  OperationDataPath() : sfr::Block<DT>::Block(0x170, 0x8, false) {}

public:
  sfr::Register<DT> operation_data_path_route{"operation_data_path_route",
                                              *this, 0x0, 0x0};
  sfr::Bitfield<DT> data_path_route_sub_context{
      "data_path_route_sub_context", operation_data_path_route, 7, 0};
};

template <class DT> class OperationDataPathMainContext : public sfr::Block<DT> {
public:
  OperationDataPathMainContext() : sfr::Block<DT>::Block(0x1080, 0x8, false) {}

public:
  sfr::Register<DT> operation_data_path_route_main_context{
      "operation_data_path_route_main_context", *this, 0x0, 0x0};
  sfr::Bitfield<DT> main_context{"main_context",
                                 operation_data_path_route_main_context, 7, 0};
  sfr::Bitfield<DT> channel_config{
      "channel_config", operation_data_path_route_main_context, 1, 32};
};

template <class DT> class RegisterConfig : public sfr::Block<DT> {
public:
  RegisterConfig() : sfr::Block<DT>::Block(0x188, 0x8, false) {}

public:
  sfr::Register<DT> register_config_range{"register_config_range", *this, 0x0,
                                          0x0};
  sfr::Bitfield<DT> base{"base", register_config_range, 16, 0};
  sfr::Bitfield<DT> size{"size", register_config_range, 8, 16};
  sfr::Bitfield<DT> access_type{"access_type", register_config_range, 1, 32};
  sfr::Bitfield<DT> words_per_input{"words_per_input", register_config_range, 6,
                                    40};
  sfr::Bitfield<DT> data_offset{"data_offset", register_config_range, 8, 48};
};

template <class DT> class CommitUnit : public sfr::Block<DT> {
public:
  CommitUnit() : sfr::Block<DT>::Block(0x400, 0x8, false) {}

public:
  sfr::Register<DT> commit_unit_commit_count{"commit_unit_commit_count", *this,
                                             0x0, 0x0};
  sfr::Bitfield<DT> commit_count{"commit_count", commit_unit_commit_count, 16,
                                 0};
};

template <class DT> class CommitUnitMainContext : public sfr::Block<DT> {
public:
  CommitUnitMainContext() : sfr::Block<DT>::Block(0x1100, 0x60, false) {}

public:
  sfr::Register<DT> commit_unit_mode{"commit_unit_mode", *this, 0x0, 0x0};
  sfr::Bitfield<DT> type_conversion{"type_conversion", commit_unit_mode, 2, 0};
  sfr::Register<DT> commit_unit_base{"commit_unit_base", *this, 0x8, 0x0};
  sfr::Bitfield<DT> base{"base", commit_unit_base, 22, 0};
  sfr::Bitfield<DT> commit_in_size{"commit_in_size", commit_unit_base, 7, 32};
  sfr::Bitfield<DT> commit_size{"commit_size", commit_unit_base, 7, 40};
  sfr::Register<DT> commit_unit_limit0{"commit_unit_limit0", *this, 0x10, 0x0};
  sfr::Bitfield<DT> limits_element0{"limits_element0", commit_unit_limit0, 16,
                                    0};
  sfr::Bitfield<DT> limits_element1{"limits_element1", commit_unit_limit0, 16,
                                    16};
  sfr::Bitfield<DT> limits_element2{"limits_element2", commit_unit_limit0, 16,
                                    32};
  sfr::Bitfield<DT> limits_element3{"limits_element3", commit_unit_limit0, 16,
                                    48};
  sfr::Register<DT> commit_unit_limit1{"commit_unit_limit1", *this, 0x18, 0x0};
  sfr::Bitfield<DT> limits_element4{"limits_element4", commit_unit_limit1, 16,
                                    0};
  sfr::Bitfield<DT> limits_element5{"limits_element5", commit_unit_limit1, 16,
                                    16};
  sfr::Bitfield<DT> limits_element6{"limits_element6", commit_unit_limit1, 16,
                                    32};
  sfr::Bitfield<DT> limits_element7{"limits_element7", commit_unit_limit1, 16,
                                    48};
  sfr::Register<DT> commit_unit_stride0{"commit_unit_stride0", *this, 0x20,
                                        0x0};
  sfr::Bitfield<DT> strides_element0{"strides_element0", commit_unit_stride0,
                                     22, 0};
  sfr::Bitfield<DT> strides_element1{"strides_element1", commit_unit_stride0,
                                     22, 32};
  sfr::Register<DT> commit_unit_stride1{"commit_unit_stride1", *this, 0x28,
                                        0x0};
  sfr::Bitfield<DT> strides_element2{"strides_element2", commit_unit_stride1,
                                     22, 0};
  sfr::Bitfield<DT> strides_element3{"strides_element3", commit_unit_stride1,
                                     22, 32};
  sfr::Register<DT> commit_unit_stride2{"commit_unit_stride2", *this, 0x30,
                                        0x0};
  sfr::Bitfield<DT> strides_element4{"strides_element4", commit_unit_stride2,
                                     22, 0};
  sfr::Bitfield<DT> strides_element5{"strides_element5", commit_unit_stride2,
                                     22, 32};
  sfr::Register<DT> commit_unit_stride3{"commit_unit_stride3", *this, 0x38,
                                        0x0};
  sfr::Bitfield<DT> strides_element6{"strides_element6", commit_unit_stride3,
                                     22, 0};
  sfr::Bitfield<DT> strides_element7{"strides_element7", commit_unit_stride3,
                                     22, 32};
  sfr::Register<DT> commit_unit_slice_enable_bitmap0{
      "commit_unit_slice_enable_bitmap0", *this, 0x40, 0x0};
  sfr::Bitfield<DT> slice_enable_bitmap_mask_element0{
      "slice_enable_bitmap_mask_element0", commit_unit_slice_enable_bitmap0, 64,
      0};
  sfr::Register<DT> commit_unit_slice_enable_bitmap1{
      "commit_unit_slice_enable_bitmap1", *this, 0x48, 0x0};
  sfr::Bitfield<DT> slice_enable_bitmap_mask_element1{
      "slice_enable_bitmap_mask_element1", commit_unit_slice_enable_bitmap1, 64,
      0};
  sfr::Register<DT> commit_unit_slice_enable_bitmap2{
      "commit_unit_slice_enable_bitmap2", *this, 0x50, 0x0};
  sfr::Bitfield<DT> slice_enable_bitmap_mask_element2{
      "slice_enable_bitmap_mask_element2", commit_unit_slice_enable_bitmap2, 64,
      0};
  sfr::Register<DT> commit_unit_slice_enable_bitmap3{
      "commit_unit_slice_enable_bitmap3", *this, 0x58, 0x0};
  sfr::Bitfield<DT> slice_enable_bitmap_mask_element3{
      "slice_enable_bitmap_mask_element3", commit_unit_slice_enable_bitmap3, 64,
      0};
};

template <class DT> class SubFetchUnit : public sfr::Block<DT> {
public:
  SubFetchUnit() : sfr::Block<DT>::Block(0x100, 0x70, false) {}

public:
  sfr::Register<DT> sub_fetch_unit_base{"sub_fetch_unit_base", *this, 0x0, 0x0};
  sfr::Bitfield<DT> base{"base", sub_fetch_unit_base, 22, 0};
  sfr::Register<DT> sub_fetch_unit_zeropoint{"sub_fetch_unit_zeropoint", *this,
                                             0x8, 0x0};
  sfr::Bitfield<DT> type_conversion{"type_conversion", sub_fetch_unit_zeropoint,
                                    4, 0};
  sfr::Bitfield<DT> num_zero_points{"num_zero_points", sub_fetch_unit_zeropoint,
                                    2, 8};
  sfr::Bitfield<DT> zero_point0{"zero_point0", sub_fetch_unit_zeropoint, 8, 16};
  sfr::Bitfield<DT> zero_point1{"zero_point1", sub_fetch_unit_zeropoint, 8, 24};
  sfr::Register<DT> sub_fetch_unit_limit0{"sub_fetch_unit_limit0", *this, 0x10,
                                          0x0};
  sfr::Bitfield<DT> limits_element0{"limits_element0", sub_fetch_unit_limit0,
                                    16, 0};
  sfr::Bitfield<DT> limits_element1{"limits_element1", sub_fetch_unit_limit0,
                                    16, 16};
  sfr::Bitfield<DT> limits_element2{"limits_element2", sub_fetch_unit_limit0,
                                    16, 32};
  sfr::Bitfield<DT> limits_element3{"limits_element3", sub_fetch_unit_limit0,
                                    16, 48};
  sfr::Register<DT> sub_fetch_unit_limit1{"sub_fetch_unit_limit1", *this, 0x18,
                                          0x0};
  sfr::Bitfield<DT> limits_element4{"limits_element4", sub_fetch_unit_limit1,
                                    16, 0};
  sfr::Bitfield<DT> limits_element5{"limits_element5", sub_fetch_unit_limit1,
                                    16, 16};
  sfr::Bitfield<DT> limits_element6{"limits_element6", sub_fetch_unit_limit1,
                                    16, 32};
  sfr::Bitfield<DT> limits_element7{"limits_element7", sub_fetch_unit_limit1,
                                    16, 48};
  sfr::Register<DT> sub_fetch_unit_stride0{"sub_fetch_unit_stride0", *this,
                                           0x20, 0x0};
  sfr::Bitfield<DT> strides_element0{"strides_element0", sub_fetch_unit_stride0,
                                     22, 0};
  sfr::Bitfield<DT> strides_element1{"strides_element1", sub_fetch_unit_stride0,
                                     22, 32};
  sfr::Register<DT> sub_fetch_unit_stride1{"sub_fetch_unit_stride1", *this,
                                           0x28, 0x0};
  sfr::Bitfield<DT> strides_element2{"strides_element2", sub_fetch_unit_stride1,
                                     22, 0};
  sfr::Bitfield<DT> strides_element3{"strides_element3", sub_fetch_unit_stride1,
                                     22, 32};
  sfr::Register<DT> sub_fetch_unit_stride2{"sub_fetch_unit_stride2", *this,
                                           0x30, 0x0};
  sfr::Bitfield<DT> strides_element4{"strides_element4", sub_fetch_unit_stride2,
                                     22, 0};
  sfr::Bitfield<DT> strides_element5{"strides_element5", sub_fetch_unit_stride2,
                                     22, 32};
  sfr::Register<DT> sub_fetch_unit_stride3{"sub_fetch_unit_stride3", *this,
                                           0x38, 0x0};
  sfr::Bitfield<DT> strides_element6{"strides_element6", sub_fetch_unit_stride3,
                                     22, 0};
  sfr::Bitfield<DT> strides_element7{"strides_element7", sub_fetch_unit_stride3,
                                     22, 32};
  sfr::Register<DT> sub_fetch_unit_fetch{"sub_fetch_unit_fetch", *this, 0x40,
                                         0x0};
  sfr::Bitfield<DT> flit_count{"flit_count", sub_fetch_unit_fetch, 24, 0};
  sfr::Bitfield<DT> words_per_packet{"words_per_packet", sub_fetch_unit_fetch,
                                     16, 32};
  sfr::Register<DT> sub_fetch_unit_topology{"sub_fetch_unit_topology", *this,
                                            0x48, 0x0};
  sfr::Bitfield<DT> topology{"topology", sub_fetch_unit_topology, 4, 0};
  sfr::Bitfield<DT> outer_slice_log_size{"outer_slice_log_size",
                                         sub_fetch_unit_topology, 4, 8};
  sfr::Bitfield<DT> outer_dim0_log_size{"outer_dim0_log_size",
                                        sub_fetch_unit_topology, 4, 16};
  sfr::Bitfield<DT> outer_dim1_log_size{"outer_dim1_log_size",
                                        sub_fetch_unit_topology, 4, 24};
  sfr::Bitfield<DT> outer_dim0_chunk_size{"outer_dim0_chunk_size",
                                          sub_fetch_unit_topology, 16, 32};
  sfr::Bitfield<DT> outer_dim1_chunk_size{"outer_dim1_chunk_size",
                                          sub_fetch_unit_topology, 16, 48};
  sfr::Register<DT> sub_fetch_unit_custom_snoop_bitmap0{
      "sub_fetch_unit_custom_snoop_bitmap0", *this, 0x50, 0x0};
  sfr::Bitfield<DT> custom_snoop_bitmap_mask_element0{
      "custom_snoop_bitmap_mask_element0", sub_fetch_unit_custom_snoop_bitmap0,
      64, 0};
  sfr::Register<DT> sub_fetch_unit_custom_snoop_bitmap1{
      "sub_fetch_unit_custom_snoop_bitmap1", *this, 0x58, 0x0};
  sfr::Bitfield<DT> custom_snoop_bitmap_mask_element1{
      "custom_snoop_bitmap_mask_element1", sub_fetch_unit_custom_snoop_bitmap1,
      64, 0};
  sfr::Register<DT> sub_fetch_unit_custom_snoop_bitmap2{
      "sub_fetch_unit_custom_snoop_bitmap2", *this, 0x60, 0x0};
  sfr::Bitfield<DT> custom_snoop_bitmap_mask_element2{
      "custom_snoop_bitmap_mask_element2", sub_fetch_unit_custom_snoop_bitmap2,
      64, 0};
  sfr::Register<DT> sub_fetch_unit_custom_snoop_bitmap3{
      "sub_fetch_unit_custom_snoop_bitmap3", *this, 0x68, 0x0};
  sfr::Bitfield<DT> custom_snoop_bitmap_mask_element3{
      "custom_snoop_bitmap_mask_element3", sub_fetch_unit_custom_snoop_bitmap3,
      64, 0};
};

template <class DT> class SubCommitUnit : public sfr::Block<DT> {
public:
  SubCommitUnit() : sfr::Block<DT>::Block(0x198, 0x68, false) {}

public:
  sfr::Register<DT> sub_commit_unit_mode{"sub_commit_unit_mode", *this, 0x0,
                                         0x0};
  sfr::Bitfield<DT> mode{"mode", sub_commit_unit_mode, 1, 0};
  sfr::Bitfield<DT> packet_valid_count{"packet_valid_count",
                                       sub_commit_unit_mode, 16, 16};
  sfr::Register<DT> sub_commit_unit_base{"sub_commit_unit_base", *this, 0x8,
                                         0x0};
  sfr::Bitfield<DT> base{"base", sub_commit_unit_base, 22, 0};
  sfr::Bitfield<DT> commit_in_size{"commit_in_size", sub_commit_unit_base, 6,
                                   32};
  sfr::Register<DT> sub_commit_unit_commit_data{"sub_commit_unit_commit_data",
                                                *this, 0x10, 0x0};
  sfr::Bitfield<DT> commit_data{"commit_data", sub_commit_unit_commit_data, 64,
                                0};
  sfr::Register<DT> sub_commit_unit_limit0{"sub_commit_unit_limit0", *this,
                                           0x18, 0x0};
  sfr::Bitfield<DT> limits_element0{"limits_element0", sub_commit_unit_limit0,
                                    16, 0};
  sfr::Bitfield<DT> limits_element1{"limits_element1", sub_commit_unit_limit0,
                                    16, 16};
  sfr::Bitfield<DT> limits_element2{"limits_element2", sub_commit_unit_limit0,
                                    16, 32};
  sfr::Bitfield<DT> limits_element3{"limits_element3", sub_commit_unit_limit0,
                                    16, 48};
  sfr::Register<DT> sub_commit_unit_limit1{"sub_commit_unit_limit1", *this,
                                           0x20, 0x0};
  sfr::Bitfield<DT> limits_element4{"limits_element4", sub_commit_unit_limit1,
                                    16, 0};
  sfr::Bitfield<DT> limits_element5{"limits_element5", sub_commit_unit_limit1,
                                    16, 16};
  sfr::Bitfield<DT> limits_element6{"limits_element6", sub_commit_unit_limit1,
                                    16, 32};
  sfr::Bitfield<DT> limits_element7{"limits_element7", sub_commit_unit_limit1,
                                    16, 48};
  sfr::Register<DT> sub_commit_unit_stride0{"sub_commit_unit_stride0", *this,
                                            0x28, 0x0};
  sfr::Bitfield<DT> strides_element0{"strides_element0",
                                     sub_commit_unit_stride0, 22, 0};
  sfr::Bitfield<DT> strides_element1{"strides_element1",
                                     sub_commit_unit_stride0, 22, 32};
  sfr::Register<DT> sub_commit_unit_stride1{"sub_commit_unit_stride1", *this,
                                            0x30, 0x0};
  sfr::Bitfield<DT> strides_element2{"strides_element2",
                                     sub_commit_unit_stride1, 22, 0};
  sfr::Bitfield<DT> strides_element3{"strides_element3",
                                     sub_commit_unit_stride1, 22, 32};
  sfr::Register<DT> sub_commit_unit_stride2{"sub_commit_unit_stride2", *this,
                                            0x38, 0x0};
  sfr::Bitfield<DT> strides_element4{"strides_element4",
                                     sub_commit_unit_stride2, 22, 0};
  sfr::Bitfield<DT> strides_element5{"strides_element5",
                                     sub_commit_unit_stride2, 22, 32};
  sfr::Register<DT> sub_commit_unit_stride3{"sub_commit_unit_stride3", *this,
                                            0x40, 0x0};
  sfr::Bitfield<DT> strides_element6{"strides_element6",
                                     sub_commit_unit_stride3, 22, 0};
  sfr::Bitfield<DT> strides_element7{"strides_element7",
                                     sub_commit_unit_stride3, 22, 32};
  sfr::Register<DT> sub_commit_unit_slice_enable_bitmap0{
      "sub_commit_unit_slice_enable_bitmap0", *this, 0x48, 0x0};
  sfr::Bitfield<DT> slice_enable_bitmap_mask_element0{
      "slice_enable_bitmap_mask_element0", sub_commit_unit_slice_enable_bitmap0,
      64, 0};
  sfr::Register<DT> sub_commit_unit_slice_enable_bitmap1{
      "sub_commit_unit_slice_enable_bitmap1", *this, 0x50, 0x0};
  sfr::Bitfield<DT> slice_enable_bitmap_mask_element1{
      "slice_enable_bitmap_mask_element1", sub_commit_unit_slice_enable_bitmap1,
      64, 0};
  sfr::Register<DT> sub_commit_unit_slice_enable_bitmap2{
      "sub_commit_unit_slice_enable_bitmap2", *this, 0x58, 0x0};
  sfr::Bitfield<DT> slice_enable_bitmap_mask_element2{
      "slice_enable_bitmap_mask_element2", sub_commit_unit_slice_enable_bitmap2,
      64, 0};
  sfr::Register<DT> sub_commit_unit_slice_enable_bitmap3{
      "sub_commit_unit_slice_enable_bitmap3", *this, 0x60, 0x0};
  sfr::Bitfield<DT> slice_enable_bitmap_mask_element3{
      "slice_enable_bitmap_mask_element3", sub_commit_unit_slice_enable_bitmap3,
      64, 0};
};

template <class DT> class DataMemoryAddressMode : public sfr::Block<DT> {
public:
  DataMemoryAddressMode() : sfr::Block<DT>::Block(0x30, 0x8, false) {}

public:
  sfr::Register<DT> data_memory_address_mode{"data_memory_address_mode", *this,
                                             0x0, 0x0};
  sfr::Bitfield<DT> address_mode{"address_mode", data_memory_address_mode, 1,
                                 0};
  sfr::Bitfield<DT> page_fault_mode{"page_fault_mode", data_memory_address_mode,
                                    1, 4};
};

template <class DT> class DataMemory : public sfr::Block<DT> {
public:
  DataMemory() : sfr::Block<DT>::Block(0x200, 0x158, false) {}

public:
  sfr::Register<DT> data_memory_page_table_entry0{
      "data_memory_page_table_entry0", *this, 0x0, 0x0};
  sfr::Bitfield<DT> page_table_entry0{"page_table_entry0",
                                      data_memory_page_table_entry0, 8, 0};
  sfr::Bitfield<DT> page_table_entry1{"page_table_entry1",
                                      data_memory_page_table_entry0, 8, 8};
  sfr::Bitfield<DT> page_table_entry2{"page_table_entry2",
                                      data_memory_page_table_entry0, 8, 16};
  sfr::Bitfield<DT> page_table_entry3{"page_table_entry3",
                                      data_memory_page_table_entry0, 8, 24};
  sfr::Bitfield<DT> page_table_entry4{"page_table_entry4",
                                      data_memory_page_table_entry0, 8, 32};
  sfr::Bitfield<DT> page_table_entry5{"page_table_entry5",
                                      data_memory_page_table_entry0, 8, 40};
  sfr::Bitfield<DT> page_table_entry6{"page_table_entry6",
                                      data_memory_page_table_entry0, 8, 48};
  sfr::Bitfield<DT> page_table_entry7{"page_table_entry7",
                                      data_memory_page_table_entry0, 8, 56};
  sfr::Register<DT> data_memory_page_table_entry1{
      "data_memory_page_table_entry1", *this, 0x8, 0x0};
  sfr::Bitfield<DT> page_table_entry8{"page_table_entry8",
                                      data_memory_page_table_entry1, 8, 0};
  sfr::Bitfield<DT> page_table_entry9{"page_table_entry9",
                                      data_memory_page_table_entry1, 8, 8};
  sfr::Bitfield<DT> page_table_entry10{"page_table_entry10",
                                       data_memory_page_table_entry1, 8, 16};
  sfr::Bitfield<DT> page_table_entry11{"page_table_entry11",
                                       data_memory_page_table_entry1, 8, 24};
  sfr::Bitfield<DT> page_table_entry12{"page_table_entry12",
                                       data_memory_page_table_entry1, 8, 32};
  sfr::Bitfield<DT> page_table_entry13{"page_table_entry13",
                                       data_memory_page_table_entry1, 8, 40};
  sfr::Bitfield<DT> page_table_entry14{"page_table_entry14",
                                       data_memory_page_table_entry1, 8, 48};
  sfr::Bitfield<DT> page_table_entry15{"page_table_entry15",
                                       data_memory_page_table_entry1, 8, 56};
  sfr::Register<DT> data_memory_page_table_entry2{
      "data_memory_page_table_entry2", *this, 0x10, 0x0};
  sfr::Bitfield<DT> page_table_entry16{"page_table_entry16",
                                       data_memory_page_table_entry2, 8, 0};
  sfr::Bitfield<DT> page_table_entry17{"page_table_entry17",
                                       data_memory_page_table_entry2, 8, 8};
  sfr::Bitfield<DT> page_table_entry18{"page_table_entry18",
                                       data_memory_page_table_entry2, 8, 16};
  sfr::Bitfield<DT> page_table_entry19{"page_table_entry19",
                                       data_memory_page_table_entry2, 8, 24};
  sfr::Bitfield<DT> page_table_entry20{"page_table_entry20",
                                       data_memory_page_table_entry2, 8, 32};
  sfr::Bitfield<DT> page_table_entry21{"page_table_entry21",
                                       data_memory_page_table_entry2, 8, 40};
  sfr::Bitfield<DT> page_table_entry22{"page_table_entry22",
                                       data_memory_page_table_entry2, 8, 48};
  sfr::Bitfield<DT> page_table_entry23{"page_table_entry23",
                                       data_memory_page_table_entry2, 8, 56};
  sfr::Register<DT> data_memory_page_table_entry3{
      "data_memory_page_table_entry3", *this, 0x18, 0x0};
  sfr::Bitfield<DT> page_table_entry24{"page_table_entry24",
                                       data_memory_page_table_entry3, 8, 0};
  sfr::Bitfield<DT> page_table_entry25{"page_table_entry25",
                                       data_memory_page_table_entry3, 8, 8};
  sfr::Bitfield<DT> page_table_entry26{"page_table_entry26",
                                       data_memory_page_table_entry3, 8, 16};
  sfr::Bitfield<DT> page_table_entry27{"page_table_entry27",
                                       data_memory_page_table_entry3, 8, 24};
  sfr::Bitfield<DT> page_table_entry28{"page_table_entry28",
                                       data_memory_page_table_entry3, 8, 32};
  sfr::Bitfield<DT> page_table_entry29{"page_table_entry29",
                                       data_memory_page_table_entry3, 8, 40};
  sfr::Bitfield<DT> page_table_entry30{"page_table_entry30",
                                       data_memory_page_table_entry3, 8, 48};
  sfr::Bitfield<DT> page_table_entry31{"page_table_entry31",
                                       data_memory_page_table_entry3, 8, 56};
  sfr::Register<DT> data_memory_page_table_entry4{
      "data_memory_page_table_entry4", *this, 0x20, 0x0};
  sfr::Bitfield<DT> page_table_entry32{"page_table_entry32",
                                       data_memory_page_table_entry4, 8, 0};
  sfr::Bitfield<DT> page_table_entry33{"page_table_entry33",
                                       data_memory_page_table_entry4, 8, 8};
  sfr::Bitfield<DT> page_table_entry34{"page_table_entry34",
                                       data_memory_page_table_entry4, 8, 16};
  sfr::Bitfield<DT> page_table_entry35{"page_table_entry35",
                                       data_memory_page_table_entry4, 8, 24};
  sfr::Bitfield<DT> page_table_entry36{"page_table_entry36",
                                       data_memory_page_table_entry4, 8, 32};
  sfr::Bitfield<DT> page_table_entry37{"page_table_entry37",
                                       data_memory_page_table_entry4, 8, 40};
  sfr::Bitfield<DT> page_table_entry38{"page_table_entry38",
                                       data_memory_page_table_entry4, 8, 48};
  sfr::Bitfield<DT> page_table_entry39{"page_table_entry39",
                                       data_memory_page_table_entry4, 8, 56};
  sfr::Register<DT> data_memory_page_table_entry5{
      "data_memory_page_table_entry5", *this, 0x28, 0x0};
  sfr::Bitfield<DT> page_table_entry40{"page_table_entry40",
                                       data_memory_page_table_entry5, 8, 0};
  sfr::Bitfield<DT> page_table_entry41{"page_table_entry41",
                                       data_memory_page_table_entry5, 8, 8};
  sfr::Bitfield<DT> page_table_entry42{"page_table_entry42",
                                       data_memory_page_table_entry5, 8, 16};
  sfr::Bitfield<DT> page_table_entry43{"page_table_entry43",
                                       data_memory_page_table_entry5, 8, 24};
  sfr::Bitfield<DT> page_table_entry44{"page_table_entry44",
                                       data_memory_page_table_entry5, 8, 32};
  sfr::Bitfield<DT> page_table_entry45{"page_table_entry45",
                                       data_memory_page_table_entry5, 8, 40};
  sfr::Bitfield<DT> page_table_entry46{"page_table_entry46",
                                       data_memory_page_table_entry5, 8, 48};
  sfr::Bitfield<DT> page_table_entry47{"page_table_entry47",
                                       data_memory_page_table_entry5, 8, 56};
  sfr::Register<DT> data_memory_page_table_entry6{
      "data_memory_page_table_entry6", *this, 0x30, 0x0};
  sfr::Bitfield<DT> page_table_entry48{"page_table_entry48",
                                       data_memory_page_table_entry6, 8, 0};
  sfr::Bitfield<DT> page_table_entry49{"page_table_entry49",
                                       data_memory_page_table_entry6, 8, 8};
  sfr::Bitfield<DT> page_table_entry50{"page_table_entry50",
                                       data_memory_page_table_entry6, 8, 16};
  sfr::Bitfield<DT> page_table_entry51{"page_table_entry51",
                                       data_memory_page_table_entry6, 8, 24};
  sfr::Bitfield<DT> page_table_entry52{"page_table_entry52",
                                       data_memory_page_table_entry6, 8, 32};
  sfr::Bitfield<DT> page_table_entry53{"page_table_entry53",
                                       data_memory_page_table_entry6, 8, 40};
  sfr::Bitfield<DT> page_table_entry54{"page_table_entry54",
                                       data_memory_page_table_entry6, 8, 48};
  sfr::Bitfield<DT> page_table_entry55{"page_table_entry55",
                                       data_memory_page_table_entry6, 8, 56};
  sfr::Register<DT> data_memory_page_table_entry7{
      "data_memory_page_table_entry7", *this, 0x38, 0x0};
  sfr::Bitfield<DT> page_table_entry56{"page_table_entry56",
                                       data_memory_page_table_entry7, 8, 0};
  sfr::Bitfield<DT> page_table_entry57{"page_table_entry57",
                                       data_memory_page_table_entry7, 8, 8};
  sfr::Bitfield<DT> page_table_entry58{"page_table_entry58",
                                       data_memory_page_table_entry7, 8, 16};
  sfr::Bitfield<DT> page_table_entry59{"page_table_entry59",
                                       data_memory_page_table_entry7, 8, 24};
  sfr::Bitfield<DT> page_table_entry60{"page_table_entry60",
                                       data_memory_page_table_entry7, 8, 32};
  sfr::Bitfield<DT> page_table_entry61{"page_table_entry61",
                                       data_memory_page_table_entry7, 8, 40};
  sfr::Bitfield<DT> page_table_entry62{"page_table_entry62",
                                       data_memory_page_table_entry7, 8, 48};
  sfr::Bitfield<DT> page_table_entry63{"page_table_entry63",
                                       data_memory_page_table_entry7, 8, 56};
  sfr::Register<DT> data_memory_page_table_entry8{
      "data_memory_page_table_entry8", *this, 0x40, 0x0};
  sfr::Bitfield<DT> page_table_entry64{"page_table_entry64",
                                       data_memory_page_table_entry8, 8, 0};
  sfr::Bitfield<DT> page_table_entry65{"page_table_entry65",
                                       data_memory_page_table_entry8, 8, 8};
  sfr::Bitfield<DT> page_table_entry66{"page_table_entry66",
                                       data_memory_page_table_entry8, 8, 16};
  sfr::Bitfield<DT> page_table_entry67{"page_table_entry67",
                                       data_memory_page_table_entry8, 8, 24};
  sfr::Bitfield<DT> page_table_entry68{"page_table_entry68",
                                       data_memory_page_table_entry8, 8, 32};
  sfr::Bitfield<DT> page_table_entry69{"page_table_entry69",
                                       data_memory_page_table_entry8, 8, 40};
  sfr::Bitfield<DT> page_table_entry70{"page_table_entry70",
                                       data_memory_page_table_entry8, 8, 48};
  sfr::Bitfield<DT> page_table_entry71{"page_table_entry71",
                                       data_memory_page_table_entry8, 8, 56};
  sfr::Register<DT> data_memory_page_table_entry9{
      "data_memory_page_table_entry9", *this, 0x48, 0x0};
  sfr::Bitfield<DT> page_table_entry72{"page_table_entry72",
                                       data_memory_page_table_entry9, 8, 0};
  sfr::Bitfield<DT> page_table_entry73{"page_table_entry73",
                                       data_memory_page_table_entry9, 8, 8};
  sfr::Bitfield<DT> page_table_entry74{"page_table_entry74",
                                       data_memory_page_table_entry9, 8, 16};
  sfr::Bitfield<DT> page_table_entry75{"page_table_entry75",
                                       data_memory_page_table_entry9, 8, 24};
  sfr::Bitfield<DT> page_table_entry76{"page_table_entry76",
                                       data_memory_page_table_entry9, 8, 32};
  sfr::Bitfield<DT> page_table_entry77{"page_table_entry77",
                                       data_memory_page_table_entry9, 8, 40};
  sfr::Bitfield<DT> page_table_entry78{"page_table_entry78",
                                       data_memory_page_table_entry9, 8, 48};
  sfr::Bitfield<DT> page_table_entry79{"page_table_entry79",
                                       data_memory_page_table_entry9, 8, 56};
  sfr::Register<DT> data_memory_page_table_entry10{
      "data_memory_page_table_entry10", *this, 0x50, 0x0};
  sfr::Bitfield<DT> page_table_entry80{"page_table_entry80",
                                       data_memory_page_table_entry10, 8, 0};
  sfr::Bitfield<DT> page_table_entry81{"page_table_entry81",
                                       data_memory_page_table_entry10, 8, 8};
  sfr::Bitfield<DT> page_table_entry82{"page_table_entry82",
                                       data_memory_page_table_entry10, 8, 16};
  sfr::Bitfield<DT> page_table_entry83{"page_table_entry83",
                                       data_memory_page_table_entry10, 8, 24};
  sfr::Bitfield<DT> page_table_entry84{"page_table_entry84",
                                       data_memory_page_table_entry10, 8, 32};
  sfr::Bitfield<DT> page_table_entry85{"page_table_entry85",
                                       data_memory_page_table_entry10, 8, 40};
  sfr::Bitfield<DT> page_table_entry86{"page_table_entry86",
                                       data_memory_page_table_entry10, 8, 48};
  sfr::Bitfield<DT> page_table_entry87{"page_table_entry87",
                                       data_memory_page_table_entry10, 8, 56};
  sfr::Register<DT> data_memory_page_table_entry11{
      "data_memory_page_table_entry11", *this, 0x58, 0x0};
  sfr::Bitfield<DT> page_table_entry88{"page_table_entry88",
                                       data_memory_page_table_entry11, 8, 0};
  sfr::Bitfield<DT> page_table_entry89{"page_table_entry89",
                                       data_memory_page_table_entry11, 8, 8};
  sfr::Bitfield<DT> page_table_entry90{"page_table_entry90",
                                       data_memory_page_table_entry11, 8, 16};
  sfr::Bitfield<DT> page_table_entry91{"page_table_entry91",
                                       data_memory_page_table_entry11, 8, 24};
  sfr::Bitfield<DT> page_table_entry92{"page_table_entry92",
                                       data_memory_page_table_entry11, 8, 32};
  sfr::Bitfield<DT> page_table_entry93{"page_table_entry93",
                                       data_memory_page_table_entry11, 8, 40};
  sfr::Bitfield<DT> page_table_entry94{"page_table_entry94",
                                       data_memory_page_table_entry11, 8, 48};
  sfr::Bitfield<DT> page_table_entry95{"page_table_entry95",
                                       data_memory_page_table_entry11, 8, 56};
  sfr::Register<DT> data_memory_page_table_entry12{
      "data_memory_page_table_entry12", *this, 0x60, 0x0};
  sfr::Bitfield<DT> page_table_entry96{"page_table_entry96",
                                       data_memory_page_table_entry12, 8, 0};
  sfr::Bitfield<DT> page_table_entry97{"page_table_entry97",
                                       data_memory_page_table_entry12, 8, 8};
  sfr::Bitfield<DT> page_table_entry98{"page_table_entry98",
                                       data_memory_page_table_entry12, 8, 16};
  sfr::Bitfield<DT> page_table_entry99{"page_table_entry99",
                                       data_memory_page_table_entry12, 8, 24};
  sfr::Bitfield<DT> page_table_entry100{"page_table_entry100",
                                        data_memory_page_table_entry12, 8, 32};
  sfr::Bitfield<DT> page_table_entry101{"page_table_entry101",
                                        data_memory_page_table_entry12, 8, 40};
  sfr::Bitfield<DT> page_table_entry102{"page_table_entry102",
                                        data_memory_page_table_entry12, 8, 48};
  sfr::Bitfield<DT> page_table_entry103{"page_table_entry103",
                                        data_memory_page_table_entry12, 8, 56};
  sfr::Register<DT> data_memory_page_table_entry13{
      "data_memory_page_table_entry13", *this, 0x68, 0x0};
  sfr::Bitfield<DT> page_table_entry104{"page_table_entry104",
                                        data_memory_page_table_entry13, 8, 0};
  sfr::Bitfield<DT> page_table_entry105{"page_table_entry105",
                                        data_memory_page_table_entry13, 8, 8};
  sfr::Bitfield<DT> page_table_entry106{"page_table_entry106",
                                        data_memory_page_table_entry13, 8, 16};
  sfr::Bitfield<DT> page_table_entry107{"page_table_entry107",
                                        data_memory_page_table_entry13, 8, 24};
  sfr::Bitfield<DT> page_table_entry108{"page_table_entry108",
                                        data_memory_page_table_entry13, 8, 32};
  sfr::Bitfield<DT> page_table_entry109{"page_table_entry109",
                                        data_memory_page_table_entry13, 8, 40};
  sfr::Bitfield<DT> page_table_entry110{"page_table_entry110",
                                        data_memory_page_table_entry13, 8, 48};
  sfr::Bitfield<DT> page_table_entry111{"page_table_entry111",
                                        data_memory_page_table_entry13, 8, 56};
  sfr::Register<DT> data_memory_page_table_entry14{
      "data_memory_page_table_entry14", *this, 0x70, 0x0};
  sfr::Bitfield<DT> page_table_entry112{"page_table_entry112",
                                        data_memory_page_table_entry14, 8, 0};
  sfr::Bitfield<DT> page_table_entry113{"page_table_entry113",
                                        data_memory_page_table_entry14, 8, 8};
  sfr::Bitfield<DT> page_table_entry114{"page_table_entry114",
                                        data_memory_page_table_entry14, 8, 16};
  sfr::Bitfield<DT> page_table_entry115{"page_table_entry115",
                                        data_memory_page_table_entry14, 8, 24};
  sfr::Bitfield<DT> page_table_entry116{"page_table_entry116",
                                        data_memory_page_table_entry14, 8, 32};
  sfr::Bitfield<DT> page_table_entry117{"page_table_entry117",
                                        data_memory_page_table_entry14, 8, 40};
  sfr::Bitfield<DT> page_table_entry118{"page_table_entry118",
                                        data_memory_page_table_entry14, 8, 48};
  sfr::Bitfield<DT> page_table_entry119{"page_table_entry119",
                                        data_memory_page_table_entry14, 8, 56};
  sfr::Register<DT> data_memory_page_table_entry15{
      "data_memory_page_table_entry15", *this, 0x78, 0x0};
  sfr::Bitfield<DT> page_table_entry120{"page_table_entry120",
                                        data_memory_page_table_entry15, 8, 0};
  sfr::Bitfield<DT> page_table_entry121{"page_table_entry121",
                                        data_memory_page_table_entry15, 8, 8};
  sfr::Bitfield<DT> page_table_entry122{"page_table_entry122",
                                        data_memory_page_table_entry15, 8, 16};
  sfr::Bitfield<DT> page_table_entry123{"page_table_entry123",
                                        data_memory_page_table_entry15, 8, 24};
  sfr::Bitfield<DT> page_table_entry124{"page_table_entry124",
                                        data_memory_page_table_entry15, 8, 32};
  sfr::Bitfield<DT> page_table_entry125{"page_table_entry125",
                                        data_memory_page_table_entry15, 8, 40};
  sfr::Bitfield<DT> page_table_entry126{"page_table_entry126",
                                        data_memory_page_table_entry15, 8, 48};
  sfr::Bitfield<DT> page_table_entry127{"page_table_entry127",
                                        data_memory_page_table_entry15, 8, 56};
  sfr::Register<DT> data_memory_error_count_enable{
      "data_memory_error_count_enable", *this, 0x100, 0x0};
  sfr::Bitfield<DT> single_bit_error_counter_enable{
      "single_bit_error_counter_enable", data_memory_error_count_enable, 16, 0};
  sfr::Bitfield<DT> multi_bit_error_counter_enable{
      "multi_bit_error_counter_enable", data_memory_error_count_enable, 16, 16};
  sfr::Register<DT> data_memory_error_count_clear{
      "data_memory_error_count_clear", *this, 0x108, 0x0};
  sfr::Bitfield<DT> single_bit_error_counter_clear{
      "single_bit_error_counter_clear", data_memory_error_count_clear, 1, 0};
  sfr::Bitfield<DT> multi_bit_error_counter_clear{
      "multi_bit_error_counter_clear", data_memory_error_count_clear, 1, 1};
  sfr::Register<DT> data_memory_single_bit_error_count_value{
      "data_memory_single_bit_error_count_value", *this, 0x110, 0x0};
  sfr::Bitfield<DT> single_bit_error_counter_value{
      "single_bit_error_counter_value",
      data_memory_single_bit_error_count_value, 16, 0};
  sfr::Register<DT> data_memory_multi_bit_error_count_value{
      "data_memory_multi_bit_error_count_value", *this, 0x118, 0x0};
  sfr::Bitfield<DT> multi_bit_error_counter_value{
      "multi_bit_error_counter_value", data_memory_multi_bit_error_count_value,
      16, 0};
  sfr::Register<DT> data_memory_interrupt_mask{"data_memory_interrupt_mask",
                                               *this, 0x120, 0x0};
  sfr::Bitfield<DT> multi_bit_error_interrupt_mask{
      "multi_bit_error_interrupt_mask", data_memory_interrupt_mask, 16, 0};
  sfr::Register<DT> data_memory_interrupt_clear{"data_memory_interrupt_clear",
                                                *this, 0x128, 0x0};
  sfr::Bitfield<DT> multi_bit_error_interrupt_clear{
      "multi_bit_error_interrupt_clear", data_memory_interrupt_clear, 16, 0};
  sfr::Register<DT> data_memory_interrupt_raw{"data_memory_interrupt_raw",
                                              *this, 0x130, 0x0};
  sfr::Bitfield<DT> multi_bit_error_interrupt_raw{
      "multi_bit_error_interrupt_raw", data_memory_interrupt_raw, 16, 0};
  sfr::Register<DT> data_memory_interrupt_status{"data_memory_interrupt_status",
                                                 *this, 0x138, 0x0};
  sfr::Bitfield<DT> multi_bit_error_interrupt_status{
      "multi_bit_error_interrupt_status", data_memory_interrupt_status, 16, 0};
  sfr::Register<DT> data_memory_page_fault_control{
      "data_memory_page_fault_control", *this, 0x140, 0x0};
  sfr::Bitfield<DT> page_fault_clear{"page_fault_clear",
                                     data_memory_page_fault_control, 1, 0};
  sfr::Register<DT> data_memory_page_fault_status{
      "data_memory_page_fault_status", *this, 0x150, 0x0};
  sfr::Bitfield<DT> page_fault_bank{"page_fault_bank",
                                    data_memory_page_fault_status, 16, 0};
};

} // namespace slice
} // namespace sfr
} // namespace mlir::furiosa
