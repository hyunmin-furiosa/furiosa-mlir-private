#pragma once

#include "furiosa-mlir/Dialect/Furiosa/IR/SfrBase.h"

namespace mlir::furiosa::sfr {

using DT = std::uint64_t;

class SubFetchUnit : public sfr::Block<DT> {
public:
  SubFetchUnit() : sfr::Block<DT>::Block(0x100, 0x70) {}

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
  sfr::Register<DT> sub_fetch_unit_custom_snoop_bitmap1{
      "sub_fetch_unit_custom_snoop_bitmap1", *this, 0x58, 0x0};
  sfr::Register<DT> sub_fetch_unit_custom_snoop_bitmap2{
      "sub_fetch_unit_custom_snoop_bitmap2", *this, 0x60, 0x0};
  sfr::Register<DT> sub_fetch_unit_custom_snoop_bitmap3{
      "sub_fetch_unit_custom_snoop_bitmap3", *this, 0x68, 0x0};
};

class SubCommitUnit : public sfr::Block<DT> {
public:
  SubCommitUnit() : sfr::Block<DT>::Block(0x198, 0x68) {}

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
  sfr::Register<DT> sub_commit_unit_slice_enable_bitmap1{
      "sub_commit_unit_slice_enable_bitmap1", *this, 0x50, 0x0};
  sfr::Register<DT> sub_commit_unit_slice_enable_bitmap2{
      "sub_commit_unit_slice_enable_bitmap2", *this, 0x58, 0x0};
  sfr::Register<DT> sub_commit_unit_slice_enable_bitmap3{
      "sub_commit_unit_slice_enable_bitmap3", *this, 0x60, 0x0};
};

} // namespace mlir::furiosa::sfr
