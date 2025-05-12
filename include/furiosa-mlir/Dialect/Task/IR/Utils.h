#pragma once

#include "furiosa-mlir/Dialect/Task/IR/RenegadeCommands.h"
#include "furiosa-mlir/Dialect/Task/IR/RenegadeSfr.h"
#include "furiosa-mlir/Dialect/Task/IR/TaskOps.h"

namespace mlir::furiosa::task {

FailureOr<std::uint32_t> getOpcode(Operation &op) {
  return llvm::TypeSwitch<Operation *, FailureOr<std::uint32_t>>(&op)
      .Case<tuc::ItosfrOp>([&](auto op) { return 0x01; })
      .Case<tuc::RtosfrOp>([&](auto op) { return 0x02; })
      .Case<tuc::RtosfriOp>([&](auto op) { return 0x03; })
      .Case<tuc::MtosfrOp>([&](auto op) { return 0x04; })
      .Case<DynamicMtosfrOp>([&](auto op) { return 0x04; })
      .Case<tuc::StosfrOp>([&](auto op) { return 0x05; })
      .Case<tuc::SfrtosOp>([&](auto op) { return 0x06; })
      .Case<tuc::StallOp>([&](auto op) { return 0x07; })
      .Case<tuc::ItosOp>([&](auto op) { return 0x08; })
      .Case<tuc::ItosiOp>([&](auto op) { return 0x09; })
      .Case<tuc::StosOp>([&](auto op) { return 0x0a; })
      .Case<tuc::StotabOp>([&](auto op) { return 0x0b; })
      .Case<tuc::StotrfOp>([&](auto op) { return 0x0c; })
      .Case<tuc::StovrfOp>([&](auto op) { return 0x0d; })
      .Case<tuc::ExecutionOp>([&](auto op) { return 0x10; })
      .Case<tuc::WaitOp>([&](auto op) { return 0x11; })
      .Case<tuc::WaitiOp>([&](auto op) { return 0x15; })
      .Case<tuc::InterruptOp>([&](auto op) { return 0x12; })
      .Case<tuc::DmaOp>([&](auto op) { return 0x13; })
      .Case<tuc::Dma1Op>([&](auto op) { return 0x14; })
      .Case<tuc::DmawOp>([&](auto op) { return 0x16; })
      .Case<DynamicDmawOp>([&](auto op) { return 0x16; })
      .Case<tuc::ProfileOp>([&](auto op) { return 0x18; })
      .Case<tuc::ProfileiOp>([&](auto op) { return 0x19; })
      .Case<tuc::PrflushOp>([&](auto op) { return 0x1a; })
      .Default([&](Operation *) {
        return op.emitOpError(
            "unable to interpret as furiosa dialect operator");
      });
}

FailureOr<std::tuple<TensorUnitCommand, SmallVector<GeneralRegister>>>
getCommand(Operation &op) {
  TensorUnitCommand command = TensorUnitCommand(*getOpcode(op));
  SmallVector<GeneralRegister> registers;
  return llvm::TypeSwitch<Operation *,
                          FailureOr<std::tuple<TensorUnitCommand,
                                               SmallVector<GeneralRegister>>>>(
             &op)
      .Case<tuc::ItosfrOp>([&](auto op) {
        command.itosfr.value = op.getValue();
        {
          GeneralRegister reg;
          reg.itosfr_0.sfr_address = op.getSfrAddress();
          reg.itosfr_0.size = op.getSize();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::RtosfrOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.rtosfr_0.value = op.getValue();
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          reg.rtosfr_1.sfr_address = op.getSfrAddress();
          reg.rtosfr_1.size = op.getSize();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::RtosfriOp>([&](auto op) {
        command.rtosfri.log_size = op.getLogSize();
        command.rtosfri.sfr_address = op.getSfrAddress();
        {
          GeneralRegister reg;
          reg.rtosfr_0.value = op.getValue();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::MtosfrOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.mtosfr_0.spm_address = op.getSpmAddress();
          reg.mtosfr_0.size = op.getSize();
          reg.mtosfr_0.sfr_address = op.getSfrAddress();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<DynamicMtosfrOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.mtosfr_0.sfr_address = op.getSfrAddress();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::StosfrOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.stosfr_0.fetch_base = op.getFetchBase();
          reg.stosfr_0.fetch_size = op.getFetchSize();
          reg.stosfr_0.sfr_address = op.getSfrAddress();
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          reg.stosfr_1.topology = op.getTopology();
          reg.stosfr_1.slice_log_size = op.getSliceLogSize();
          reg.stosfr_1.dim0_log_size = op.getDim0LogSize();
          reg.stosfr_1.dim1_log_size = op.getDim1LogSize();
          reg.stosfr_1.data_offset = op.getDataOffset();
          reg.stosfr_1.size = op.getSize();
          reg.stosfr_1.words_per_packet = op.getWordsPerPacket();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::SfrtosOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.sfrtos_0.commit_base = op.getCommitBase();
          reg.sfrtos_0.commit_limit = op.getCommitLimit();
          reg.sfrtos_0.sfr_address = op.getSfrAddress();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::StallOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.stall_0.cycle = op.getCycle();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::ItosOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.itos_0.address_begin = op.getAddressBegin();
          reg.itos_0.address_end = op.getAddressEnd();
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          reg.itos_1.value = op.getValue();
          reg.itos_1.dim1_log_size = op.getDim1LogSize();
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          reg.itos_2.limit1 = op.getLimit1();
          reg.itos_2.stride1 = op.getStride1();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::ItosiOp>([&](auto op) {
        command.value = op.getValue();
        {
          GeneralRegister reg;
          reg.itos_0.address_begin = op.getAddressBegin();
          reg.itos_0.address_end = op.getAddressEnd();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::StosOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.stos_0.address_begin = op.getAddressBegin();
          reg.stos_0.address_end = op.getAddressEnd();
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          reg.stos_1.destination_begin = op.getDestinationBegin();
          reg.stos_1.slice_log_size = op.getSliceLogSize();
          reg.stos_1.dim1_log_size = op.getDim1LogSize();
          reg.stos_1.words_per_packet = op.getWordsPerPacket();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::StotabOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.stotab_0.fetch_base = op.getFetchBase();
          reg.stotab_0.fetch_limit = op.getFetchLimit();
          reg.stotab_0.tables = op.getTables();
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          reg.stotab_1.commit_base = op.getCommitBase();
          reg.stotab_1.topology = op.getTopology();
          reg.stotab_1.slice_log_size = op.getSliceLogSize();
          reg.stotab_1.dim0_log_size = op.getDim0LogSize();
          reg.stotab_1.dim1_log_size = op.getDim1LogSize();
          reg.stotab_1.words_per_packet = op.getWordsPerPacket();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::StotrfOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.stotrf_0.fetch_base = op.getFetchBase();
          reg.stotrf_0.fetch_limit = op.getFetchLimit();
          reg.stotrf_0.type_conversion = op.getTypeConversion();
          reg.stotrf_0.write_mode = op.getWriteMode();
          reg.stotrf_0.zeropoint = op.getZeropoint();
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          reg.stotrf_1.topology = op.getTopology();
          reg.stotrf_1.slice_log_size = op.getSliceLogSize();
          reg.stotrf_1.dim0_log_size = op.getDim0LogSize();
          reg.stotrf_1.dim1_log_size = op.getDim1LogSize();
          reg.stotrf_1.flits_per_packet = op.getFlitsPerPacket();
          reg.stotrf_1.dim0_chunk_size = op.getDim0ChunkSize();
          reg.stotrf_1.skip_flit_count = op.getSkipFlitCount();
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          reg.stotrf_2.write_row_base = op.getWriteRowBase();
          reg.stotrf_2.write_row_count = op.getWriteRowCount();
          reg.stotrf_2.write_mac_row = op.getWriteMacRow();
          reg.stotrf_2.flits_per_period = op.getFlitsPerPeriod();
          reg.stotrf_2.valid_flits_per_period = op.getValidFlitsPerPeriod();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::StovrfOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.stovrf_0.fetch_base = op.getFetchBase();
          reg.stovrf_0.fetch_limit = op.getFetchLimit();
          reg.stovrf_0.type_conversion = op.getTypeConversion();
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          reg.stovrf_1.topology = op.getTopology();
          reg.stovrf_1.slice_log_size = op.getSliceLogSize();
          reg.stovrf_1.dim0_log_size = op.getDim0LogSize();
          reg.stovrf_1.dim1_log_size = op.getDim1LogSize();
          reg.stovrf_1.words_per_packet = op.getWordsPerPacket();
          reg.stovrf_1.skip_flit_count = op.getSkipFlitCount();
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          reg.stovrf_2.write_row_base = op.getWriteRowBase();
          reg.stovrf_2.write_row_count = op.getWriteRowCount();
          reg.stovrf_2.write_row_stride = op.getWriteRowStride();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::ExecutionOp>([&](auto op) {
        command.execution.subunit_bitmap = op.getSubunitBitmap();
        command.execution.context_id = op.getContextId();
        command.execution.target_context = op.getTargetContext();
        return std::make_tuple(command, registers);
      })
      .Case<tuc::WaitOp, tuc::WaitiOp>([&](auto op) {
        command.wait.dma_tag_id = op.getDmaTagId();
        command.wait.type = op.getType();
        command.wait.target_context = op.getTargetContext();
        return std::make_tuple(command, registers);
      })
      .Case<tuc::InterruptOp>(
          [&](auto op) { return std::make_tuple(command, registers); })
      .Case<tuc::DmaOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.dma_0.pe0_desc_addr = op.getPe0DescAddr();
          reg.dma_0.pe1_desc_addr = op.getPe1DescAddr();
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          reg.dma_1.pe2_desc_addr = op.getPe2DescAddr();
          reg.dma_1.pe3_desc_addr = op.getPe3DescAddr();
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          reg.dma_2.dma_tag_id = op.getDmaTagId();
          reg.dma_2.profile = op.getProfile();
          reg.dma_2.profile_id = op.getProfileId();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::Dma1Op>([&](auto op) {
        {
          GeneralRegister reg;
          reg.dma1_0.desc_addr = op.getDescAddr();
          reg.dma1_0.pe_valid_bitmap = op.getPeValidBitmap();
          reg.dma1_0.dma_tag_id = op.getDmaTagId();
          reg.dma1_0.profile = op.getProfile();
          reg.dma1_0.profile_id = op.getProfileId();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::DmawOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.dmaw_0.desc_addr = op.getPe0DescAddr();
          reg.dmaw_0.dma_tag_id = op.getDmaTagId();
          reg.dmaw_0.profile = op.getProfile();
          reg.dmaw_0.profile_id = op.getProfileId();
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          reg.dmaw_0.desc_addr = op.getPe1DescAddr();
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          reg.dmaw_0.desc_addr = op.getPe2DescAddr();
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          reg.dmaw_0.desc_addr = op.getPe3DescAddr();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<DynamicDmawOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.dmaw_0.dma_tag_id = op.getDmaTagId();
          reg.dmaw_0.profile = op.getProfile();
          reg.dmaw_0.profile_id = op.getProfileId();
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          registers.push_back(reg);
        }
        {
          GeneralRegister reg;
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::ProfileOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.profile_0.profile_id = op.getProfileId();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<tuc::ProfileiOp>([&](auto op) {
        command.profilei.profile_id = op.getProfileId();
        return std::make_tuple(command, registers);
      })
      .Case<tuc::PrflushOp>(
          [&](auto op) { return std::make_tuple(command, registers); })
      .Default([&](Operation *) {
        return op.emitOpError(
            "unable to interpret as furiosa dialect operator");
      });
}

template <typename T, std::enable_if_t<
                          std::is_same_v<T, sfr::StaticSfrDotProductEngineOp> ||
                              std::is_same_v<T, sfr::SfrDotProductEngineOp>,
                          bool> = true>
std::vector<sfr_data_t> getSfrDotProductEngine(T &op) {
  sfr::slice::DotProductEngineMainContext<sfr_data_t> sfr{};
  sfr.reg_indexer_base = op.getRegIndexerBase();
  sfr.acc_indexer_base = op.getAccIndexerBase();
  sfr.flits_per_input = op.getFlitsPerInput();
  sfr.feed_input_transpose = op.getFeedInputTranspose();
  sfr.initial_shift_dim = op.getInitialShiftDim();
  sfr.shift_stride = op.getShiftStride();
  sfr.pop_dim = op.getPopDim();
  sfr.shift_dim = op.getShiftDim();
  sfr.channel_config = op.getChannelConfig();
  sfr.feed_data_type = op.getFeedDataType();
  auto initial_shift = op.getInitialShift();
  sfr.initial_shift_element0 =
      dyn_cast_or_null<IntegerAttr>(initial_shift[0]).getInt();
  sfr.initial_shift_element1 =
      dyn_cast_or_null<IntegerAttr>(initial_shift[1]).getInt();
  sfr.initial_shift_element2 =
      dyn_cast_or_null<IntegerAttr>(initial_shift[2]).getInt();
  sfr.initial_shift_element3 =
      dyn_cast_or_null<IntegerAttr>(initial_shift[3]).getInt();
  sfr.initial_shift_element4 =
      dyn_cast_or_null<IntegerAttr>(initial_shift[4]).getInt();
  sfr.initial_shift_element5 =
      dyn_cast_or_null<IntegerAttr>(initial_shift[5]).getInt();
  sfr.initial_shift_element6 =
      dyn_cast_or_null<IntegerAttr>(initial_shift[6]).getInt();
  sfr.initial_shift_element7 =
      dyn_cast_or_null<IntegerAttr>(initial_shift[7]).getInt();
  auto iter_seq_limits = op.getIterSeqLimits();
  sfr.iter_seq_limits_element0 =
      dyn_cast_or_null<IntegerAttr>(iter_seq_limits[0]).getInt();
  sfr.iter_seq_limits_element1 =
      dyn_cast_or_null<IntegerAttr>(iter_seq_limits[1]).getInt();
  sfr.iter_seq_limits_element2 =
      dyn_cast_or_null<IntegerAttr>(iter_seq_limits[2]).getInt();
  sfr.iter_seq_limits_element3 =
      dyn_cast_or_null<IntegerAttr>(iter_seq_limits[3]).getInt();
  sfr.iter_seq_limits_element4 =
      dyn_cast_or_null<IntegerAttr>(iter_seq_limits[4]).getInt();
  sfr.iter_seq_limits_element5 =
      dyn_cast_or_null<IntegerAttr>(iter_seq_limits[5]).getInt();
  sfr.iter_seq_limits_element6 =
      dyn_cast_or_null<IntegerAttr>(iter_seq_limits[6]).getInt();
  sfr.iter_seq_limits_element7 =
      dyn_cast_or_null<IntegerAttr>(iter_seq_limits[7]).getInt();
  auto reg_indexer_strides = op.getRegIndexerStrides();
  sfr.reg_indexer_strides_element1 =
      dyn_cast_or_null<IntegerAttr>(reg_indexer_strides[0]).getInt();
  sfr.reg_indexer_strides_element1 =
      dyn_cast_or_null<IntegerAttr>(reg_indexer_strides[1]).getInt();
  sfr.reg_indexer_strides_element2 =
      dyn_cast_or_null<IntegerAttr>(reg_indexer_strides[2]).getInt();
  sfr.reg_indexer_strides_element3 =
      dyn_cast_or_null<IntegerAttr>(reg_indexer_strides[3]).getInt();
  sfr.reg_indexer_strides_element4 =
      dyn_cast_or_null<IntegerAttr>(reg_indexer_strides[4]).getInt();
  sfr.reg_indexer_strides_element5 =
      dyn_cast_or_null<IntegerAttr>(reg_indexer_strides[5]).getInt();
  sfr.reg_indexer_strides_element6 =
      dyn_cast_or_null<IntegerAttr>(reg_indexer_strides[6]).getInt();
  sfr.reg_indexer_strides_element7 =
      dyn_cast_or_null<IntegerAttr>(reg_indexer_strides[7]).getInt();
  auto acc_indexer_strides = op.getAccIndexerStrides();
  sfr.acc_indexer_strides_element1 =
      dyn_cast_or_null<IntegerAttr>(acc_indexer_strides[0]).getInt();
  sfr.acc_indexer_strides_element1 =
      dyn_cast_or_null<IntegerAttr>(acc_indexer_strides[1]).getInt();
  sfr.acc_indexer_strides_element2 =
      dyn_cast_or_null<IntegerAttr>(acc_indexer_strides[2]).getInt();
  sfr.acc_indexer_strides_element3 =
      dyn_cast_or_null<IntegerAttr>(acc_indexer_strides[3]).getInt();
  sfr.acc_indexer_strides_element4 =
      dyn_cast_or_null<IntegerAttr>(acc_indexer_strides[4]).getInt();
  sfr.acc_indexer_strides_element5 =
      dyn_cast_or_null<IntegerAttr>(acc_indexer_strides[5]).getInt();
  sfr.acc_indexer_strides_element6 =
      dyn_cast_or_null<IntegerAttr>(acc_indexer_strides[6]).getInt();
  sfr.acc_indexer_strides_element7 =
      dyn_cast_or_null<IntegerAttr>(acc_indexer_strides[7]).getInt();
  sfr.acc_limit = op.getAccLimit();
  sfr.acc_cols = op.getAccCols();
  sfr.acc_reset = op.getAccReset();
  sfr.output_major = op.getOutputMajor();
  sfr.acc_init_value = op.getAccInitValue();
  sfr.mac_tree_operation = op.getMacTreeOperation();
  sfr.mac_tree_depth = op.getMacTreeDepth();
  sfr.mac_type = op.getMacType();
  sfr.mac_rows = op.getMacRows();
  sfr.fp_ieee_nan_multiplication = op.getFpIeeeNanMultiplication();
  sfr.fxp_shift_rounding_mode = op.getFxpShiftRoundingMode();
  sfr.data_type = op.getDataType();
  sfr.reg_read_log_size = op.getRegReadLogSize();
  sfr.reg_read_mode = op.getRegReadMode();
  sfr.reg_read_cache_mode = op.getRegReadCacheMode();

  return sfr.get_blocks();
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, sfr::StaticSfrMainCommitUnitOp> ||
                               std::is_same_v<T, sfr::SfrMainCommitUnitOp>,
                           bool> = true>
std::vector<sfr_data_t> getSfrMainCommitUnit(T &op) {
  sfr::slice::CommitUnitMainContext<sfr_data_t> sfr{};
  sfr.type_conversion = op.getTypeConversion();
  sfr.base = op.getBase();
  sfr.commit_in_size = op.getCommitInSize();
  sfr.commit_size = op.getCommitSize();
  auto limits = op.getLimits();
  sfr.limits_element0 = dyn_cast_or_null<IntegerAttr>(limits[0]).getInt();
  sfr.limits_element1 = dyn_cast_or_null<IntegerAttr>(limits[1]).getInt();
  sfr.limits_element2 = dyn_cast_or_null<IntegerAttr>(limits[2]).getInt();
  sfr.limits_element3 = dyn_cast_or_null<IntegerAttr>(limits[3]).getInt();
  sfr.limits_element4 = dyn_cast_or_null<IntegerAttr>(limits[4]).getInt();
  sfr.limits_element5 = dyn_cast_or_null<IntegerAttr>(limits[5]).getInt();
  sfr.limits_element6 = dyn_cast_or_null<IntegerAttr>(limits[6]).getInt();
  sfr.limits_element7 = dyn_cast_or_null<IntegerAttr>(limits[7]).getInt();
  auto strides = op.getStrides();
  sfr.strides_element0 = dyn_cast_or_null<IntegerAttr>(strides[0]).getInt();
  sfr.strides_element1 = dyn_cast_or_null<IntegerAttr>(strides[1]).getInt();
  sfr.strides_element2 = dyn_cast_or_null<IntegerAttr>(strides[2]).getInt();
  sfr.strides_element3 = dyn_cast_or_null<IntegerAttr>(strides[3]).getInt();
  sfr.strides_element4 = dyn_cast_or_null<IntegerAttr>(strides[4]).getInt();
  sfr.strides_element5 = dyn_cast_or_null<IntegerAttr>(strides[5]).getInt();
  sfr.strides_element6 = dyn_cast_or_null<IntegerAttr>(strides[6]).getInt();
  sfr.strides_element7 = dyn_cast_or_null<IntegerAttr>(strides[7]).getInt();
  auto slice_enable_bitmap_mask = op.getSliceEnableBitmapMask();
  sfr.commit_unit_slice_enable_bitmap0 =
      dyn_cast_or_null<IntegerAttr>(slice_enable_bitmap_mask[0]).getInt();
  sfr.commit_unit_slice_enable_bitmap1 =
      dyn_cast_or_null<IntegerAttr>(slice_enable_bitmap_mask[1]).getInt();
  sfr.commit_unit_slice_enable_bitmap2 =
      dyn_cast_or_null<IntegerAttr>(slice_enable_bitmap_mask[2]).getInt();
  sfr.commit_unit_slice_enable_bitmap3 =
      dyn_cast_or_null<IntegerAttr>(slice_enable_bitmap_mask[3]).getInt();

  return sfr.get_blocks();
}

template <typename T, std::enable_if_t<
                          std::is_same_v<T, sfr::StaticSfrMainDataPathUnitOp> ||
                              std::is_same_v<T, sfr::SfrMainDataPathUnitOp>,
                          bool> = true>
std::vector<sfr_data_t> getSfrMainDataPathUnit(T &op) {
  sfr::slice::OperationDataPathMainContext<sfr_data_t> sfr{};
  sfr.main_context = op.getMainContext();
  sfr.channel_config = op.getChannelConfig();

  return sfr.get_blocks();
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, sfr::StaticSfrMainFetchUnitOp> ||
                               std::is_same_v<T, sfr::SfrMainFetchUnitOp>,
                           bool> = true>
std::vector<sfr_data_t> getSfrMainFetchUnit(T &op) {
  sfr::slice::FetchUnitMainContext<sfr_data_t> sfr{};
  sfr.fetch_mode = op.getFetchMode();
  sfr.num_zero_points = op.getNumZeroPoints();
  sfr.zero_point0 = op.getZeroPoint0();
  sfr.zero_point1 = op.getZeroPoint1();
  sfr.table_entry_size = op.getTableEntrySize();
  sfr.tables = op.getTables();
  sfr.indirect_base = op.getIndirectBase();
  sfr.indirect_dim = op.getIndirectDim();
  sfr.table_base_mode = op.getTableBaseMode();
  sfr.indirect_pointer_size = op.getIndirectPointerSize();
  sfr.zeropoint_tail_mode = op.getZeropointTailMode();
  sfr.last_dim_pad_value = op.getLastDimPadValue();
  sfr.last_dim = op.getLastDim();
  sfr.pad_order = op.getPadOrder();
  sfr.last_dim_rightmost_valid_count_dim =
      op.getLastDimRightmostValidCountDim();
  sfr.last_dim_left_pad_count = op.getLastDimLeftPadCount();
  sfr.type_conversion = op.getTypeConversion();
  sfr.last_dim_left_pad_mode = op.getLastDimLeftPadMode();
  sfr.zeropoint_dims = op.getZeropointDims();
  auto last_dim_rightmost_valid_count = op.getLastDimRightmostValidCount();
  sfr.last_dim_rightmost_valid_count0 =
      dyn_cast_or_null<IntegerAttr>(last_dim_rightmost_valid_count[0]).getInt();
  sfr.last_dim_rightmost_valid_count1 =
      dyn_cast_or_null<IntegerAttr>(last_dim_rightmost_valid_count[1]).getInt();
  sfr.last_dim_rightmost_valid_count2 =
      dyn_cast_or_null<IntegerAttr>(last_dim_rightmost_valid_count[2]).getInt();
  sfr.last_dim_rightmost_valid_count3 =
      dyn_cast_or_null<IntegerAttr>(last_dim_rightmost_valid_count[3]).getInt();
  sfr.last_dim_rightmost_valid_count4 =
      dyn_cast_or_null<IntegerAttr>(last_dim_rightmost_valid_count[4]).getInt();
  sfr.last_dim_rightmost_valid_count5 =
      dyn_cast_or_null<IntegerAttr>(last_dim_rightmost_valid_count[5]).getInt();
  sfr.last_dim_rightmost_valid_count6 =
      dyn_cast_or_null<IntegerAttr>(last_dim_rightmost_valid_count[6]).getInt();
  sfr.last_dim_rightmost_valid_count7 =
      dyn_cast_or_null<IntegerAttr>(last_dim_rightmost_valid_count[7]).getInt();
  sfr.base = op.getBase();
  sfr.fetch_size = op.getFetchSize();
  auto limits = op.getLimits();
  sfr.limits_element0 = dyn_cast_or_null<IntegerAttr>(limits[0]).getInt();
  sfr.limits_element1 = dyn_cast_or_null<IntegerAttr>(limits[1]).getInt();
  sfr.limits_element2 = dyn_cast_or_null<IntegerAttr>(limits[2]).getInt();
  sfr.limits_element3 = dyn_cast_or_null<IntegerAttr>(limits[3]).getInt();
  sfr.limits_element4 = dyn_cast_or_null<IntegerAttr>(limits[4]).getInt();
  sfr.limits_element5 = dyn_cast_or_null<IntegerAttr>(limits[5]).getInt();
  sfr.limits_element6 = dyn_cast_or_null<IntegerAttr>(limits[6]).getInt();
  sfr.limits_element7 = dyn_cast_or_null<IntegerAttr>(limits[7]).getInt();
  auto strides = op.getStrides();
  sfr.strides_element0 = dyn_cast_or_null<IntegerAttr>(strides[0]).getInt();
  sfr.strides_element1 = dyn_cast_or_null<IntegerAttr>(strides[1]).getInt();
  sfr.strides_element2 = dyn_cast_or_null<IntegerAttr>(strides[2]).getInt();
  sfr.strides_element3 = dyn_cast_or_null<IntegerAttr>(strides[3]).getInt();
  sfr.strides_element4 = dyn_cast_or_null<IntegerAttr>(strides[4]).getInt();
  sfr.strides_element5 = dyn_cast_or_null<IntegerAttr>(strides[5]).getInt();
  sfr.strides_element6 = dyn_cast_or_null<IntegerAttr>(strides[6]).getInt();
  sfr.strides_element7 = dyn_cast_or_null<IntegerAttr>(strides[7]).getInt();
  sfr.flit_count = op.getFlitCount();
  sfr.words_per_packet = op.getWordsPerPacket();
  sfr.zeropoint_fetch_limit = op.getZeropointFetchLimit();
  sfr.topology = op.getTopology();
  sfr.channel_config = op.getChannelConfig();
  sfr.outer_slice_log_size = op.getOuterSliceLogSize();
  sfr.outer_dim0_log_size = op.getOuterDim0LogSize();
  sfr.outer_dim1_log_size = op.getOuterDim1LogSize();
  sfr.outer_dim0_chunk_size = op.getOuterDim0ChunkSize();
  sfr.outer_dim1_chunk_size = op.getOuterDim1ChunkSize();
  auto custom_snoop_bitmap_mask = op.getCustomSnoopBitmapMask();
  sfr.custom_snoop_bitmap_mask_element0 =
      dyn_cast_or_null<IntegerAttr>(custom_snoop_bitmap_mask[0]).getInt();
  sfr.custom_snoop_bitmap_mask_element1 =
      dyn_cast_or_null<IntegerAttr>(custom_snoop_bitmap_mask[1]).getInt();
  sfr.custom_snoop_bitmap_mask_element2 =
      dyn_cast_or_null<IntegerAttr>(custom_snoop_bitmap_mask[2]).getInt();
  sfr.custom_snoop_bitmap_mask_element3 =
      dyn_cast_or_null<IntegerAttr>(custom_snoop_bitmap_mask[3]).getInt();

  return sfr.get_blocks();
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, sfr::StaticSfrRegisterConfigUnitOp> ||
                         std::is_same_v<T, sfr::SfrRegisterConfigUnitOp>,
                     bool> = true>
std::vector<sfr_data_t> getSfrRegisterConfigUnit(T &op) {
  sfr::slice::RegisterConfig<sfr_data_t> sfr{};
  sfr.base = op.getBase();
  sfr.size = op.getSize();
  sfr.access_type = op.getAccessType();
  sfr.words_per_input = op.getWordsPerInput();
  sfr.data_offset = op.getDataOffset();

  return sfr.get_blocks();
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, sfr::StaticSfrSubCommitUnitOp> ||
                               std::is_same_v<T, sfr::SfrSubCommitUnitOp>,
                           bool> = true>
std::vector<sfr_data_t> getSfrSubCommitUnit(T &op) {
  sfr::slice::SubCommitUnit<sfr_data_t> sfr{};
  sfr.mode = op.getMode();
  sfr.packet_valid_count = op.getPacketValidCount();
  sfr.base = op.getBase();
  sfr.commit_in_size = op.getCommitInSize();
  sfr.sub_commit_unit_commit_data = op.getCommitData();
  auto limits = op.getLimits();
  sfr.limits_element0 = dyn_cast_or_null<IntegerAttr>(limits[0]).getInt();
  sfr.limits_element1 = dyn_cast_or_null<IntegerAttr>(limits[1]).getInt();
  sfr.limits_element2 = dyn_cast_or_null<IntegerAttr>(limits[2]).getInt();
  sfr.limits_element3 = dyn_cast_or_null<IntegerAttr>(limits[3]).getInt();
  sfr.limits_element4 = dyn_cast_or_null<IntegerAttr>(limits[4]).getInt();
  sfr.limits_element5 = dyn_cast_or_null<IntegerAttr>(limits[5]).getInt();
  sfr.limits_element6 = dyn_cast_or_null<IntegerAttr>(limits[6]).getInt();
  sfr.limits_element7 = dyn_cast_or_null<IntegerAttr>(limits[7]).getInt();
  auto strides = op.getStrides();
  sfr.strides_element0 = dyn_cast_or_null<IntegerAttr>(strides[0]).getInt();
  sfr.strides_element1 = dyn_cast_or_null<IntegerAttr>(strides[1]).getInt();
  sfr.strides_element2 = dyn_cast_or_null<IntegerAttr>(strides[2]).getInt();
  sfr.strides_element3 = dyn_cast_or_null<IntegerAttr>(strides[3]).getInt();
  sfr.strides_element4 = dyn_cast_or_null<IntegerAttr>(strides[4]).getInt();
  sfr.strides_element5 = dyn_cast_or_null<IntegerAttr>(strides[5]).getInt();
  sfr.strides_element6 = dyn_cast_or_null<IntegerAttr>(strides[6]).getInt();
  sfr.strides_element7 = dyn_cast_or_null<IntegerAttr>(strides[7]).getInt();
  auto slice_enable_bitmap_mask = op.getSliceEnableBitmapMask();
  sfr.sub_commit_unit_slice_enable_bitmap0 =
      dyn_cast_or_null<IntegerAttr>(slice_enable_bitmap_mask[0]).getInt();
  sfr.sub_commit_unit_slice_enable_bitmap1 =
      dyn_cast_or_null<IntegerAttr>(slice_enable_bitmap_mask[1]).getInt();
  sfr.sub_commit_unit_slice_enable_bitmap2 =
      dyn_cast_or_null<IntegerAttr>(slice_enable_bitmap_mask[2]).getInt();
  sfr.sub_commit_unit_slice_enable_bitmap3 =
      dyn_cast_or_null<IntegerAttr>(slice_enable_bitmap_mask[3]).getInt();

  return sfr.get_blocks();
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, sfr::StaticSfrSubDataPathUnitOp> ||
                               std::is_same_v<T, sfr::SfrSubDataPathUnitOp>,
                           bool> = true>
std::vector<sfr_data_t> getSfrSubDataPathUnit(T &op) {
  sfr::slice::OperationDataPath<sfr_data_t> sfr =
      sfr::slice::OperationDataPath<sfr_data_t>();
  sfr.data_path_route_sub_context = op.getDataPathRouteSubContext();

  return sfr.get_blocks();
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, sfr::StaticSfrSubFetchUnitOp> ||
                               std::is_same_v<T, sfr::SfrSubFetchUnitOp>,
                           bool> = true>
std::vector<sfr_data_t> getSfrSubFetchUnit(T &op) {
  sfr::slice::SubFetchUnit<sfr_data_t> sfr{};
  sfr.base = op.getBase();
  sfr.type_conversion = op.getTypeConversion();
  sfr.num_zero_points = op.getNumZeroPoints();
  sfr.zero_point0 = op.getZeroPoint0();
  sfr.zero_point1 = op.getZeroPoint1();
  auto limits = op.getLimits();
  sfr.limits_element0 = dyn_cast_or_null<IntegerAttr>(limits[0]).getInt();
  sfr.limits_element1 = dyn_cast_or_null<IntegerAttr>(limits[1]).getInt();
  sfr.limits_element2 = dyn_cast_or_null<IntegerAttr>(limits[2]).getInt();
  sfr.limits_element3 = dyn_cast_or_null<IntegerAttr>(limits[3]).getInt();
  sfr.limits_element4 = dyn_cast_or_null<IntegerAttr>(limits[4]).getInt();
  sfr.limits_element5 = dyn_cast_or_null<IntegerAttr>(limits[5]).getInt();
  sfr.limits_element6 = dyn_cast_or_null<IntegerAttr>(limits[6]).getInt();
  sfr.limits_element7 = dyn_cast_or_null<IntegerAttr>(limits[7]).getInt();
  auto strides = op.getStrides();
  sfr.strides_element0 = dyn_cast_or_null<IntegerAttr>(strides[0]).getInt();
  sfr.strides_element1 = dyn_cast_or_null<IntegerAttr>(strides[1]).getInt();
  sfr.strides_element2 = dyn_cast_or_null<IntegerAttr>(strides[2]).getInt();
  sfr.strides_element3 = dyn_cast_or_null<IntegerAttr>(strides[3]).getInt();
  sfr.strides_element4 = dyn_cast_or_null<IntegerAttr>(strides[4]).getInt();
  sfr.strides_element5 = dyn_cast_or_null<IntegerAttr>(strides[5]).getInt();
  sfr.strides_element6 = dyn_cast_or_null<IntegerAttr>(strides[6]).getInt();
  sfr.strides_element7 = dyn_cast_or_null<IntegerAttr>(strides[7]).getInt();
  sfr.flit_count = op.getFlitCount();
  sfr.words_per_packet = op.getWordsPerPacket();
  sfr.topology = op.getTopology();
  sfr.outer_slice_log_size = op.getOuterSliceLogSize();
  sfr.outer_dim0_log_size = op.getOuterDim0LogSize();
  sfr.outer_dim1_log_size = op.getOuterDim1LogSize();
  sfr.outer_dim0_chunk_size = op.getOuterDim0ChunkSize();
  sfr.outer_dim1_chunk_size = op.getOuterDim1ChunkSize();
  auto custom_snoop_bitmap_mask = op.getCustomSnoopBitmapMask();
  sfr.sub_fetch_unit_custom_snoop_bitmap0 =
      dyn_cast_or_null<IntegerAttr>(custom_snoop_bitmap_mask[0]).getInt();
  sfr.sub_fetch_unit_custom_snoop_bitmap1 =
      dyn_cast_or_null<IntegerAttr>(custom_snoop_bitmap_mask[1]).getInt();
  sfr.sub_fetch_unit_custom_snoop_bitmap2 =
      dyn_cast_or_null<IntegerAttr>(custom_snoop_bitmap_mask[2]).getInt();
  sfr.sub_fetch_unit_custom_snoop_bitmap3 =
      dyn_cast_or_null<IntegerAttr>(custom_snoop_bitmap_mask[3]).getInt();

  return sfr.get_blocks();
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, sfr::StaticSfrTensorRegisterFileOp> ||
                         std::is_same_v<T, sfr::SfrTensorRegisterFileOp>,
                     bool> = true>
std::vector<sfr_data_t> getSfrTensorRegisterFile(T &op) {
  sfr::slice::DotProductEngineRegisterFile<sfr_data_t> sfr{};
  sfr.write_interleaving_flit_count = op.getWriteInterleavingFlitCount();
  sfr.write_mode = op.getWriteMode();
  sfr.write_mac_rows = op.getWriteMacRows();
  sfr.write_skip_flit_count = op.getWriteSkipFlitCount();
  sfr.write_row_base = op.getWriteRowBase();
  sfr.write_mac_row_interleaving = op.getWriteMacRowInterleaving();
  sfr.write_row_count = op.getWriteRowCount();
  sfr.write_flits_per_period = op.getWriteFlitsPerPeriod();
  sfr.write_valid_flits_per_period = op.getWriteValidFlitsPerPeriod();

  return sfr.get_blocks();
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, sfr::StaticSfrTransposeEngineOp> ||
                               std::is_same_v<T, sfr::SfrTransposeEngineOp>,
                           bool> = true>
std::vector<sfr_data_t> getSfrTransposeEngine(T &op) {
  sfr::slice::TransposeEngineMainContext<sfr_data_t> sfr{};
  sfr.fetch_in_cols = op.getFetchInCols();
  sfr.fetch_in_rows = op.getFetchInRows();
  sfr.fetch_out_rows = op.getFetchOutRows();
  sfr.data_type = op.getDataType();
  sfr.fetch_in_width_shift = op.getFetchInWidthShift();

  return sfr.get_blocks();
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, sfr::StaticSfrVectorArithmeticUnitOp> ||
                         std::is_same_v<T, sfr::SfrVectorArithmeticUnitOp>,
                     bool> = true>
std::vector<sfr_data_t> getSfrVectorArithmeticUnit(T &op) {
  sfr::slice::VectorArithmeticUnitMainContext<sfr_data_t> sfr{};
  sfr.branch_mode_mode = op.getBranchModeMode();
  sfr.branch_mode_format = op.getBranchModeFormat();
  sfr.branch_mode_compare_operation0 = op.getBranchModeCompareOperation0();
  sfr.branch_mode_compare_operation1 = op.getBranchModeCompareOperation1();
  sfr.branch_mode_compare_operation2 = op.getBranchModeCompareOperation2();
  sfr.branch_mode_compare_operation3 = op.getBranchModeCompareOperation3();
  sfr.branch_mode_group_size = op.getBranchModeGroupSize();
  sfr.branch_mode_branch_read_base = op.getBranchModeBranchReadBase();
  sfr.branch_mode_branch_read_limit = op.getBranchModeBranchReadLimit();
  sfr.branch_data0_scalar_register_element0 =
      op.getBranchData0ScalarRegisterElement0();
  sfr.branch_data0_scalar_register_element1 =
      op.getBranchData0ScalarRegisterElement1();
  sfr.branch_data1_scalar_register_element2 =
      op.getBranchData1ScalarRegisterElement2();
  sfr.branch_data1_scalar_register_element3 =
      op.getBranchData1ScalarRegisterElement3();
  sfr.register_file_write_mode_branch_write_mode =
      op.getRegisterFileWriteModeBranchWriteMode();
  sfr.register_file_write_mode_branch_write_base =
      op.getRegisterFileWriteModeBranchWriteBase();
  sfr.register_file_write_mode_branch_write_limit =
      op.getRegisterFileWriteModeBranchWriteLimit();
  sfr.register_file_write_mode_write_cmp_op =
      op.getRegisterFileWriteModeWriteCmpOp();
  sfr.register_file_write_mode_write_execution_id =
      op.getRegisterFileWriteModeWriteExecutionId();
  sfr.register_file_write_mode_write_execution_id_mask =
      op.getRegisterFileWriteModeWriteExecutionIdMask();
  sfr.logic_cluster_route_logic_and_source =
      op.getLogicClusterRouteLogicAndSource();
  sfr.logic_cluster_route_logic_or_source =
      op.getLogicClusterRouteLogicOrSource();
  sfr.logic_cluster_route_logic_xor_source =
      op.getLogicClusterRouteLogicXorSource();
  sfr.logic_cluster_route_logic_left_shift_source =
      op.getLogicClusterRouteLogicLeftShiftSource();
  sfr.logic_cluster_route_logic_right_shift_source =
      op.getLogicClusterRouteLogicRightShiftSource();
  sfr.logic_cluster_route_logic_cluster_source =
      op.getLogicClusterRouteLogicClusterSource();
  sfr.logic_and_control_op_mode = op.getLogicAndControlOpMode();
  sfr.logic_and_control_arg_mode = op.getLogicAndControlArgMode();
  sfr.logic_and_control_reg0_cmp_op = op.getLogicAndControlReg0CmpOp();
  sfr.logic_and_control_reg1_cmp_op = op.getLogicAndControlReg1CmpOp();
  sfr.logic_and_control_reg2_cmp_op = op.getLogicAndControlReg2CmpOp();
  sfr.logic_and_control_rf_cmp_op = op.getLogicAndControlRfCmpOp();
  sfr.logic_and_control_reg0_execution_id =
      op.getLogicAndControlReg0ExecutionId();
  sfr.logic_and_control_reg0_execution_id_mask =
      op.getLogicAndControlReg0ExecutionIdMask();
  sfr.logic_and_control_reg1_execution_id =
      op.getLogicAndControlReg1ExecutionId();
  sfr.logic_and_control_reg1_execution_id_mask =
      op.getLogicAndControlReg1ExecutionIdMask();
  sfr.logic_and_control_reg2_execution_id =
      op.getLogicAndControlReg2ExecutionId();
  sfr.logic_and_control_reg2_execution_id_mask =
      op.getLogicAndControlReg2ExecutionIdMask();
  sfr.logic_and_control_rf_execution_id = op.getLogicAndControlRfExecutionId();
  sfr.logic_and_control_rf_execution_id_mask =
      op.getLogicAndControlRfExecutionIdMask();
  sfr.logic_and_data0_scalar_register_element0 =
      op.getLogicAndData0ScalarRegisterElement0();
  sfr.logic_and_data0_scalar_register_element1 =
      op.getLogicAndData0ScalarRegisterElement1();
  sfr.logic_and_data1_scalar_register_element2 =
      op.getLogicAndData1ScalarRegisterElement2();
  sfr.logic_or_control_op_mode = op.getLogicOrControlOpMode();
  sfr.logic_or_control_arg_mode = op.getLogicOrControlArgMode();
  sfr.logic_or_control_reg0_cmp_op = op.getLogicOrControlReg0CmpOp();
  sfr.logic_or_control_reg1_cmp_op = op.getLogicOrControlReg1CmpOp();
  sfr.logic_or_control_reg2_cmp_op = op.getLogicOrControlReg2CmpOp();
  sfr.logic_or_control_rf_cmp_op = op.getLogicOrControlRfCmpOp();
  sfr.logic_or_control_reg0_execution_id =
      op.getLogicOrControlReg0ExecutionId();
  sfr.logic_or_control_reg0_execution_id_mask =
      op.getLogicOrControlReg0ExecutionIdMask();
  sfr.logic_or_control_reg1_execution_id =
      op.getLogicOrControlReg1ExecutionId();
  sfr.logic_or_control_reg1_execution_id_mask =
      op.getLogicOrControlReg1ExecutionIdMask();
  sfr.logic_or_control_reg2_execution_id =
      op.getLogicOrControlReg2ExecutionId();
  sfr.logic_or_control_reg2_execution_id_mask =
      op.getLogicOrControlReg2ExecutionIdMask();
  sfr.logic_or_control_rf_execution_id = op.getLogicOrControlRfExecutionId();
  sfr.logic_or_control_rf_execution_id_mask =
      op.getLogicOrControlRfExecutionIdMask();
  sfr.logic_or_data0_scalar_register_element0 =
      op.getLogicOrData0ScalarRegisterElement0();
  sfr.logic_or_data0_scalar_register_element1 =
      op.getLogicOrData0ScalarRegisterElement1();
  sfr.logic_or_data1_scalar_register_element2 =
      op.getLogicOrData1ScalarRegisterElement2();
  sfr.logic_xor_control_op_mode = op.getLogicXorControlOpMode();
  sfr.logic_xor_control_arg_mode = op.getLogicXorControlArgMode();
  sfr.logic_xor_control_reg0_cmp_op = op.getLogicXorControlReg0CmpOp();
  sfr.logic_xor_control_reg1_cmp_op = op.getLogicXorControlReg1CmpOp();
  sfr.logic_xor_control_reg2_cmp_op = op.getLogicXorControlReg2CmpOp();
  sfr.logic_xor_control_rf_cmp_op = op.getLogicXorControlRfCmpOp();
  sfr.logic_xor_control_reg0_execution_id =
      op.getLogicXorControlReg0ExecutionId();
  sfr.logic_xor_control_reg0_execution_id_mask =
      op.getLogicXorControlReg0ExecutionIdMask();
  sfr.logic_xor_control_reg1_execution_id =
      op.getLogicXorControlReg1ExecutionId();
  sfr.logic_xor_control_reg1_execution_id_mask =
      op.getLogicXorControlReg1ExecutionIdMask();
  sfr.logic_xor_control_reg2_execution_id =
      op.getLogicXorControlReg2ExecutionId();
  sfr.logic_xor_control_reg2_execution_id_mask =
      op.getLogicXorControlReg2ExecutionIdMask();
  sfr.logic_xor_control_rf_execution_id = op.getLogicXorControlRfExecutionId();
  sfr.logic_xor_control_rf_execution_id_mask =
      op.getLogicXorControlRfExecutionIdMask();
  sfr.logic_xor_data0_scalar_register_element0 =
      op.getLogicXorData0ScalarRegisterElement0();
  sfr.logic_xor_data0_scalar_register_element1 =
      op.getLogicXorData0ScalarRegisterElement1();
  sfr.logic_xor_data1_scalar_register_element2 =
      op.getLogicXorData1ScalarRegisterElement2();
  sfr.logic_left_shift_control_op_mode = op.getLogicLeftShiftControlOpMode();
  sfr.logic_left_shift_control_arg_mode = op.getLogicLeftShiftControlArgMode();
  sfr.logic_left_shift_control_reg0_cmp_op =
      op.getLogicLeftShiftControlReg0CmpOp();
  sfr.logic_left_shift_control_reg1_cmp_op =
      op.getLogicLeftShiftControlReg1CmpOp();
  sfr.logic_left_shift_control_reg2_cmp_op =
      op.getLogicLeftShiftControlReg2CmpOp();
  sfr.logic_left_shift_control_rf_cmp_op = op.getLogicLeftShiftControlRfCmpOp();
  sfr.logic_left_shift_control_reg0_execution_id =
      op.getLogicLeftShiftControlReg0ExecutionId();
  sfr.logic_left_shift_control_reg0_execution_id_mask =
      op.getLogicLeftShiftControlReg0ExecutionIdMask();
  sfr.logic_left_shift_control_reg1_execution_id =
      op.getLogicLeftShiftControlReg1ExecutionId();
  sfr.logic_left_shift_control_reg1_execution_id_mask =
      op.getLogicLeftShiftControlReg1ExecutionIdMask();
  sfr.logic_left_shift_control_reg2_execution_id =
      op.getLogicLeftShiftControlReg2ExecutionId();
  sfr.logic_left_shift_control_reg2_execution_id_mask =
      op.getLogicLeftShiftControlReg2ExecutionIdMask();
  sfr.logic_left_shift_control_rf_execution_id =
      op.getLogicLeftShiftControlRfExecutionId();
  sfr.logic_left_shift_control_rf_execution_id_mask =
      op.getLogicLeftShiftControlRfExecutionIdMask();
  sfr.logic_left_shift_data0_scalar_register_element0 =
      op.getLogicLeftShiftData0ScalarRegisterElement0();
  sfr.logic_left_shift_data0_scalar_register_element1 =
      op.getLogicLeftShiftData0ScalarRegisterElement1();
  sfr.logic_left_shift_data1_scalar_register_element2 =
      op.getLogicLeftShiftData1ScalarRegisterElement2();
  sfr.logic_right_shift_control_op_mode = op.getLogicRightShiftControlOpMode();
  sfr.logic_right_shift_control_arg_mode =
      op.getLogicRightShiftControlArgMode();
  sfr.logic_right_shift_control_reg0_cmp_op =
      op.getLogicRightShiftControlReg0CmpOp();
  sfr.logic_right_shift_control_reg1_cmp_op =
      op.getLogicRightShiftControlReg1CmpOp();
  sfr.logic_right_shift_control_reg2_cmp_op =
      op.getLogicRightShiftControlReg2CmpOp();
  sfr.logic_right_shift_control_rf_cmp_op =
      op.getLogicRightShiftControlRfCmpOp();
  sfr.logic_right_shift_control_reg0_execution_id =
      op.getLogicRightShiftControlReg0ExecutionId();
  sfr.logic_right_shift_control_reg0_execution_id_mask =
      op.getLogicRightShiftControlReg0ExecutionIdMask();
  sfr.logic_right_shift_control_reg1_execution_id =
      op.getLogicRightShiftControlReg1ExecutionId();
  sfr.logic_right_shift_control_reg1_execution_id_mask =
      op.getLogicRightShiftControlReg1ExecutionIdMask();
  sfr.logic_right_shift_control_reg2_execution_id =
      op.getLogicRightShiftControlReg2ExecutionId();
  sfr.logic_right_shift_control_reg2_execution_id_mask =
      op.getLogicRightShiftControlReg2ExecutionIdMask();
  sfr.logic_right_shift_control_rf_execution_id =
      op.getLogicRightShiftControlRfExecutionId();
  sfr.logic_right_shift_control_rf_execution_id_mask =
      op.getLogicRightShiftControlRfExecutionIdMask();
  sfr.logic_right_shift_data0_scalar_register_element0 =
      op.getLogicRightShiftData0ScalarRegisterElement0();
  sfr.logic_right_shift_data0_scalar_register_element1 =
      op.getLogicRightShiftData0ScalarRegisterElement1();
  sfr.logic_right_shift_data1_scalar_register_element2 =
      op.getLogicRightShiftData1ScalarRegisterElement2();
  sfr.fxp_cluster_route_fxp_add_source = op.getFxpClusterRouteFxpAddSource();
  sfr.fxp_cluster_route_fxp_left_shift_source =
      op.getFxpClusterRouteFxpLeftShiftSource();
  sfr.fxp_cluster_route_fxp_mul_source = op.getFxpClusterRouteFxpMulSource();
  sfr.fxp_cluster_route_fxp_right_shift_source =
      op.getFxpClusterRouteFxpRightShiftSource();
  sfr.fxp_cluster_route_fxp_cluster_source =
      op.getFxpClusterRouteFxpClusterSource();
  sfr.fxp_add_control_op_mode = op.getFxpAddControlOpMode();
  sfr.fxp_add_control_arg_mode = op.getFxpAddControlArgMode();
  sfr.fxp_add_control_reg0_cmp_op = op.getFxpAddControlReg0CmpOp();
  sfr.fxp_add_control_reg1_cmp_op = op.getFxpAddControlReg1CmpOp();
  sfr.fxp_add_control_reg2_cmp_op = op.getFxpAddControlReg2CmpOp();
  sfr.fxp_add_control_rf_cmp_op = op.getFxpAddControlRfCmpOp();
  sfr.fxp_add_control_reg0_execution_id = op.getFxpAddControlReg0ExecutionId();
  sfr.fxp_add_control_reg0_execution_id_mask =
      op.getFxpAddControlReg0ExecutionIdMask();
  sfr.fxp_add_control_reg1_execution_id = op.getFxpAddControlReg1ExecutionId();
  sfr.fxp_add_control_reg1_execution_id_mask =
      op.getFxpAddControlReg1ExecutionIdMask();
  sfr.fxp_add_control_reg2_execution_id = op.getFxpAddControlReg2ExecutionId();
  sfr.fxp_add_control_reg2_execution_id_mask =
      op.getFxpAddControlReg2ExecutionIdMask();
  sfr.fxp_add_control_rf_execution_id = op.getFxpAddControlRfExecutionId();
  sfr.fxp_add_control_rf_execution_id_mask =
      op.getFxpAddControlRfExecutionIdMask();
  sfr.fxp_add_data0_scalar_register_element0 =
      op.getFxpAddData0ScalarRegisterElement0();
  sfr.fxp_add_data0_scalar_register_element1 =
      op.getFxpAddData0ScalarRegisterElement1();
  sfr.fxp_add_data1_scalar_register_element2 =
      op.getFxpAddData1ScalarRegisterElement2();
  sfr.fxp_left_shift_control_op_mode = op.getFxpLeftShiftControlOpMode();
  sfr.fxp_left_shift_control_arg_mode = op.getFxpLeftShiftControlArgMode();
  sfr.fxp_left_shift_control_reg0_cmp_op = op.getFxpLeftShiftControlReg0CmpOp();
  sfr.fxp_left_shift_control_reg1_cmp_op = op.getFxpLeftShiftControlReg1CmpOp();
  sfr.fxp_left_shift_control_reg2_cmp_op = op.getFxpLeftShiftControlReg2CmpOp();
  sfr.fxp_left_shift_control_rf_cmp_op = op.getFxpLeftShiftControlRfCmpOp();
  sfr.fxp_left_shift_control_reg0_execution_id =
      op.getFxpLeftShiftControlReg0ExecutionId();
  sfr.fxp_left_shift_control_reg0_execution_id_mask =
      op.getFxpLeftShiftControlReg0ExecutionIdMask();
  sfr.fxp_left_shift_control_reg1_execution_id =
      op.getFxpLeftShiftControlReg1ExecutionId();
  sfr.fxp_left_shift_control_reg1_execution_id_mask =
      op.getFxpLeftShiftControlReg1ExecutionIdMask();
  sfr.fxp_left_shift_control_reg2_execution_id =
      op.getFxpLeftShiftControlReg2ExecutionId();
  sfr.fxp_left_shift_control_reg2_execution_id_mask =
      op.getFxpLeftShiftControlReg2ExecutionIdMask();
  sfr.fxp_left_shift_control_rf_execution_id =
      op.getFxpLeftShiftControlRfExecutionId();
  sfr.fxp_left_shift_control_rf_execution_id_mask =
      op.getFxpLeftShiftControlRfExecutionIdMask();
  sfr.fxp_left_shift_data0_scalar_register_element0 =
      op.getFxpLeftShiftData0ScalarRegisterElement0();
  sfr.fxp_left_shift_data0_scalar_register_element1 =
      op.getFxpLeftShiftData0ScalarRegisterElement1();
  sfr.fxp_left_shift_data1_scalar_register_element2 =
      op.getFxpLeftShiftData1ScalarRegisterElement2();
  sfr.fxp_mul_control_op_mode = op.getFxpMulControlOpMode();
  sfr.fxp_mul_control_arg_mode = op.getFxpMulControlArgMode();
  sfr.fxp_mul_control_reg0_cmp_op = op.getFxpMulControlReg0CmpOp();
  sfr.fxp_mul_control_reg1_cmp_op = op.getFxpMulControlReg1CmpOp();
  sfr.fxp_mul_control_reg2_cmp_op = op.getFxpMulControlReg2CmpOp();
  sfr.fxp_mul_control_rf_cmp_op = op.getFxpMulControlRfCmpOp();
  sfr.fxp_mul_control_reg0_execution_id = op.getFxpMulControlReg0ExecutionId();
  sfr.fxp_mul_control_reg0_execution_id_mask =
      op.getFxpMulControlReg0ExecutionIdMask();
  sfr.fxp_mul_control_reg1_execution_id = op.getFxpMulControlReg1ExecutionId();
  sfr.fxp_mul_control_reg1_execution_id_mask =
      op.getFxpMulControlReg1ExecutionIdMask();
  sfr.fxp_mul_control_reg2_execution_id = op.getFxpMulControlReg2ExecutionId();
  sfr.fxp_mul_control_reg2_execution_id_mask =
      op.getFxpMulControlReg2ExecutionIdMask();
  sfr.fxp_mul_control_rf_execution_id = op.getFxpMulControlRfExecutionId();
  sfr.fxp_mul_control_rf_execution_id_mask =
      op.getFxpMulControlRfExecutionIdMask();
  sfr.fxp_mul_data0_scalar_register_element0 =
      op.getFxpMulData0ScalarRegisterElement0();
  sfr.fxp_mul_data0_scalar_register_element1 =
      op.getFxpMulData0ScalarRegisterElement1();
  sfr.fxp_mul_data1_scalar_register_element2 =
      op.getFxpMulData1ScalarRegisterElement2();
  sfr.fxp_right_shift_control_op_mode = op.getFxpRightShiftControlOpMode();
  sfr.fxp_right_shift_control_arg_mode = op.getFxpRightShiftControlArgMode();
  sfr.fxp_right_shift_control_reg0_cmp_op =
      op.getFxpRightShiftControlReg0CmpOp();
  sfr.fxp_right_shift_control_reg1_cmp_op =
      op.getFxpRightShiftControlReg1CmpOp();
  sfr.fxp_right_shift_control_reg2_cmp_op =
      op.getFxpRightShiftControlReg2CmpOp();
  sfr.fxp_right_shift_control_rf_cmp_op = op.getFxpRightShiftControlRfCmpOp();
  sfr.fxp_right_shift_control_reg0_execution_id =
      op.getFxpRightShiftControlReg0ExecutionId();
  sfr.fxp_right_shift_control_reg0_execution_id_mask =
      op.getFxpRightShiftControlReg0ExecutionIdMask();
  sfr.fxp_right_shift_control_reg1_execution_id =
      op.getFxpRightShiftControlReg1ExecutionId();
  sfr.fxp_right_shift_control_reg1_execution_id_mask =
      op.getFxpRightShiftControlReg1ExecutionIdMask();
  sfr.fxp_right_shift_control_reg2_execution_id =
      op.getFxpRightShiftControlReg2ExecutionId();
  sfr.fxp_right_shift_control_reg2_execution_id_mask =
      op.getFxpRightShiftControlReg2ExecutionIdMask();
  sfr.fxp_right_shift_control_rf_execution_id =
      op.getFxpRightShiftControlRfExecutionId();
  sfr.fxp_right_shift_control_rf_execution_id_mask =
      op.getFxpRightShiftControlRfExecutionIdMask();
  sfr.fxp_right_shift_data0_scalar_register_element0 =
      op.getFxpRightShiftData0ScalarRegisterElement0();
  sfr.fxp_right_shift_data0_scalar_register_element1 =
      op.getFxpRightShiftData0ScalarRegisterElement1();
  sfr.fxp_right_shift_data1_scalar_register_element2 =
      op.getFxpRightShiftData1ScalarRegisterElement2();
  sfr.fp_cluster_route_fp_fma_source = op.getFpClusterRouteFpFmaSource();
  sfr.fp_cluster_route_fp_fpu_source = op.getFpClusterRouteFpFpuSource();
  sfr.fp_cluster_route_fp_exp_source = op.getFpClusterRouteFpExpSource();
  sfr.fp_cluster_route_fp_mul0_source = op.getFpClusterRouteFpMul0Source();
  sfr.fp_cluster_route_fp_mul1_source = op.getFpClusterRouteFpMul1Source();
  sfr.fp_cluster_route_fp_cluster_source =
      op.getFpClusterRouteFpClusterSource();
  sfr.fp_fma_control_op_mode = op.getFpFmaControlOpMode();
  sfr.fp_fma_control_arg_mode = op.getFpFmaControlArgMode();
  sfr.fp_fma_control_reg0_cmp_op = op.getFpFmaControlReg0CmpOp();
  sfr.fp_fma_control_reg1_cmp_op = op.getFpFmaControlReg1CmpOp();
  sfr.fp_fma_control_reg2_cmp_op = op.getFpFmaControlReg2CmpOp();
  sfr.fp_fma_control_rf_cmp_op = op.getFpFmaControlRfCmpOp();
  sfr.fp_fma_control_reg0_execution_id = op.getFpFmaControlReg0ExecutionId();
  sfr.fp_fma_control_reg0_execution_id_mask =
      op.getFpFmaControlReg0ExecutionIdMask();
  sfr.fp_fma_control_reg1_execution_id = op.getFpFmaControlReg1ExecutionId();
  sfr.fp_fma_control_reg1_execution_id_mask =
      op.getFpFmaControlReg1ExecutionIdMask();
  sfr.fp_fma_control_reg2_execution_id = op.getFpFmaControlReg2ExecutionId();
  sfr.fp_fma_control_reg2_execution_id_mask =
      op.getFpFmaControlReg2ExecutionIdMask();
  sfr.fp_fma_control_rf_execution_id = op.getFpFmaControlRfExecutionId();
  sfr.fp_fma_control_rf_execution_id_mask =
      op.getFpFmaControlRfExecutionIdMask();
  sfr.fp_fma_data0_scalar_register_element0 =
      op.getFpFmaData0ScalarRegisterElement0();
  sfr.fp_fma_data0_scalar_register_element1 =
      op.getFpFmaData0ScalarRegisterElement1();
  sfr.fp_fma_data1_scalar_register_element2 =
      op.getFpFmaData1ScalarRegisterElement2();
  sfr.fp_fma_data1_secondary_scalar_register_element0 =
      op.getFpFmaData1SecondaryScalarRegisterElement0();
  sfr.fp_fma_data2_secondary_scalar_register_element1 =
      op.getFpFmaData2SecondaryScalarRegisterElement1();
  sfr.fp_fma_data2_secondary_scalar_register_element2 =
      op.getFpFmaData2SecondaryScalarRegisterElement2();
  sfr.fp_fpu_control_op_mode = op.getFpFpuControlOpMode();
  sfr.fp_fpu_control_arg_mode = op.getFpFpuControlArgMode();
  sfr.fp_fpu_control_reg0_cmp_op = op.getFpFpuControlReg0CmpOp();
  sfr.fp_fpu_control_reg1_cmp_op = op.getFpFpuControlReg1CmpOp();
  sfr.fp_fpu_control_reg2_cmp_op = op.getFpFpuControlReg2CmpOp();
  sfr.fp_fpu_control_rf_cmp_op = op.getFpFpuControlRfCmpOp();
  sfr.fp_fpu_control_reg0_execution_id = op.getFpFpuControlReg0ExecutionId();
  sfr.fp_fpu_control_reg0_execution_id_mask =
      op.getFpFpuControlReg0ExecutionIdMask();
  sfr.fp_fpu_control_reg1_execution_id = op.getFpFpuControlReg1ExecutionId();
  sfr.fp_fpu_control_reg1_execution_id_mask =
      op.getFpFpuControlReg1ExecutionIdMask();
  sfr.fp_fpu_control_reg2_execution_id = op.getFpFpuControlReg2ExecutionId();
  sfr.fp_fpu_control_reg2_execution_id_mask =
      op.getFpFpuControlReg2ExecutionIdMask();
  sfr.fp_fpu_control_rf_execution_id = op.getFpFpuControlRfExecutionId();
  sfr.fp_fpu_control_rf_execution_id_mask =
      op.getFpFpuControlRfExecutionIdMask();
  sfr.fp_fpu_data0_scalar_register_element0 =
      op.getFpFpuData0ScalarRegisterElement0();
  sfr.fp_fpu_data0_scalar_register_element1 =
      op.getFpFpuData0ScalarRegisterElement1();
  sfr.fp_fpu_data1_scalar_register_element2 =
      op.getFpFpuData1ScalarRegisterElement2();
  sfr.fp_exp_control_op_mode = op.getFpExpControlOpMode();
  sfr.fp_exp_control_arg_mode = op.getFpExpControlArgMode();
  sfr.fp_exp_control_reg0_cmp_op = op.getFpExpControlReg0CmpOp();
  sfr.fp_exp_control_reg1_cmp_op = op.getFpExpControlReg1CmpOp();
  sfr.fp_exp_control_reg2_cmp_op = op.getFpExpControlReg2CmpOp();
  sfr.fp_exp_control_rf_cmp_op = op.getFpExpControlRfCmpOp();
  sfr.fp_exp_control_reg0_execution_id = op.getFpExpControlReg0ExecutionId();
  sfr.fp_exp_control_reg0_execution_id_mask =
      op.getFpExpControlReg0ExecutionIdMask();
  sfr.fp_exp_control_reg1_execution_id = op.getFpExpControlReg1ExecutionId();
  sfr.fp_exp_control_reg1_execution_id_mask =
      op.getFpExpControlReg1ExecutionIdMask();
  sfr.fp_exp_control_reg2_execution_id = op.getFpExpControlReg2ExecutionId();
  sfr.fp_exp_control_reg2_execution_id_mask =
      op.getFpExpControlReg2ExecutionIdMask();
  sfr.fp_exp_control_rf_execution_id = op.getFpExpControlRfExecutionId();
  sfr.fp_exp_control_rf_execution_id_mask =
      op.getFpExpControlRfExecutionIdMask();
  sfr.fp_mul0_control_op_mode = op.getFpMul0ControlOpMode();
  sfr.fp_mul0_control_arg_mode = op.getFpMul0ControlArgMode();
  sfr.fp_mul0_control_reg0_cmp_op = op.getFpMul0ControlReg0CmpOp();
  sfr.fp_mul0_control_reg1_cmp_op = op.getFpMul0ControlReg1CmpOp();
  sfr.fp_mul0_control_reg2_cmp_op = op.getFpMul0ControlReg2CmpOp();
  sfr.fp_mul0_control_rf_cmp_op = op.getFpMul0ControlRfCmpOp();
  sfr.fp_mul0_control_reg0_execution_id = op.getFpMul0ControlReg0ExecutionId();
  sfr.fp_mul0_control_reg0_execution_id_mask =
      op.getFpMul0ControlReg0ExecutionIdMask();
  sfr.fp_mul0_control_reg1_execution_id = op.getFpMul0ControlReg1ExecutionId();
  sfr.fp_mul0_control_reg1_execution_id_mask =
      op.getFpMul0ControlReg1ExecutionIdMask();
  sfr.fp_mul0_control_reg2_execution_id = op.getFpMul0ControlReg2ExecutionId();
  sfr.fp_mul0_control_reg2_execution_id_mask =
      op.getFpMul0ControlReg2ExecutionIdMask();
  sfr.fp_mul0_control_rf_execution_id = op.getFpMul0ControlRfExecutionId();
  sfr.fp_mul0_control_rf_execution_id_mask =
      op.getFpMul0ControlRfExecutionIdMask();
  sfr.fp_mul0_data0_scalar_register_element0 =
      op.getFpMul0Data0ScalarRegisterElement0();
  sfr.fp_mul0_data0_scalar_register_element1 =
      op.getFpMul0Data0ScalarRegisterElement1();
  sfr.fp_mul0_data1_scalar_register_element2 =
      op.getFpMul0Data1ScalarRegisterElement2();
  sfr.fp_mul1_control_op_mode = op.getFpMul1ControlOpMode();
  sfr.fp_mul1_control_arg_mode = op.getFpMul1ControlArgMode();
  sfr.fp_mul1_control_reg0_cmp_op = op.getFpMul1ControlReg0CmpOp();
  sfr.fp_mul1_control_reg1_cmp_op = op.getFpMul1ControlReg1CmpOp();
  sfr.fp_mul1_control_reg2_cmp_op = op.getFpMul1ControlReg2CmpOp();
  sfr.fp_mul1_control_rf_cmp_op = op.getFpMul1ControlRfCmpOp();
  sfr.fp_mul1_control_reg0_execution_id = op.getFpMul1ControlReg0ExecutionId();
  sfr.fp_mul1_control_reg0_execution_id_mask =
      op.getFpMul1ControlReg0ExecutionIdMask();
  sfr.fp_mul1_control_reg1_execution_id = op.getFpMul1ControlReg1ExecutionId();
  sfr.fp_mul1_control_reg1_execution_id_mask =
      op.getFpMul1ControlReg1ExecutionIdMask();
  sfr.fp_mul1_control_reg2_execution_id = op.getFpMul1ControlReg2ExecutionId();
  sfr.fp_mul1_control_reg2_execution_id_mask =
      op.getFpMul1ControlReg2ExecutionIdMask();
  sfr.fp_mul1_control_rf_execution_id = op.getFpMul1ControlRfExecutionId();
  sfr.fp_mul1_control_rf_execution_id_mask =
      op.getFpMul1ControlRfExecutionIdMask();
  sfr.fp_mul1_data0_scalar_register_element0 =
      op.getFpMul1Data0ScalarRegisterElement0();
  sfr.fp_mul1_data0_scalar_register_element1 =
      op.getFpMul1Data0ScalarRegisterElement1();
  sfr.fp_mul1_data1_scalar_register_element2 =
      op.getFpMul1Data1ScalarRegisterElement2();
  sfr.reduce_layer_mode_reduce_data_path =
      op.getReduceLayerModeReduceDataPath();
  sfr.reduce_layer_mode_reduce_rows = op.getReduceLayerModeReduceRows();
  sfr.reduce_layer_mode_reduce_tree_depth =
      op.getReduceLayerModeReduceTreeDepth();
  sfr.reduce_layer_mode_acc_mode = op.getReduceLayerModeAccMode();
  sfr.reduce_layer_mode_acc_indexer_proceed =
      op.getReduceLayerModeAccIndexerProceed();
  sfr.reduce_layer_mode_fxp_shift_rounding_mode =
      op.getReduceLayerModeFxpShiftRoundingMode();
  sfr.reduce_layer_mode_reduce_row0_op_mode =
      op.getReduceLayerModeReduceRow0OpMode();
  sfr.reduce_layer_mode_reduce_row1_op_mode =
      op.getReduceLayerModeReduceRow1OpMode();
  sfr.reduce_layer_mode_accumulation_limit =
      op.getReduceLayerModeAccumulationLimit();
  sfr.reduce_layer_mode_acc_indexer_base =
      op.getReduceLayerModeAccIndexerBase();
  sfr.reduce_layer_acc_init_reduce_row0_acc_init =
      op.getReduceLayerAccInitReduceRow0AccInit();
  sfr.reduce_layer_acc_init_reduce_row1_acc_init =
      op.getReduceLayerAccInitReduceRow1AccInit();
  auto reduce_layer_acc_indexer_limits = op.getReduceLayerAccIndexerLimits();
  sfr.reduce_layer_acc_limit0_acc_indexer_limit_element0 =
      dyn_cast_or_null<IntegerAttr>(reduce_layer_acc_indexer_limits[0])
          .getInt();
  sfr.reduce_layer_acc_limit0_acc_indexer_limit_element1 =
      dyn_cast_or_null<IntegerAttr>(reduce_layer_acc_indexer_limits[1])
          .getInt();
  sfr.reduce_layer_acc_limit0_acc_indexer_limit_element2 =
      dyn_cast_or_null<IntegerAttr>(reduce_layer_acc_indexer_limits[2])
          .getInt();
  sfr.reduce_layer_acc_limit0_acc_indexer_limit_element3 =
      dyn_cast_or_null<IntegerAttr>(reduce_layer_acc_indexer_limits[3])
          .getInt();
  sfr.reduce_layer_acc_limit1_acc_indexer_limit_element4 =
      dyn_cast_or_null<IntegerAttr>(reduce_layer_acc_indexer_limits[4])
          .getInt();
  sfr.reduce_layer_acc_limit1_acc_indexer_limit_element5 =
      dyn_cast_or_null<IntegerAttr>(reduce_layer_acc_indexer_limits[5])
          .getInt();
  sfr.reduce_layer_acc_limit1_acc_indexer_limit_element6 =
      dyn_cast_or_null<IntegerAttr>(reduce_layer_acc_indexer_limits[6])
          .getInt();
  sfr.reduce_layer_acc_limit1_acc_indexer_limit_element7 =
      dyn_cast_or_null<IntegerAttr>(reduce_layer_acc_indexer_limits[7])
          .getInt();
  auto reduce_layer_acc_indexer_strides = op.getReduceLayerAccIndexerStrides();
  sfr.reduce_layer_acc_stride_acc_indexer_stride_element0 =
      dyn_cast_or_null<IntegerAttr>(reduce_layer_acc_indexer_strides[0])
          .getInt();
  sfr.reduce_layer_acc_stride_acc_indexer_stride_element1 =
      dyn_cast_or_null<IntegerAttr>(reduce_layer_acc_indexer_strides[1])
          .getInt();
  sfr.reduce_layer_acc_stride_acc_indexer_stride_element2 =
      dyn_cast_or_null<IntegerAttr>(reduce_layer_acc_indexer_strides[2])
          .getInt();
  sfr.reduce_layer_acc_stride_acc_indexer_stride_element3 =
      dyn_cast_or_null<IntegerAttr>(reduce_layer_acc_indexer_strides[3])
          .getInt();
  sfr.reduce_layer_acc_stride_acc_indexer_stride_element4 =
      dyn_cast_or_null<IntegerAttr>(reduce_layer_acc_indexer_strides[4])
          .getInt();
  sfr.reduce_layer_acc_stride_acc_indexer_stride_element5 =
      dyn_cast_or_null<IntegerAttr>(reduce_layer_acc_indexer_strides[5])
          .getInt();
  sfr.reduce_layer_acc_stride_acc_indexer_stride_element6 =
      dyn_cast_or_null<IntegerAttr>(reduce_layer_acc_indexer_strides[6])
          .getInt();
  sfr.reduce_layer_acc_stride_acc_indexer_stride_element7 =
      dyn_cast_or_null<IntegerAttr>(reduce_layer_acc_indexer_strides[7])
          .getInt();
  sfr.fp_div_control_op_mode = op.getFpDivControlOpMode();
  sfr.fp_div_control_arg_mode = op.getFpDivControlArgMode();
  sfr.fp_div_control_reg0_cmp_op = op.getFpDivControlReg0CmpOp();
  sfr.fp_div_control_reg1_cmp_op = op.getFpDivControlReg1CmpOp();
  sfr.fp_div_control_reg2_cmp_op = op.getFpDivControlReg2CmpOp();
  sfr.fp_div_control_rf_cmp_op = op.getFpDivControlRfCmpOp();
  sfr.fp_div_control_acc_cmp_op = op.getFpDivControlAccCmpOp();
  sfr.fp_div_control_reg0_execution_id = op.getFpDivControlReg0ExecutionId();
  sfr.fp_div_control_reg0_execution_id_mask =
      op.getFpDivControlReg0ExecutionIdMask();
  sfr.fp_div_control_reg1_execution_id = op.getFpDivControlReg1ExecutionId();
  sfr.fp_div_control_reg1_execution_id_mask =
      op.getFpDivControlReg1ExecutionIdMask();
  sfr.fp_div_control_reg2_execution_id = op.getFpDivControlReg2ExecutionId();
  sfr.fp_div_control_reg2_execution_id_mask =
      op.getFpDivControlReg2ExecutionIdMask();
  sfr.fp_div_control_rf_execution_id = op.getFpDivControlRfExecutionId();
  sfr.fp_div_control_rf_execution_id_mask =
      op.getFpDivControlRfExecutionIdMask();
  sfr.fp_div_control_acc_execution_id = op.getFpDivControlAccExecutionId();
  sfr.fp_div_control_acc_execution_id_mask =
      op.getFpDivControlAccExecutionIdMask();
  sfr.fp_div_data0_scalar_register_element0 =
      op.getFpDivData0ScalarRegisterElement0();
  sfr.fp_div_data0_scalar_register_element1 =
      op.getFpDivData0ScalarRegisterElement1();
  sfr.fp_div_data1_scalar_register_element2 =
      op.getFpDivData1ScalarRegisterElement2();
  sfr.float_adapter_fxp_to_fp_mode = op.getFloatAdapterFxpToFpMode();
  sfr.float_adapter_fxp_to_fp_int_width = op.getFloatAdapterFxpToFpIntWidth();
  sfr.float_adapter_fxp_to_fp_round_mode = op.getFloatAdapterFxpToFpRoundMode();
  sfr.float_adapter_fp_to_fxp_mode = op.getFloatAdapterFpToFxpMode();
  sfr.float_adapter_fp_to_fxp_int_width = op.getFloatAdapterFpToFxpIntWidth();
  sfr.float_adapter_fp_to_fxp_round_mode = op.getFloatAdapterFpToFxpRoundMode();
  sfr.float_adapter_split_layer_mode = op.getFloatAdapterSplitLayerMode();
  sfr.float_adapter_concat_layer_mode = op.getFloatAdapterConcatLayerMode();
  sfr.clip_cluster_route_clip_add_source =
      op.getClipClusterRouteClipAddSource();
  sfr.clip_cluster_route_clip_max_source =
      op.getClipClusterRouteClipMaxSource();
  sfr.clip_cluster_route_clip_min_source =
      op.getClipClusterRouteClipMinSource();
  sfr.clip_cluster_route_clip_cluster_source =
      op.getClipClusterRouteClipClusterSource();
  sfr.clip_add_control_op_mode = op.getClipAddControlOpMode();
  sfr.clip_add_control_arg_mode = op.getClipAddControlArgMode();
  sfr.clip_add_control_reg0_cmp_op = op.getClipAddControlReg0CmpOp();
  sfr.clip_add_control_reg1_cmp_op = op.getClipAddControlReg1CmpOp();
  sfr.clip_add_control_reg2_cmp_op = op.getClipAddControlReg2CmpOp();
  sfr.clip_add_control_rf_cmp_op = op.getClipAddControlRfCmpOp();
  sfr.clip_add_control_reg0_execution_id =
      op.getClipAddControlReg0ExecutionId();
  sfr.clip_add_control_reg0_execution_id_mask =
      op.getClipAddControlReg0ExecutionIdMask();
  sfr.clip_add_control_reg1_execution_id =
      op.getClipAddControlReg1ExecutionId();
  sfr.clip_add_control_reg1_execution_id_mask =
      op.getClipAddControlReg1ExecutionIdMask();
  sfr.clip_add_control_reg2_execution_id =
      op.getClipAddControlReg2ExecutionId();
  sfr.clip_add_control_reg2_execution_id_mask =
      op.getClipAddControlReg2ExecutionIdMask();
  sfr.clip_add_control_rf_execution_id = op.getClipAddControlRfExecutionId();
  sfr.clip_add_control_rf_execution_id_mask =
      op.getClipAddControlRfExecutionIdMask();
  sfr.clip_add_data0_scalar_register_element0 =
      op.getClipAddData0ScalarRegisterElement0();
  sfr.clip_add_data0_scalar_register_element1 =
      op.getClipAddData0ScalarRegisterElement1();
  sfr.clip_add_data1_scalar_register_element2 =
      op.getClipAddData1ScalarRegisterElement2();
  sfr.clip_max_control_op_mode = op.getClipMaxControlOpMode();
  sfr.clip_max_control_arg_mode = op.getClipMaxControlArgMode();
  sfr.clip_max_control_reg0_cmp_op = op.getClipMaxControlReg0CmpOp();
  sfr.clip_max_control_reg1_cmp_op = op.getClipMaxControlReg1CmpOp();
  sfr.clip_max_control_reg2_cmp_op = op.getClipMaxControlReg2CmpOp();
  sfr.clip_max_control_rf_cmp_op = op.getClipMaxControlRfCmpOp();
  sfr.clip_max_control_reg0_execution_id =
      op.getClipMaxControlReg0ExecutionId();
  sfr.clip_max_control_reg0_execution_id_mask =
      op.getClipMaxControlReg0ExecutionIdMask();
  sfr.clip_max_control_reg1_execution_id =
      op.getClipMaxControlReg1ExecutionId();
  sfr.clip_max_control_reg1_execution_id_mask =
      op.getClipMaxControlReg1ExecutionIdMask();
  sfr.clip_max_control_reg2_execution_id =
      op.getClipMaxControlReg2ExecutionId();
  sfr.clip_max_control_reg2_execution_id_mask =
      op.getClipMaxControlReg2ExecutionIdMask();
  sfr.clip_max_control_rf_execution_id = op.getClipMaxControlRfExecutionId();
  sfr.clip_max_control_rf_execution_id_mask =
      op.getClipMaxControlRfExecutionIdMask();
  sfr.clip_max_data0_scalar_register_element0 =
      op.getClipMaxData0ScalarRegisterElement0();
  sfr.clip_max_data0_scalar_register_element1 =
      op.getClipMaxData0ScalarRegisterElement1();
  sfr.clip_max_data1_scalar_register_element2 =
      op.getClipMaxData1ScalarRegisterElement2();
  sfr.clip_min_control_op_mode = op.getClipMinControlOpMode();
  sfr.clip_min_control_arg_mode = op.getClipMinControlArgMode();
  sfr.clip_min_control_reg0_cmp_op = op.getClipMinControlReg0CmpOp();
  sfr.clip_min_control_reg1_cmp_op = op.getClipMinControlReg1CmpOp();
  sfr.clip_min_control_reg2_cmp_op = op.getClipMinControlReg2CmpOp();
  sfr.clip_min_control_rf_cmp_op = op.getClipMinControlRfCmpOp();
  sfr.clip_min_control_reg0_execution_id =
      op.getClipMinControlReg0ExecutionId();
  sfr.clip_min_control_reg0_execution_id_mask =
      op.getClipMinControlReg0ExecutionIdMask();
  sfr.clip_min_control_reg1_execution_id =
      op.getClipMinControlReg1ExecutionId();
  sfr.clip_min_control_reg1_execution_id_mask =
      op.getClipMinControlReg1ExecutionIdMask();
  sfr.clip_min_control_reg2_execution_id =
      op.getClipMinControlReg2ExecutionId();
  sfr.clip_min_control_reg2_execution_id_mask =
      op.getClipMinControlReg2ExecutionIdMask();
  sfr.clip_min_control_rf_execution_id = op.getClipMinControlRfExecutionId();
  sfr.clip_min_control_rf_execution_id_mask =
      op.getClipMinControlRfExecutionIdMask();
  sfr.clip_min_data0_scalar_register_element0 =
      op.getClipMinData0ScalarRegisterElement0();
  sfr.clip_min_data0_scalar_register_element1 =
      op.getClipMinData0ScalarRegisterElement1();
  sfr.clip_min_data1_scalar_register_element2 =
      op.getClipMinData1ScalarRegisterElement2();
  sfr.alloc_indexer_read_indexer0_module =
      op.getAllocIndexerReadIndexer0Module();
  sfr.alloc_indexer_read_indexer1_module =
      op.getAllocIndexerReadIndexer1Module();
  sfr.alloc_indexer_read_indexer2_module =
      op.getAllocIndexerReadIndexer2Module();
  sfr.alloc_indexer_read_indexer3_module =
      op.getAllocIndexerReadIndexer3Module();
  sfr.alloc_indexer_operand_indexer_module =
      op.getAllocIndexerOperandIndexerModule();
  sfr.alloc_indexer_write_indexer_module =
      op.getAllocIndexerWriteIndexerModule();
  sfr.read_indexer0_operation = op.getReadIndexer0Operation();
  sfr.operation_read_indexer0_proceed = op.getOperationReadIndexer0Proceed();
  sfr.operation_read_indexer0_element_size =
      op.getOperationReadIndexer0ElementSize();
  sfr.read_indexer1_operation = op.getReadIndexer1Operation();
  sfr.operation_read_indexer1_proceed = op.getOperationReadIndexer1Proceed();
  sfr.operation_read_indexer1_element_size =
      op.getOperationReadIndexer1ElementSize();
  sfr.read_indexer2_operation = op.getReadIndexer2Operation();
  sfr.operation_read_indexer2_proceed = op.getOperationReadIndexer2Proceed();
  sfr.operation_read_indexer2_element_size =
      op.getOperationReadIndexer2ElementSize();
  sfr.read_indexer3_operation = op.getReadIndexer3Operation();
  sfr.operation_read_indexer3_proceed = op.getOperationReadIndexer3Proceed();
  sfr.operation_read_indexer3_element_size =
      op.getOperationReadIndexer3ElementSize();
  sfr.operand_indexer_operation = op.getOperandIndexerOperation();
  sfr.operation_operand_indexer_proceed =
      op.getOperationOperandIndexerProceed();
  sfr.operation_operand_indexer_update_mode =
      op.getOperationOperandIndexerUpdateMode();
  sfr.operation_operand_indexer_element_size =
      op.getOperationOperandIndexerElementSize();
  sfr.write_indexer_operation = op.getWriteIndexerOperation();
  sfr.indexer_base0_read_indexer0_base = op.getIndexerBase0ReadIndexer0Base();
  sfr.indexer_base0_read_indexer1_base = op.getIndexerBase0ReadIndexer1Base();
  sfr.indexer_base0_read_indexer2_base = op.getIndexerBase0ReadIndexer2Base();
  sfr.indexer_base0_read_indexer3_base = op.getIndexerBase0ReadIndexer3Base();
  sfr.indexer_base1_operand_indexer_base =
      op.getIndexerBase1OperandIndexerBase();
  sfr.indexer_base1_write_indexer_base = op.getIndexerBase1WriteIndexerBase();
  auto read_indexer0_limits = op.getReadIndexer0Limits();
  sfr.read_indexer0_limit_info0_read_indexer0_limit_element0 =
      dyn_cast_or_null<IntegerAttr>(read_indexer0_limits[0]).getInt();
  sfr.read_indexer0_limit_info0_read_indexer0_limit_element1 =
      dyn_cast_or_null<IntegerAttr>(read_indexer0_limits[1]).getInt();
  sfr.read_indexer0_limit_info0_read_indexer0_limit_element2 =
      dyn_cast_or_null<IntegerAttr>(read_indexer0_limits[2]).getInt();
  sfr.read_indexer0_limit_info0_read_indexer0_limit_element3 =
      dyn_cast_or_null<IntegerAttr>(read_indexer0_limits[3]).getInt();
  sfr.read_indexer0_limit_info1_read_indexer0_limit_element4 =
      dyn_cast_or_null<IntegerAttr>(read_indexer0_limits[4]).getInt();
  sfr.read_indexer0_limit_info1_read_indexer0_limit_element5 =
      dyn_cast_or_null<IntegerAttr>(read_indexer0_limits[5]).getInt();
  sfr.read_indexer0_limit_info1_read_indexer0_limit_element6 =
      dyn_cast_or_null<IntegerAttr>(read_indexer0_limits[6]).getInt();
  sfr.read_indexer0_limit_info1_read_indexer0_limit_element7 =
      dyn_cast_or_null<IntegerAttr>(read_indexer0_limits[7]).getInt();
  auto read_indexer0_strides = op.getReadIndexer0Strides();
  sfr.read_indexer0_stride_info0_read_indexer0_stride_element0 =
      dyn_cast_or_null<IntegerAttr>(read_indexer0_strides[0]).getInt();
  sfr.read_indexer0_stride_info0_read_indexer0_stride_element1 =
      dyn_cast_or_null<IntegerAttr>(read_indexer0_strides[1]).getInt();
  sfr.read_indexer0_stride_info0_read_indexer0_stride_element2 =
      dyn_cast_or_null<IntegerAttr>(read_indexer0_strides[2]).getInt();
  sfr.read_indexer0_stride_info0_read_indexer0_stride_element3 =
      dyn_cast_or_null<IntegerAttr>(read_indexer0_strides[3]).getInt();
  sfr.read_indexer0_stride_info1_read_indexer0_stride_element4 =
      dyn_cast_or_null<IntegerAttr>(read_indexer0_strides[4]).getInt();
  sfr.read_indexer0_stride_info1_read_indexer0_stride_element5 =
      dyn_cast_or_null<IntegerAttr>(read_indexer0_strides[5]).getInt();
  sfr.read_indexer0_stride_info1_read_indexer0_stride_element6 =
      dyn_cast_or_null<IntegerAttr>(read_indexer0_strides[6]).getInt();
  sfr.read_indexer0_stride_info1_read_indexer0_stride_element7 =
      dyn_cast_or_null<IntegerAttr>(read_indexer0_strides[7]).getInt();
  auto read_indexer1_limits = op.getReadIndexer1Limits();
  sfr.read_indexer1_limit_info0_read_indexer1_limit_element0 =
      dyn_cast_or_null<IntegerAttr>(read_indexer1_limits[0]).getInt();
  sfr.read_indexer1_limit_info0_read_indexer1_limit_element1 =
      dyn_cast_or_null<IntegerAttr>(read_indexer1_limits[1]).getInt();
  sfr.read_indexer1_limit_info0_read_indexer1_limit_element2 =
      dyn_cast_or_null<IntegerAttr>(read_indexer1_limits[2]).getInt();
  sfr.read_indexer1_limit_info0_read_indexer1_limit_element3 =
      dyn_cast_or_null<IntegerAttr>(read_indexer1_limits[3]).getInt();
  sfr.read_indexer1_limit_info1_read_indexer1_limit_element4 =
      dyn_cast_or_null<IntegerAttr>(read_indexer1_limits[4]).getInt();
  sfr.read_indexer1_limit_info1_read_indexer1_limit_element5 =
      dyn_cast_or_null<IntegerAttr>(read_indexer1_limits[5]).getInt();
  sfr.read_indexer1_limit_info1_read_indexer1_limit_element6 =
      dyn_cast_or_null<IntegerAttr>(read_indexer1_limits[6]).getInt();
  sfr.read_indexer1_limit_info1_read_indexer1_limit_element7 =
      dyn_cast_or_null<IntegerAttr>(read_indexer1_limits[7]).getInt();
  auto read_indexer1_strides = op.getReadIndexer1Strides();
  sfr.read_indexer1_stride_info0_read_indexer1_stride_element0 =
      dyn_cast_or_null<IntegerAttr>(read_indexer1_strides[0]).getInt();
  sfr.read_indexer1_stride_info0_read_indexer1_stride_element1 =
      dyn_cast_or_null<IntegerAttr>(read_indexer1_strides[1]).getInt();
  sfr.read_indexer1_stride_info0_read_indexer1_stride_element2 =
      dyn_cast_or_null<IntegerAttr>(read_indexer1_strides[2]).getInt();
  sfr.read_indexer1_stride_info0_read_indexer1_stride_element3 =
      dyn_cast_or_null<IntegerAttr>(read_indexer1_strides[3]).getInt();
  sfr.read_indexer1_stride_info1_read_indexer1_stride_element4 =
      dyn_cast_or_null<IntegerAttr>(read_indexer1_strides[4]).getInt();
  sfr.read_indexer1_stride_info1_read_indexer1_stride_element5 =
      dyn_cast_or_null<IntegerAttr>(read_indexer1_strides[5]).getInt();
  sfr.read_indexer1_stride_info1_read_indexer1_stride_element6 =
      dyn_cast_or_null<IntegerAttr>(read_indexer1_strides[6]).getInt();
  sfr.read_indexer1_stride_info1_read_indexer1_stride_element7 =
      dyn_cast_or_null<IntegerAttr>(read_indexer1_strides[7]).getInt();
  auto read_indexer2_limits = op.getReadIndexer2Limits();
  sfr.read_indexer2_limit_info0_read_indexer2_limit_element0 =
      dyn_cast_or_null<IntegerAttr>(read_indexer2_limits[0]).getInt();
  sfr.read_indexer2_limit_info0_read_indexer2_limit_element1 =
      dyn_cast_or_null<IntegerAttr>(read_indexer2_limits[1]).getInt();
  sfr.read_indexer2_limit_info0_read_indexer2_limit_element2 =
      dyn_cast_or_null<IntegerAttr>(read_indexer2_limits[2]).getInt();
  sfr.read_indexer2_limit_info0_read_indexer2_limit_element3 =
      dyn_cast_or_null<IntegerAttr>(read_indexer2_limits[3]).getInt();
  sfr.read_indexer2_limit_info1_read_indexer2_limit_element4 =
      dyn_cast_or_null<IntegerAttr>(read_indexer2_limits[4]).getInt();
  sfr.read_indexer2_limit_info1_read_indexer2_limit_element5 =
      dyn_cast_or_null<IntegerAttr>(read_indexer2_limits[5]).getInt();
  sfr.read_indexer2_limit_info1_read_indexer2_limit_element6 =
      dyn_cast_or_null<IntegerAttr>(read_indexer2_limits[6]).getInt();
  sfr.read_indexer2_limit_info1_read_indexer2_limit_element7 =
      dyn_cast_or_null<IntegerAttr>(read_indexer2_limits[7]).getInt();
  auto read_indexer2_strides = op.getReadIndexer2Strides();
  sfr.read_indexer2_stride_info0_read_indexer2_stride_element0 =
      dyn_cast_or_null<IntegerAttr>(read_indexer2_strides[0]).getInt();
  sfr.read_indexer2_stride_info0_read_indexer2_stride_element1 =
      dyn_cast_or_null<IntegerAttr>(read_indexer2_strides[1]).getInt();
  sfr.read_indexer2_stride_info0_read_indexer2_stride_element2 =
      dyn_cast_or_null<IntegerAttr>(read_indexer2_strides[2]).getInt();
  sfr.read_indexer2_stride_info0_read_indexer2_stride_element3 =
      dyn_cast_or_null<IntegerAttr>(read_indexer2_strides[3]).getInt();
  sfr.read_indexer2_stride_info1_read_indexer2_stride_element4 =
      dyn_cast_or_null<IntegerAttr>(read_indexer2_strides[4]).getInt();
  sfr.read_indexer2_stride_info1_read_indexer2_stride_element5 =
      dyn_cast_or_null<IntegerAttr>(read_indexer2_strides[5]).getInt();
  sfr.read_indexer2_stride_info1_read_indexer2_stride_element6 =
      dyn_cast_or_null<IntegerAttr>(read_indexer2_strides[6]).getInt();
  sfr.read_indexer2_stride_info1_read_indexer2_stride_element7 =
      dyn_cast_or_null<IntegerAttr>(read_indexer2_strides[7]).getInt();
  auto read_indexer3_limits = op.getReadIndexer3Limits();
  sfr.read_indexer3_limit_info0_read_indexer3_limit_element0 =
      dyn_cast_or_null<IntegerAttr>(read_indexer3_limits[0]).getInt();
  sfr.read_indexer3_limit_info0_read_indexer3_limit_element1 =
      dyn_cast_or_null<IntegerAttr>(read_indexer3_limits[1]).getInt();
  sfr.read_indexer3_limit_info0_read_indexer3_limit_element2 =
      dyn_cast_or_null<IntegerAttr>(read_indexer3_limits[2]).getInt();
  sfr.read_indexer3_limit_info0_read_indexer3_limit_element3 =
      dyn_cast_or_null<IntegerAttr>(read_indexer3_limits[3]).getInt();
  sfr.read_indexer3_limit_info1_read_indexer3_limit_element4 =
      dyn_cast_or_null<IntegerAttr>(read_indexer3_limits[4]).getInt();
  sfr.read_indexer3_limit_info1_read_indexer3_limit_element5 =
      dyn_cast_or_null<IntegerAttr>(read_indexer3_limits[5]).getInt();
  sfr.read_indexer3_limit_info1_read_indexer3_limit_element6 =
      dyn_cast_or_null<IntegerAttr>(read_indexer3_limits[6]).getInt();
  sfr.read_indexer3_limit_info1_read_indexer3_limit_element7 =
      dyn_cast_or_null<IntegerAttr>(read_indexer3_limits[7]).getInt();
  auto read_indexer3_strides = op.getReadIndexer3Strides();
  sfr.read_indexer3_stride_info0_read_indexer3_stride_element0 =
      dyn_cast_or_null<IntegerAttr>(read_indexer3_strides[0]).getInt();
  sfr.read_indexer3_stride_info0_read_indexer3_stride_element1 =
      dyn_cast_or_null<IntegerAttr>(read_indexer3_strides[1]).getInt();
  sfr.read_indexer3_stride_info0_read_indexer3_stride_element2 =
      dyn_cast_or_null<IntegerAttr>(read_indexer3_strides[2]).getInt();
  sfr.read_indexer3_stride_info0_read_indexer3_stride_element3 =
      dyn_cast_or_null<IntegerAttr>(read_indexer3_strides[3]).getInt();
  sfr.read_indexer3_stride_info1_read_indexer3_stride_element4 =
      dyn_cast_or_null<IntegerAttr>(read_indexer3_strides[4]).getInt();
  sfr.read_indexer3_stride_info1_read_indexer3_stride_element5 =
      dyn_cast_or_null<IntegerAttr>(read_indexer3_strides[5]).getInt();
  sfr.read_indexer3_stride_info1_read_indexer3_stride_element6 =
      dyn_cast_or_null<IntegerAttr>(read_indexer3_strides[6]).getInt();
  sfr.read_indexer3_stride_info1_read_indexer3_stride_element7 =
      dyn_cast_or_null<IntegerAttr>(read_indexer3_strides[7]).getInt();
  auto operand_indexer_limits = op.getOperandIndexerLimits();
  sfr.operand_indexer_limit_info0_operand_indexer_limit_element0 =
      dyn_cast_or_null<IntegerAttr>(operand_indexer_limits[0]).getInt();
  sfr.operand_indexer_limit_info0_operand_indexer_limit_element1 =
      dyn_cast_or_null<IntegerAttr>(operand_indexer_limits[1]).getInt();
  sfr.operand_indexer_limit_info0_operand_indexer_limit_element2 =
      dyn_cast_or_null<IntegerAttr>(operand_indexer_limits[2]).getInt();
  sfr.operand_indexer_limit_info0_operand_indexer_limit_element3 =
      dyn_cast_or_null<IntegerAttr>(operand_indexer_limits[3]).getInt();
  sfr.operand_indexer_limit_info1_operand_indexer_limit_element4 =
      dyn_cast_or_null<IntegerAttr>(operand_indexer_limits[4]).getInt();
  sfr.operand_indexer_limit_info1_operand_indexer_limit_element5 =
      dyn_cast_or_null<IntegerAttr>(operand_indexer_limits[5]).getInt();
  sfr.operand_indexer_limit_info1_operand_indexer_limit_element6 =
      dyn_cast_or_null<IntegerAttr>(operand_indexer_limits[6]).getInt();
  sfr.operand_indexer_limit_info1_operand_indexer_limit_element7 =
      dyn_cast_or_null<IntegerAttr>(operand_indexer_limits[7]).getInt();
  auto operand_indexer_strides = op.getOperandIndexerStrides();
  sfr.operand_indexer_stride_info0_operand_indexer_stride_element0 =
      dyn_cast_or_null<IntegerAttr>(operand_indexer_strides[0]).getInt();
  sfr.operand_indexer_stride_info0_operand_indexer_stride_element1 =
      dyn_cast_or_null<IntegerAttr>(operand_indexer_strides[1]).getInt();
  sfr.operand_indexer_stride_info0_operand_indexer_stride_element2 =
      dyn_cast_or_null<IntegerAttr>(operand_indexer_strides[2]).getInt();
  sfr.operand_indexer_stride_info0_operand_indexer_stride_element3 =
      dyn_cast_or_null<IntegerAttr>(operand_indexer_strides[3]).getInt();
  sfr.operand_indexer_stride_info1_operand_indexer_stride_element4 =
      dyn_cast_or_null<IntegerAttr>(operand_indexer_strides[4]).getInt();
  sfr.operand_indexer_stride_info1_operand_indexer_stride_element5 =
      dyn_cast_or_null<IntegerAttr>(operand_indexer_strides[5]).getInt();
  sfr.operand_indexer_stride_info1_operand_indexer_stride_element6 =
      dyn_cast_or_null<IntegerAttr>(operand_indexer_strides[6]).getInt();
  sfr.operand_indexer_stride_info1_operand_indexer_stride_element7 =
      dyn_cast_or_null<IntegerAttr>(operand_indexer_strides[7]).getInt();
  auto write_indexer_limits = op.getWriteIndexerLimits();
  sfr.write_indexer_limit_info0_write_indexer_limit_element0 =
      dyn_cast_or_null<IntegerAttr>(write_indexer_limits[0]).getInt();
  sfr.write_indexer_limit_info0_write_indexer_limit_element1 =
      dyn_cast_or_null<IntegerAttr>(write_indexer_limits[1]).getInt();
  sfr.write_indexer_limit_info0_write_indexer_limit_element2 =
      dyn_cast_or_null<IntegerAttr>(write_indexer_limits[2]).getInt();
  sfr.write_indexer_limit_info0_write_indexer_limit_element3 =
      dyn_cast_or_null<IntegerAttr>(write_indexer_limits[3]).getInt();
  sfr.write_indexer_limit_info1_write_indexer_limit_element4 =
      dyn_cast_or_null<IntegerAttr>(write_indexer_limits[4]).getInt();
  sfr.write_indexer_limit_info1_write_indexer_limit_element5 =
      dyn_cast_or_null<IntegerAttr>(write_indexer_limits[5]).getInt();
  sfr.write_indexer_limit_info1_write_indexer_limit_element6 =
      dyn_cast_or_null<IntegerAttr>(write_indexer_limits[6]).getInt();
  sfr.write_indexer_limit_info1_write_indexer_limit_element7 =
      dyn_cast_or_null<IntegerAttr>(write_indexer_limits[7]).getInt();
  auto write_indexer_strides = op.getWriteIndexerStrides();
  sfr.write_indexer_stride_info0_write_indexer_stride_element0 =
      dyn_cast_or_null<IntegerAttr>(write_indexer_strides[0]).getInt();
  sfr.write_indexer_stride_info0_write_indexer_stride_element1 =
      dyn_cast_or_null<IntegerAttr>(write_indexer_strides[1]).getInt();
  sfr.write_indexer_stride_info0_write_indexer_stride_element2 =
      dyn_cast_or_null<IntegerAttr>(write_indexer_strides[2]).getInt();
  sfr.write_indexer_stride_info0_write_indexer_stride_element3 =
      dyn_cast_or_null<IntegerAttr>(write_indexer_strides[3]).getInt();
  sfr.write_indexer_stride_info1_write_indexer_stride_element4 =
      dyn_cast_or_null<IntegerAttr>(write_indexer_strides[4]).getInt();
  sfr.write_indexer_stride_info1_write_indexer_stride_element5 =
      dyn_cast_or_null<IntegerAttr>(write_indexer_strides[5]).getInt();
  sfr.write_indexer_stride_info1_write_indexer_stride_element6 =
      dyn_cast_or_null<IntegerAttr>(write_indexer_strides[6]).getInt();
  sfr.write_indexer_stride_info1_write_indexer_stride_element7 =
      dyn_cast_or_null<IntegerAttr>(write_indexer_strides[7]).getInt();

  return sfr.get_blocks();
}

template <typename T, std::enable_if_t<
                          std::is_same_v<T, sfr::StaticSfrVectorReduceUnitOp> ||
                              std::is_same_v<T, sfr::SfrVectorReduceUnitOp>,
                          bool> = true>
std::vector<sfr_data_t> getSfrVectorReduceUnit(T &op) {
  sfr::slice::VectorReduceUnitMainContext<sfr_data_t> sfr{};
  sfr.cluster_route_add_source = op.getClusterRouteAddSource();
  sfr.cluster_route_max_source = op.getClusterRouteMaxSource();
  sfr.cluster_route_min_source = op.getClusterRouteMinSource();
  sfr.cluster_route_mul_source = op.getClusterRouteMulSource();
  sfr.cluster_route_cluster_source = op.getClusterRouteClusterSource();
  sfr.mul_control_op_mode = op.getMulControlOpMode();
  sfr.mul_control_arg_mode = op.getMulControlArgMode();
  sfr.mul_control_reg0_cmp_op = op.getMulControlReg0CmpOp();
  sfr.mul_control_reg1_cmp_op = op.getMulControlReg1CmpOp();
  sfr.mul_control_reg2_cmp_op = op.getMulControlReg2CmpOp();
  sfr.mul_control_rhs_cmp_op = op.getMulControlRhsCmpOp();
  sfr.mul_control_reg0_execution_id = op.getMulControlReg0ExecutionId();
  sfr.mul_control_reg0_execution_id_mask =
      op.getMulControlReg0ExecutionIdMask();
  sfr.mul_control_reg1_execution_id = op.getMulControlReg1ExecutionId();
  sfr.mul_control_reg1_execution_id_mask =
      op.getMulControlReg1ExecutionIdMask();
  sfr.mul_control_reg2_execution_id = op.getMulControlReg2ExecutionId();
  sfr.mul_control_reg2_execution_id_mask =
      op.getMulControlReg2ExecutionIdMask();
  sfr.mul_control_io_execution_id = op.getMulControlIoExecutionId();
  sfr.mul_control_io_execution_id_mask = op.getMulControlIoExecutionIdMask();
  sfr.mul_data0_scalar_register_element0 =
      op.getMulData0ScalarRegisterElement0();
  sfr.mul_data0_scalar_register_element1 =
      op.getMulData0ScalarRegisterElement1();
  sfr.mul_data1_scalar_register_element2 =
      op.getMulData1ScalarRegisterElement2();
  sfr.add_control_op_mode = op.getAddControlOpMode();
  sfr.add_control_arg_mode = op.getAddControlArgMode();
  sfr.add_control_reg0_cmp_op = op.getAddControlReg0CmpOp();
  sfr.add_control_reg1_cmp_op = op.getAddControlReg1CmpOp();
  sfr.add_control_reg2_cmp_op = op.getAddControlReg2CmpOp();
  sfr.add_control_rhs_cmp_op = op.getAddControlRhsCmpOp();
  sfr.add_control_reg0_execution_id = op.getAddControlReg0ExecutionId();
  sfr.add_control_reg0_execution_id_mask =
      op.getAddControlReg0ExecutionIdMask();
  sfr.add_control_reg1_execution_id = op.getAddControlReg1ExecutionId();
  sfr.add_control_reg1_execution_id_mask =
      op.getAddControlReg1ExecutionIdMask();
  sfr.add_control_reg2_execution_id = op.getAddControlReg2ExecutionId();
  sfr.add_control_reg2_execution_id_mask =
      op.getAddControlReg2ExecutionIdMask();
  sfr.add_control_io_execution_id = op.getAddControlIoExecutionId();
  sfr.add_control_io_execution_id_mask = op.getAddControlIoExecutionIdMask();
  sfr.add_data0_scalar_register_element0 =
      op.getAddData0ScalarRegisterElement0();
  sfr.add_data0_scalar_register_element1 =
      op.getAddData0ScalarRegisterElement1();
  sfr.add_data1_scalar_register_element2 =
      op.getAddData1ScalarRegisterElement2();
  sfr.max_control_op_mode = op.getMaxControlOpMode();
  sfr.max_control_arg_mode = op.getMaxControlArgMode();
  sfr.max_control_reg0_cmp_op = op.getMaxControlReg0CmpOp();
  sfr.max_control_reg1_cmp_op = op.getMaxControlReg1CmpOp();
  sfr.max_control_reg2_cmp_op = op.getMaxControlReg2CmpOp();
  sfr.max_control_rhs_cmp_op = op.getMaxControlRhsCmpOp();
  sfr.max_control_reg0_execution_id = op.getMaxControlReg0ExecutionId();
  sfr.max_control_reg0_execution_id_mask =
      op.getMaxControlReg0ExecutionIdMask();
  sfr.max_control_reg1_execution_id = op.getMaxControlReg1ExecutionId();
  sfr.max_control_reg1_execution_id_mask =
      op.getMaxControlReg1ExecutionIdMask();
  sfr.max_control_reg2_execution_id = op.getMaxControlReg2ExecutionId();
  sfr.max_control_reg2_execution_id_mask =
      op.getMaxControlReg2ExecutionIdMask();
  sfr.max_control_io_execution_id = op.getMaxControlIoExecutionId();
  sfr.max_control_io_execution_id_mask = op.getMaxControlIoExecutionIdMask();
  sfr.max_data0_scalar_register_element0 =
      op.getMaxData0ScalarRegisterElement0();
  sfr.max_data0_scalar_register_element1 =
      op.getMaxData0ScalarRegisterElement1();
  sfr.max_data1_scalar_register_element2 =
      op.getMaxData1ScalarRegisterElement2();
  sfr.min_control_op_mode = op.getMinControlOpMode();
  sfr.min_control_arg_mode = op.getMinControlArgMode();
  sfr.min_control_reg0_cmp_op = op.getMinControlReg0CmpOp();
  sfr.min_control_reg1_cmp_op = op.getMinControlReg1CmpOp();
  sfr.min_control_reg2_cmp_op = op.getMinControlReg2CmpOp();
  sfr.min_control_rhs_cmp_op = op.getMinControlRhsCmpOp();
  sfr.min_control_reg0_execution_id = op.getMinControlReg0ExecutionId();
  sfr.min_control_reg0_execution_id_mask =
      op.getMinControlReg0ExecutionIdMask();
  sfr.min_control_reg1_execution_id = op.getMinControlReg1ExecutionId();
  sfr.min_control_reg1_execution_id_mask =
      op.getMinControlReg1ExecutionIdMask();
  sfr.min_control_reg2_execution_id = op.getMinControlReg2ExecutionId();
  sfr.min_control_reg2_execution_id_mask =
      op.getMinControlReg2ExecutionIdMask();
  sfr.min_control_io_execution_id = op.getMinControlIoExecutionId();
  sfr.min_control_io_execution_id_mask = op.getMinControlIoExecutionIdMask();
  sfr.min_data0_scalar_register_element0 =
      op.getMinData0ScalarRegisterElement0();
  sfr.min_data0_scalar_register_element1 =
      op.getMinData0ScalarRegisterElement1();
  sfr.min_data1_scalar_register_element2 =
      op.getMinData1ScalarRegisterElement2();

  return sfr.get_blocks();
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, sfr::StaticSfrVectorRegisterFileOp> ||
                         std::is_same_v<T, sfr::SfrVectorRegisterFileOp>,
                     bool> = true>
std::vector<sfr_data_t> getSfrVectorRegisterFile(T &op) {
  sfr::slice::VectorRegisterFile<sfr_data_t> sfr{};
  sfr.write_row_base = op.getWriteRowBase();
  sfr.write_row_count = op.getWriteRowCount();
  sfr.write_skip_flit_count = op.getWriteSkipFlitCount();
  sfr.write_row_stride = op.getWriteRowStride();

  return sfr.get_blocks();
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, sfr::StaticSfrVectorRouteUnitOp> ||
                               std::is_same_v<T, sfr::SfrVectorRouteUnitOp>,
                           bool> = true>
std::vector<sfr_data_t> getSfrVectorRouteUnit(T &op) {
  sfr::slice::VectorRouteUnitMainContext<sfr_data_t> sfr{};
  sfr.route_info_data_out_source = op.getRouteInfoDataOutSource();
  sfr.route_info_reduce_channel_out_source =
      op.getRouteInfoReduceChannelOutSource();
  sfr.route_info_reduce_unit_in_source = op.getRouteInfoReduceUnitInSource();
  sfr.route_info_arithmetic_unit_in_source =
      op.getRouteInfoArithmeticUnitInSource();
  sfr.route_info_valid_generator_mode = op.getRouteInfoValidGeneratorMode();
  sfr.route_info_route_mask = op.getRouteInfoRouteMask();
  sfr.route_info_route_group_size = op.getRouteInfoRouteGroupSize();
  sfr.route_info_index_base = op.getRouteInfoIndexBase();
  auto indexer_limits = op.getIndexerLimits();
  sfr.indexer_limit0_index_limit_element0 =
      dyn_cast_or_null<IntegerAttr>(indexer_limits[0]).getInt();
  sfr.indexer_limit0_index_limit_element1 =
      dyn_cast_or_null<IntegerAttr>(indexer_limits[0]).getInt();
  sfr.indexer_limit0_index_limit_element2 =
      dyn_cast_or_null<IntegerAttr>(indexer_limits[0]).getInt();
  sfr.indexer_limit0_index_limit_element3 =
      dyn_cast_or_null<IntegerAttr>(indexer_limits[0]).getInt();
  sfr.indexer_limit1_index_limit_element4 =
      dyn_cast_or_null<IntegerAttr>(indexer_limits[0]).getInt();
  sfr.indexer_limit1_index_limit_element5 =
      dyn_cast_or_null<IntegerAttr>(indexer_limits[0]).getInt();
  sfr.indexer_limit1_index_limit_element6 =
      dyn_cast_or_null<IntegerAttr>(indexer_limits[0]).getInt();
  sfr.indexer_limit1_index_limit_element7 =
      dyn_cast_or_null<IntegerAttr>(indexer_limits[0]).getInt();
  auto indexer_strides = op.getIndexerStrides();
  sfr.indexer_stride0_index_stride_element0 =
      dyn_cast_or_null<IntegerAttr>(indexer_strides[0]).getInt();
  sfr.indexer_stride0_index_stride_element1 =
      dyn_cast_or_null<IntegerAttr>(indexer_strides[1]).getInt();
  sfr.indexer_stride0_index_stride_element2 =
      dyn_cast_or_null<IntegerAttr>(indexer_strides[2]).getInt();
  sfr.indexer_stride0_index_stride_element3 =
      dyn_cast_or_null<IntegerAttr>(indexer_strides[3]).getInt();
  sfr.indexer_stride1_index_stride_element4 =
      dyn_cast_or_null<IntegerAttr>(indexer_strides[4]).getInt();
  sfr.indexer_stride1_index_stride_element5 =
      dyn_cast_or_null<IntegerAttr>(indexer_strides[5]).getInt();
  sfr.indexer_stride1_index_stride_element6 =
      dyn_cast_or_null<IntegerAttr>(indexer_strides[6]).getInt();
  sfr.indexer_stride1_index_stride_element7 =
      dyn_cast_or_null<IntegerAttr>(indexer_strides[7]).getInt();
  auto valid_generator_lowered_limits = op.getValidGeneratorLoweredLimits();
  sfr.valid_generator_limit0_lowered_limit_element0 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_lowered_limits[0]).getInt();
  sfr.valid_generator_limit0_lowered_limit_element1 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_lowered_limits[1]).getInt();
  sfr.valid_generator_limit0_lowered_limit_element2 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_lowered_limits[2]).getInt();
  sfr.valid_generator_limit0_lowered_limit_element3 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_lowered_limits[3]).getInt();
  sfr.valid_generator_limit1_lowered_limit_element4 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_lowered_limits[4]).getInt();
  sfr.valid_generator_limit1_lowered_limit_element5 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_lowered_limits[5]).getInt();
  sfr.valid_generator_limit1_lowered_limit_element6 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_lowered_limits[6]).getInt();
  sfr.valid_generator_limit1_lowered_limit_element7 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_lowered_limits[7]).getInt();
  auto valid_generator_lowered_strides = op.getValidGeneratorLoweredStrides();
  sfr.valid_generator_stride0_lowered_stride_element0 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_lowered_strides[0])
          .getInt();
  sfr.valid_generator_stride0_lowered_stride_element1 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_lowered_strides[1])
          .getInt();
  sfr.valid_generator_stride0_lowered_stride_element2 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_lowered_strides[2])
          .getInt();
  sfr.valid_generator_stride0_lowered_stride_element3 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_lowered_strides[3])
          .getInt();
  sfr.valid_generator_stride1_lowered_stride_element4 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_lowered_strides[4])
          .getInt();
  sfr.valid_generator_stride1_lowered_stride_element5 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_lowered_strides[5])
          .getInt();
  sfr.valid_generator_stride1_lowered_stride_element6 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_lowered_strides[6])
          .getInt();
  sfr.valid_generator_stride1_lowered_stride_element7 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_lowered_strides[7])
          .getInt();
  auto valid_generator_allocated_original_dim =
      op.getValidGeneratorAllocatedOriginalDim();
  sfr.valid_generator_original_dim_allocated_original_dim0 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_allocated_original_dim[0])
          .getInt();
  sfr.valid_generator_original_dim_allocated_original_dim1 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_allocated_original_dim[1])
          .getInt();
  sfr.valid_generator_original_dim_allocated_original_dim2 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_allocated_original_dim[2])
          .getInt();
  sfr.valid_generator_original_dim_allocated_original_dim3 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_allocated_original_dim[3])
          .getInt();
  sfr.valid_generator_original_dim_allocated_original_dim4 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_allocated_original_dim[4])
          .getInt();
  sfr.valid_generator_original_dim_allocated_original_dim5 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_allocated_original_dim[5])
          .getInt();
  sfr.valid_generator_original_dim_allocated_original_dim6 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_allocated_original_dim[6])
          .getInt();
  sfr.valid_generator_original_dim_allocated_original_dim7 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_allocated_original_dim[7])
          .getInt();
  auto valid_generator_original_dim_partition_config =
      op.getValidGeneratorOriginalDimPartitionConfig();
  sfr.valid_generator_original_dim_original_dim_partition_config0 =
      dyn_cast_or_null<IntegerAttr>(
          valid_generator_original_dim_partition_config[0])
          .getInt();
  sfr.valid_generator_original_dim_original_dim_partition_config1 =
      dyn_cast_or_null<IntegerAttr>(
          valid_generator_original_dim_partition_config[1])
          .getInt();
  sfr.valid_generator_original_dim_original_dim_partition_config2 =
      dyn_cast_or_null<IntegerAttr>(
          valid_generator_original_dim_partition_config[2])
          .getInt();
  sfr.valid_generator_original_dim_original_dim_partition_config3 =
      dyn_cast_or_null<IntegerAttr>(
          valid_generator_original_dim_partition_config[3])
          .getInt();
  auto valid_generator_original_dim_valid_count =
      op.getValidGeneratorOriginalDimValidCount();
  sfr.valid_generator_valid_count_original_dim_valid_count0 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_original_dim_valid_count[0])
          .getInt();
  sfr.valid_generator_valid_count_original_dim_valid_count1 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_original_dim_valid_count[1])
          .getInt();
  sfr.valid_generator_valid_count_original_dim_valid_count2 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_original_dim_valid_count[2])
          .getInt();
  sfr.valid_generator_valid_count_original_dim_valid_count3 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_original_dim_valid_count[3])
          .getInt();
  auto valid_generator_slice_mask = op.getValidGeneratorSliceMask();
  sfr.valid_generator_slice_info_slice_mask0 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_slice_mask[0]).getInt();
  sfr.valid_generator_slice_info_slice_mask1 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_slice_mask[1]).getInt();
  sfr.valid_generator_slice_info_slice_mask2 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_slice_mask[2]).getInt();
  sfr.valid_generator_slice_info_slice_mask3 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_slice_mask[3]).getInt();
  auto valid_generator_slice_id_match = op.getValidGeneratorSliceIdMatch();
  sfr.valid_generator_slice_info_slice_id_match0 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_slice_id_match[0]).getInt();
  sfr.valid_generator_slice_info_slice_id_match1 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_slice_id_match[1]).getInt();
  sfr.valid_generator_slice_info_slice_id_match2 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_slice_id_match[2]).getInt();
  sfr.valid_generator_slice_info_slice_id_match3 =
      dyn_cast_or_null<IntegerAttr>(valid_generator_slice_id_match[3]).getInt();
  sfr.collect_compaction_mode = op.getCollectCompactionMode();
  sfr.compaction_mode_collect_compaction_cmp_op =
      op.getCompactionModeCollectCompactionCmpOp();
  sfr.compaction_mode_collect_compaction_execution_id =
      op.getCompactionModeCollectCompactionExecutionId();
  sfr.compaction_mode_collect_compaction_execution_id_mask =
      op.getCompactionModeCollectCompactionExecutionIdMask();
  sfr.cast_compaction_mode = op.getCastCompactionMode();
  sfr.compaction_mode_cast_compaction_count =
      op.getCompactionModeCastCompactionCount();

  return sfr.get_blocks();
}

FailureOr<std::tuple<std::uint64_t, std::vector<sfr_data_t>>>
getStaticSfr(Operation &op) {
  return llvm::TypeSwitch<
             Operation *,
             FailureOr<std::tuple<std::uint64_t, std::vector<sfr_data_t>>>>(&op)
      .Case<sfr::StaticSfrDotProductEngineOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrDotProductEngine(op));
      })
      .Case<sfr::StaticSfrMainCommitUnitOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrMainCommitUnit(op));
      })
      .Case<sfr::StaticSfrMainDataPathUnitOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrMainDataPathUnit(op));
      })
      .Case<sfr::StaticSfrMainFetchUnitOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrMainFetchUnit(op));
      })
      .Case<sfr::StaticSfrRegisterConfigUnitOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrRegisterConfigUnit(op));
      })
      .Case<sfr::StaticSfrSubCommitUnitOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrSubCommitUnit(op));
      })
      .Case<sfr::StaticSfrSubDataPathUnitOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrSubDataPathUnit(op));
      })
      .Case<sfr::StaticSfrSubFetchUnitOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrSubFetchUnit(op));
      })
      .Case<sfr::StaticSfrTensorRegisterFileOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrTensorRegisterFile(op));
      })
      .Case<sfr::StaticSfrTransposeEngineOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrTransposeEngine(op));
      })
      .Case<sfr::StaticSfrVectorArithmeticUnitOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrVectorArithmeticUnit(op));
      })
      .Case<sfr::StaticSfrVectorReduceUnitOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrVectorReduceUnit(op));
      })
      .Case<sfr::StaticSfrVectorRegisterFileOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrVectorRegisterFile(op));
      })
      .Case<sfr::StaticSfrVectorRouteUnitOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrVectorRouteUnit(op));
      })
      .Default([&](Operation *) {
        return op.emitOpError(
            "unable to interpret as furiosa dialect operator");
      });
}

FailureOr<std::vector<sfr_data_t>> getSfr(Operation &op) {
  return llvm::TypeSwitch<Operation *, FailureOr<std::vector<sfr_data_t>>>(&op)
      .Case<sfr::SfrDotProductEngineOp>(
          [&](auto op) { return getSfrDotProductEngine(op); })
      .Case<sfr::SfrMainCommitUnitOp>(
          [&](auto op) { return getSfrMainCommitUnit(op); })
      .Case<sfr::SfrMainDataPathUnitOp>(
          [&](auto op) { return getSfrMainDataPathUnit(op); })
      .Case<sfr::SfrMainFetchUnitOp>(
          [&](auto op) { return getSfrMainFetchUnit(op); })
      .Case<sfr::SfrRegisterConfigUnitOp>(
          [&](auto op) { return getSfrRegisterConfigUnit(op); })
      .Case<sfr::SfrSubCommitUnitOp>(
          [&](auto op) { return getSfrSubCommitUnit(op); })
      .Case<sfr::SfrSubDataPathUnitOp>(
          [&](auto op) { return getSfrSubDataPathUnit(op); })
      .Case<sfr::SfrSubFetchUnitOp>(
          [&](auto op) { return getSfrSubFetchUnit(op); })
      .Case<sfr::SfrTensorRegisterFileOp>(
          [&](auto op) { return getSfrTensorRegisterFile(op); })
      .Case<sfr::SfrTransposeEngineOp>(
          [&](auto op) { return getSfrTransposeEngine(op); })
      .Case<sfr::SfrVectorArithmeticUnitOp>(
          [&](auto op) { return getSfrVectorArithmeticUnit(op); })
      .Case<sfr::SfrVectorReduceUnitOp>(
          [&](auto op) { return getSfrVectorReduceUnit(op); })
      .Case<sfr::SfrVectorRegisterFileOp>(
          [&](auto op) { return getSfrVectorRegisterFile(op); })
      .Case<sfr::SfrVectorRouteUnitOp>(
          [&](auto op) { return getSfrVectorRouteUnit(op); })
      .Default([&](Operation *) {
        return op.emitOpError(
            "unable to interpret as furiosa dialect operator");
      });
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, StaticDmaDescriptorOp> ||
                               std::is_same_v<T, DmaDescriptorOp>,
                           bool> = true>
FailureOr<TensorDmaDescriptor> getDmaDescriptor(T &op) {
  TensorDmaDescriptor descriptor{};
  descriptor.opcode = op.getOpcode();
  descriptor.indirect.value = op.getIndirect();
  if constexpr (std::is_same_v<T, StaticDmaDescriptorOp>) {
    descriptor.source_base = op.getSourceBase();
    descriptor.destination_base = op.getDestinationBase();
  } else { // DmaDescriptorOp
    if (auto source_base = op.getSourceBase()) {
      descriptor.source_base = *source_base;
    } else if (!op.getSource()) {
      return op.emitOpError("source is not set");
    }
    if (auto destination_base = op.getDestinationBase()) {
      descriptor.destination_base = *destination_base;
    } else if (!op.getDestination()) {
      return op.emitOpError("destination is not set");
    }
  }
  auto source_limits = op.getSourceLimits();
  auto source_strides = op.getSourceStrides();
  auto destination_limits = op.getDestinationLimits();
  auto destination_strides = op.getDestinationStrides();
  for (auto i = 0; i < DIMS; ++i) {
    descriptor.source_limits[i] =
        dyn_cast_or_null<IntegerAttr>(source_limits[i]).getInt();
    descriptor.source_strides[i] =
        dyn_cast_or_null<IntegerAttr>(source_strides[i]).getInt();
    descriptor.destination_limits[i] =
        dyn_cast_or_null<IntegerAttr>(destination_limits[i]).getInt();
    descriptor.destination_strides[i] =
        dyn_cast_or_null<IntegerAttr>(destination_strides[i]).getInt();
  }
  return descriptor;
}

} // namespace mlir::furiosa::task
