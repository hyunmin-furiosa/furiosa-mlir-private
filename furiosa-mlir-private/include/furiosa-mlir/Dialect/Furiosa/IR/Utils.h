#pragma once

#include "furiosa-mlir/Dialect/Furiosa/IR/Commands.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/Sfr.h"

namespace mlir::furiosa {

FailureOr<std::uint32_t> getOpcode(Operation &op) {
  return llvm::TypeSwitch<Operation *, FailureOr<std::uint32_t>>(&op)
      .Case<TucItosfrOp>([&](auto op) { return 0x01; })
      .Case<TucRtosfrOp>([&](auto op) { return 0x02; })
      .Case<TucRtosfriOp>([&](auto op) { return 0x03; })
      .Case<TucMtosfrOp>([&](auto op) { return 0x04; })
      .Case<TaskMtosfrOp>([&](auto op) { return 0x04; })
      .Case<TucStosfrOp>([&](auto op) { return 0x05; })
      .Case<TucSfrtosOp>([&](auto op) { return 0x06; })
      .Case<TucStallOp>([&](auto op) { return 0x07; })
      .Case<TucItosOp>([&](auto op) { return 0x08; })
      .Case<TucItosiOp>([&](auto op) { return 0x09; })
      .Case<TucStosOp>([&](auto op) { return 0x0a; })
      .Case<TucStotabOp>([&](auto op) { return 0x0b; })
      .Case<TucStotrfOp>([&](auto op) { return 0x0c; })
      .Case<TucStovrfOp>([&](auto op) { return 0x0d; })
      .Case<TucExecutionOp>([&](auto op) { return 0x10; })
      .Case<TucWaitOp>([&](auto op) { return 0x11; })
      .Case<TucWaitiOp>([&](auto op) { return 0x15; })
      .Case<TucInterruptOp>([&](auto op) { return 0x12; })
      .Case<TucDmaOp>([&](auto op) { return 0x13; })
      .Case<TucDma1Op>([&](auto op) { return 0x14; })
      .Case<TucDmawOp>([&](auto op) { return 0x16; })
      .Case<TaskDmawOp>([&](auto op) { return 0x16; })
      .Case<TucProfileOp>([&](auto op) { return 0x18; })
      .Case<TucProfileiOp>([&](auto op) { return 0x19; })
      .Case<TucPrflushOp>([&](auto op) { return 0x1a; })
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
      .Case<TucItosfrOp>([&](auto op) {
        command.itosfr.value = op.getValue();
        {
          GeneralRegister reg;
          reg.itosfr_0.sfr_address = op.getSfrAddress();
          reg.itosfr_0.size = op.getSize();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<TucRtosfrOp>([&](auto op) {
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
      .Case<TucRtosfriOp>([&](auto op) {
        command.rtosfri.log_size = op.getLogSize();
        command.rtosfri.sfr_address = op.getSfrAddress();
        {
          GeneralRegister reg;
          reg.rtosfr_0.value = op.getValue();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<TucMtosfrOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.mtosfr_0.spm_address = op.getSpmAddress();
          reg.mtosfr_0.size = op.getSize();
          reg.mtosfr_0.sfr_address = op.getSfrAddress();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<TaskMtosfrOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.mtosfr_0.sfr_address = op.getSfrAddress();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<TucStosfrOp>([&](auto op) {
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
      .Case<TucSfrtosOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.sfrtos_0.commit_base = op.getCommitBase();
          reg.sfrtos_0.commit_limit = op.getCommitLimit();
          reg.sfrtos_0.sfr_address = op.getSfrAddress();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<TucStallOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.stall_0.cycle = op.getCycle();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<TucItosOp>([&](auto op) {
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
      .Case<TucItosiOp>([&](auto op) {
        command.value = op.getValue();
        {
          GeneralRegister reg;
          reg.itos_0.address_begin = op.getAddressBegin();
          reg.itos_0.address_end = op.getAddressEnd();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<TucStosOp>([&](auto op) {
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
      .Case<TucStotabOp>([&](auto op) {
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
      .Case<TucStotrfOp>([&](auto op) {
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
      .Case<TucStovrfOp>([&](auto op) {
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
      .Case<TucExecutionOp>([&](auto op) {
        command.execution.subunit_bitmap = op.getSubunitBitmap();
        command.execution.context_id = op.getContextId();
        command.execution.target_context = op.getTargetContext();
        return std::make_tuple(command, registers);
      })
      .Case<TucWaitOp, TucWaitiOp>([&](auto op) {
        command.wait.dma_tag_id = op.getDmaTagId();
        command.wait.type = op.getType();
        command.wait.target_context = op.getTargetContext();
        return std::make_tuple(command, registers);
      })
      .Case<TucInterruptOp>(
          [&](auto op) { return std::make_tuple(command, registers); })
      .Case<TucDmaOp>([&](auto op) {
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
      .Case<TucDma1Op>([&](auto op) {
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
      .Case<TucDmawOp>([&](auto op) {
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
      .Case<TaskDmawOp>([&](auto op) {
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
      .Case<TucProfileOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.profile_0.profile_id = op.getProfileId();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<TucProfileiOp>([&](auto op) {
        command.profilei.profile_id = op.getProfileId();
        return std::make_tuple(command, registers);
      })
      .Case<TucPrflushOp>(
          [&](auto op) { return std::make_tuple(command, registers); })
      .Default([&](Operation *) {
        return op.emitOpError(
            "unable to interpret as furiosa dialect operator");
      });
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, TaskStaticSfrSubFetchOp> ||
                               std::is_same_v<T, TaskSfrSubFetchOp>,
                           bool> = true>
std::vector<sfr_data_t> getSfrSubFetch(T &op) {
  sfr::slice::SubFetchUnit<sfr_data_t> sfr =
      sfr::slice::SubFetchUnit<sfr_data_t>();
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
  auto custom_snoop_bitmap = op.getCustomSnoopBitmap();
  sfr.sub_fetch_unit_custom_snoop_bitmap0 =
      dyn_cast_or_null<IntegerAttr>(custom_snoop_bitmap[0]).getInt();
  sfr.sub_fetch_unit_custom_snoop_bitmap1 =
      dyn_cast_or_null<IntegerAttr>(custom_snoop_bitmap[1]).getInt();
  sfr.sub_fetch_unit_custom_snoop_bitmap2 =
      dyn_cast_or_null<IntegerAttr>(custom_snoop_bitmap[2]).getInt();
  sfr.sub_fetch_unit_custom_snoop_bitmap3 =
      dyn_cast_or_null<IntegerAttr>(custom_snoop_bitmap[3]).getInt();

  return sfr.get_blocks();
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, TaskStaticSfrSubCommitOp> ||
                               std::is_same_v<T, TaskSfrSubCommitOp>,
                           bool> = true>
std::vector<sfr_data_t> getSfrSubCommit(T &op) {
  sfr::slice::SubCommitUnit<sfr_data_t> sfr =
      sfr::slice::SubCommitUnit<sfr_data_t>();
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
  auto slice_enable_bitmap = op.getSliceEnableBitmap();
  sfr.sub_commit_unit_slice_enable_bitmap0 =
      dyn_cast_or_null<IntegerAttr>(slice_enable_bitmap[0]).getInt();
  sfr.sub_commit_unit_slice_enable_bitmap1 =
      dyn_cast_or_null<IntegerAttr>(slice_enable_bitmap[1]).getInt();
  sfr.sub_commit_unit_slice_enable_bitmap2 =
      dyn_cast_or_null<IntegerAttr>(slice_enable_bitmap[2]).getInt();
  sfr.sub_commit_unit_slice_enable_bitmap3 =
      dyn_cast_or_null<IntegerAttr>(slice_enable_bitmap[3]).getInt();

  return sfr.get_blocks();
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, TaskStaticSfrSubDataPathOp> ||
                               std::is_same_v<T, TaskSfrSubDataPathOp>,
                           bool> = true>
std::vector<sfr_data_t> getSfrSubDataPath(T &op) {
  sfr::slice::OperationDataPath<sfr_data_t> sfr =
      sfr::slice::OperationDataPath<sfr_data_t>();
  sfr.data_path_route_sub_context = op.getRoute();

  return sfr.get_blocks();
}

FailureOr<std::tuple<std::uint64_t, std::vector<sfr_data_t>>>
getStaticSfr(Operation &op) {
  return llvm::TypeSwitch<
             Operation *,
             FailureOr<std::tuple<std::uint64_t, std::vector<sfr_data_t>>>>(&op)
      .Case<TaskStaticSfrSubFetchOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrSubFetch(op));
      })
      .Case<TaskStaticSfrSubCommitOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrSubCommit(op));
      })
      .Case<TaskStaticSfrSubDataPathOp>([&](auto op) {
        return std::make_tuple<std::uint64_t, std::vector<sfr_data_t>>(
            op.getSfrAddr(), getSfrSubDataPath(op));
      })
      .Default([&](Operation *) {
        return op.emitOpError(
            "unable to interpret as furiosa dialect operator");
      });
}

FailureOr<std::vector<sfr_data_t>> getSfr(Operation &op) {
  return llvm::TypeSwitch<Operation *, FailureOr<std::vector<sfr_data_t>>>(&op)
      .Case<TaskSfrSubFetchOp>([&](auto op) { return getSfrSubFetch(op); })
      .Case<TaskSfrSubCommitOp>([&](auto op) { return getSfrSubCommit(op); })
      .Case<TaskSfrSubDataPathOp>(
          [&](auto op) { return getSfrSubDataPath(op); })
      .Default([&](Operation *) {
        return op.emitOpError(
            "unable to interpret as furiosa dialect operator");
      });
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, TaskStaticDmaDescriptorOp> ||
                               std::is_same_v<T, TaskDmaDescriptorOp>,
                           bool> = true>
FailureOr<TensorDmaDescriptor> getDmaDescriptor(T &op) {
  TensorDmaDescriptor descriptor{};
  descriptor.opcode = op.getOpcode();
  // descriptor.indirect = op.getIndirect();
  descriptor.source_base = op.getSourceBase();
  descriptor.destination_base = op.getDestinationBase();
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

} // namespace mlir::furiosa
