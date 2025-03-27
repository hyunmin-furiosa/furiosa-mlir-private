#ifndef FURIOSA_DIALECT_UTILS_H
#define FURIOSA_DIALECT_UTILS_H

#include "furiosa-mlir/Dialect/Furiosa/IR/Commands.h"
#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"

namespace mlir::furiosa {

FailureOr<std::uint32_t> getOpcode(Operation &op) {
  return llvm::TypeSwitch<Operation *, FailureOr<std::uint32_t>>(&op)
      .Case<ItosfrOp>([&](auto op) { return 0x01; })
      .Case<RtosfrOp>([&](auto op) { return 0x02; })
      .Case<RtosfriOp>([&](auto op) { return 0x03; })
      .Case<MtosfrOp>([&](auto op) { return 0x04; })
      .Case<StosfrOp>([&](auto op) { return 0x05; })
      .Case<SfrtosOp>([&](auto op) { return 0x06; })
      .Case<StallOp>([&](auto op) { return 0x07; })
      .Case<ItosOp>([&](auto op) { return 0x08; })
      .Case<ItosiOp>([&](auto op) { return 0x09; })
      .Case<StosOp>([&](auto op) { return 0x0a; })
      .Case<StotabOp>([&](auto op) { return 0x0b; })
      .Case<StotrfOp>([&](auto op) { return 0x0c; })
      .Case<StovrfOp>([&](auto op) { return 0x0d; })
      .Case<ExecutionOp>([&](auto op) { return 0x10; })
      .Case<WaitOp>([&](auto op) { return 0x11; })
      .Case<WaitiOp>([&](auto op) { return 0x15; })
      .Case<InterruptOp>([&](auto op) { return 0x12; })
      .Case<DmaOp>([&](auto op) { return 0x13; })
      .Case<Dma1Op>([&](auto op) { return 0x14; })
      .Case<DmawOp>([&](auto op) { return 0x16; })
      .Case<ProfileOp>([&](auto op) { return 0x18; })
      .Case<ProfileiOp>([&](auto op) { return 0x19; })
      .Case<PrflushOp>([&](auto op) { return 0x1a; })
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
      .Case<ItosfrOp>([&](auto op) {
        command.itosfr.value = op.getValue();
        {
          GeneralRegister reg;
          reg.itosfr_0.sfr_address = op.getSfrAddress();
          reg.itosfr_0.size = op.getSize();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<RtosfrOp>([&](auto op) {
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
      .Case<RtosfriOp>([&](auto op) {
        command.rtosfri.log_size = op.getLogSize();
        command.rtosfri.sfr_address = op.getSfrAddress();
        {
          GeneralRegister reg;
          reg.rtosfr_0.value = op.getValue();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<MtosfrOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.mtosfr_0.spm_address = op.getSpmAddress();
          reg.mtosfr_0.size = op.getSize();
          reg.mtosfr_0.sfr_address = op.getSfrAddress();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<StosfrOp>([&](auto op) {
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
      .Case<SfrtosOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.sfrtos_0.commit_base = op.getCommitBase();
          reg.sfrtos_0.commit_limit = op.getCommitLimit();
          reg.sfrtos_0.sfr_address = op.getSfrAddress();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<StallOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.stall_0.cycle = op.getCycle();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<ItosOp>([&](auto op) {
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
      .Case<ItosiOp>([&](auto op) {
        command.value = op.getValue();
        {
          GeneralRegister reg;
          reg.itos_0.address_begin = op.getAddressBegin();
          reg.itos_0.address_end = op.getAddressEnd();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<StosOp>([&](auto op) {
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
      .Case<StotabOp>([&](auto op) {
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
      .Case<StotrfOp>([&](auto op) {
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
      .Case<StovrfOp>([&](auto op) {
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
      .Case<ExecutionOp>([&](auto op) {
        command.execution.subunit_bitmap = op.getSubunitBitmap();
        command.execution.context_id = op.getContextId();
        command.execution.target_context = op.getTargetContext();
        return std::make_tuple(command, registers);
      })
      .Case<WaitOp, WaitiOp>([&](auto op) {
        command.wait.dma_tag_id = op.getDmaTagId();
        command.wait.type = op.getType();
        command.wait.target_context = op.getTargetContext();
        return std::make_tuple(command, registers);
      })
      .Case<InterruptOp>(
          [&](auto op) { return std::make_tuple(command, registers); })
      .Case<DmaOp>([&](auto op) {
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
      .Case<Dma1Op>([&](auto op) {
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
      .Case<DmawOp>([&](auto op) {
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
      .Case<ProfileOp>([&](auto op) {
        {
          GeneralRegister reg;
          reg.profile_0.profile_id = op.getProfileId();
          registers.push_back(reg);
        }
        return std::make_tuple(command, registers);
      })
      .Case<ProfileiOp>([&](auto op) {
        command.profilei.profile_id = op.getProfileId();
        return std::make_tuple(command, registers);
      })
      .Case<PrflushOp>(
          [&](auto op) { return std::make_tuple(command, registers); })
      .Default([&](Operation *) {
        return op.emitOpError(
            "unable to interpret as furiosa dialect operator");
      });
}

} // namespace mlir::furiosa

#endif
