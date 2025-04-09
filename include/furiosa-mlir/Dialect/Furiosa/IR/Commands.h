#pragma once

namespace mlir::furiosa {

struct TensorUnitCommand {
  TensorUnitCommand() : value(0) {}
  TensorUnitCommand(std::uint32_t opcode) : opcode(opcode) {}
  union {
    std::uint32_t value;
    struct {
      std::uint32_t operand : 25;
      std::uint32_t opcode : 7;
    };
    struct {
      std::uint32_t : 1;
      std::uint32_t r3 : 6;
      std::uint32_t r2 : 6;
      std::uint32_t r1 : 6;
      std::uint32_t r0 : 6;
      std::uint32_t opcode : 7;
    } reg;
    struct {
      std::uint32_t value : 16;
      std::uint32_t : 3;
      std::uint32_t r0 : 6;
      std::uint32_t opcode : 7;
    } itosfr;
    struct {
      std::uint32_t sfr_address : 16;
      std::uint32_t log_size : 2;
      std::uint32_t : 1;
      std::uint32_t r0 : 6;
      std::uint32_t opcode : 7;
    } rtosfri;
    struct {
      std::uint32_t value : 16;
      std::uint32_t : 3;
      std::uint32_t r0 : 6;
      std::uint32_t opcode : 7;
    } itosi;
    struct {
      std::uint32_t subunit_bitmap : 11;
      std::uint32_t : 5;
      std::uint32_t context_id : 1;
      std::uint32_t : 7;
      std::uint32_t target_context : 1;
      std::uint32_t opcode : 7;
    } execution;
    struct {
      std::uint32_t dma_tag_id : 6;
      std::uint32_t : 10;
      std::uint32_t type : 1;
      std::uint32_t : 7;
      std::uint32_t target_context : 1;
      std::uint32_t opcode : 7;
    } wait;
    struct {
      std::uint32_t profile_id : 16;
      std::uint32_t : 9;
      std::uint32_t opcode : 7;
    } profilei;
  };
  void setReg(std::uint32_t command_idx, std::uint32_t general_idx) {
    switch (command_idx) {
    case 0:
      this->reg.r0 = general_idx;
      break;
    case 1:
      this->reg.r1 = general_idx;
      break;
    case 2:
      this->reg.r2 = general_idx;
      break;
    case 3:
      this->reg.r3 = general_idx;
      break;
    }
  }
};

struct GeneralRegister {
  GeneralRegister() : value(0) {}
  union {
    std::uint64_t value;
    struct {
      std::uint64_t sfr_address : 25;
      std::uint64_t : 7;
      std::uint64_t size : 2;
      std::uint64_t : 30;
    } itosfr_0;
    struct {
      std::uint64_t value;
    } rtosfr_0;
    struct {
      std::uint64_t sfr_address : 25;
      std::uint64_t : 7;
      std::uint64_t size : 4;
      std::uint64_t : 28;
    } rtosfr_1;
    struct {
      std::uint32_t spm_address : 24;
      std::uint32_t size : 8;
      std::uint32_t sfr_address : 25;
      std::uint32_t : 7;
    } mtosfr_0;
    struct {
      std::uint64_t fetch_base : 22;
      std::uint64_t : 2;
      std::uint64_t fetch_size : 19;
      std::uint64_t : 5;
      std::uint64_t sfr_address : 16;
    } stosfr_0;
    struct {
      std::uint64_t topology : 8;
      std::uint64_t slice_log_size : 8;
      std::uint64_t dim0_log_size : 8;
      std::uint64_t dim1_log_size : 8;
      std::uint64_t data_offset : 8;
      std::uint64_t size : 16;
      std::uint64_t words_per_packet : 8;
    } stosfr_1;
    struct {
      std::uint64_t commit_base : 22;
      std::uint64_t : 2;
      std::uint64_t commit_limit : 16;
      std::uint64_t : 8;
      std::uint64_t sfr_address : 16;
    } sfrtos_0;
    struct {
      std::uint64_t cycle : 16;
      std::uint64_t : 48;
    } stall_0;
    struct {
      std::uint64_t address_begin : 30;
      std::uint64_t : 2;
      std::uint64_t address_end : 30;
      std::uint64_t : 2;
    } itos_0;
    struct {
      std::uint64_t value : 32;
      std::uint64_t dim1_log_size : 8;
      std::uint64_t : 24;
    } itos_1;
    struct {
      std::uint64_t limit1 : 16;
      std::uint64_t stride1 : 16;
      std::uint64_t : 32;
    } itos_2;
    struct {
      std::uint64_t address_begin : 30;
      std::uint64_t : 2;
      std::uint64_t address_end : 30;
      std::uint64_t : 2;
    } stos_0;
    struct {
      std::uint64_t destination_begin : 30;
      std::uint64_t : 2;
      std::uint64_t slice_log_size : 8;
      std::uint64_t dim1_log_size : 8;
      std::uint64_t words_per_packet : 8;
      std::uint64_t : 8;
    } stos_1;
    struct {
      std::uint64_t fetch_base : 22;
      std::uint64_t : 2;
      std::uint64_t fetch_limit : 19;
      std::uint64_t : 5;
      std::uint64_t tables : 4;
      std::uint64_t : 12;
    } stotab_0;
    struct {
      std::uint64_t commit_base : 22;
      std::uint64_t : 2;
      std::uint64_t topology : 8;
      std::uint64_t slice_log_size : 8;
      std::uint64_t dim0_log_size : 8;
      std::uint64_t dim1_log_size : 8;
      std::uint64_t words_per_packet : 8;
    } stotab_1;
    struct {
      std::uint64_t fetch_base : 22;
      std::uint64_t : 2;
      std::uint64_t fetch_limit : 19;
      std::uint64_t : 5;
      std::uint64_t type_conversion : 4;
      std::uint64_t write_mode : 2;
      std::uint64_t : 2;
      std::uint64_t zeropoint : 8;
    } stotrf_0;
    struct {
      std::uint64_t topology : 8;
      std::uint64_t slice_log_size : 8;
      std::uint64_t dim0_log_size : 8;
      std::uint64_t dim1_log_size : 8;
      std::uint64_t flits_per_packet : 8;
      std::uint64_t dim0_chunk_size : 8;
      std::uint64_t skip_flit_count : 10;
      std::uint64_t : 6;
    } stotrf_1;
    struct {
      std::uint64_t write_row_base : 12;
      std::uint64_t : 4;
      std::uint64_t write_row_count : 12;
      std::uint64_t write_mac_row : 4;
      std::uint64_t flits_per_period : 12;
      std::uint64_t : 4;
      std::uint64_t valid_flits_per_period : 12;
      std::uint64_t : 4;
    } stotrf_2;
    struct {
      std::uint64_t fetch_base : 22;
      std::uint64_t : 2;
      std::uint64_t fetch_limit : 19;
      std::uint64_t : 5;
      std::uint64_t type_conversion : 4;
      std::uint64_t : 12;
    } stovrf_0;
    struct {
      std::uint64_t topology : 8;
      std::uint64_t slice_log_size : 8;
      std::uint64_t dim0_log_size : 8;
      std::uint64_t dim1_log_size : 8;
      std::uint64_t words_per_packet : 16;
      std::uint64_t skip_flit_count : 16;
    } stovrf_1;
    struct {
      std::uint64_t write_row_base : 12;
      std::uint64_t : 4;
      std::uint64_t write_row_count : 9;
      std::uint64_t : 7;
      std::uint64_t write_row_stride : 8;
      std::uint64_t : 24;
    } stovrf_2;
    struct {
      std::uint64_t pe0_desc_addr : 32;
      std::uint64_t pe1_desc_addr : 32;
    } dma_0;
    struct {
      std::uint64_t pe2_desc_addr : 32;
      std::uint64_t pe3_desc_addr : 32;
    } dma_1;
    struct {
      std::uint64_t dma_tag_id : 6;
      std::uint64_t : 2;
      std::uint64_t profile : 1;
      std::uint64_t : 7;
      std::uint64_t profile_id : 16;
      std::uint64_t : 32;
    } dma_2;
    struct {
      std::uint64_t desc_addr : 24;
      std::uint64_t pe_valid_bitmap : 4;
      std::uint64_t : 4;
      std::uint64_t dma_tag_id : 5;
      std::uint64_t : 3;
      std::uint64_t profile : 1;
      std::uint64_t : 7;
      std::uint64_t profile_id : 16;
    } dma1_0;
    struct {
      std::uint64_t desc_addr : 40;
      std::uint64_t dma_tag_id : 6;
      std::uint64_t : 1;
      std::uint64_t profile : 1;
      std::uint64_t profile_id : 16;
    } dmaw_0;
    struct {
      std::uint64_t profile_id : 16;
      std::uint64_t : 48;
    } profile_0;
  };
};

static constexpr auto NUM_INDIRECT_INDICES = 32;
static constexpr auto DIMS = 8;
struct TensorDmaDescriptor {
  std::uint64_t opcode;
  struct {
    std::uint64_t dimension : 8;
    std::uint64_t entry_type : 1;
    std::uint64_t : 23;
    std::uint64_t indirect_descriptor_access_count : 24;
    std::uint64_t : 8;
  } indirect;
  std::uint64_t source_base;
  std::uint64_t destination_base;
  std::uint16_t source_limit[DIMS];
  std::int32_t source_stride[DIMS];
  std::uint16_t destination_limit[DIMS];
  std::int32_t destination_stride[DIMS];
  std::array<std::uint32_t, NUM_INDIRECT_INDICES> indirect_indices;
};

} // namespace mlir::furiosa
