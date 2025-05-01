module {
  func.func @main() {
    %binary = furiosa_host.alloc { size = 0x100 }
    %arg0 = furiosa_host.alloc { size = 0x400 }
    %res0 = furiosa_host.alloc { size = 0x400 }
    %pe0 = furiosa_host.pe_program_load_inst { dram_address = 0x0, spm_address = 0x0, size = 0x100 }
    %pe1 = furiosa_host.pe_program_launch { spm_address = 0x0 }
    %pe = furiosa_host.pe_program_seq %pe0, %pe1
    %hal0 = furiosa_host.hal_program_write_at %binary { dram_address = 0x0 }
    %hal1 = furiosa_host.hal_program_write_at %arg0 { dram_address = 0x10000 }
    %hal2 = furiosa_host.hal_program_execute %pe
    %hal3 = furiosa_host.hal_program_read_at %res0 { dram_address = 0x20000 }
    %hal = furiosa_host.hal_program_seq %hal0, %hal1, %hal2, %hal3
    %dev = furiosa_host.device_new { target = #furiosa_host.target<npu 0 pe 0:0> }
    furiosa_host.device_execute %dev %hal
    return
  }
}
