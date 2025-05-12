MEMORY
{
  RAM (rwx) : org = 0x0, l = 2M
}

SECTIONS
{
  .text : { 
    *(.text.main);
    *(.text*);
  }

  .data : { *(.data*) }

  .rodata : { *(.rodata*) }

  .bss (NOLOAD) :
  {
    . = ALIGN(8);
    __bss_start__ = .;
    *(.bss) *(COMMON)
    . = ALIGN(8);
    __bss_end__ = .;
  }

  /DISCARD/ : { *(.debug_*) *(.comment) *(.note*) *(.eh_frame) }
}
