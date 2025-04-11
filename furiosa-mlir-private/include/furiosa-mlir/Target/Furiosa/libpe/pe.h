#ifndef _PE_H
#define _PE_H

#include <assert.h>
#include <printf.h>
#include <stdalign.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "tuc.h"

static inline void puts(char *str) {
  while (*str) {
    *((volatile char *)0xc000ff8) = *str++;
  }
  *((volatile char *)0xc000ff8) = '\n';
}

#define max(a, b) ((a) < (b) ? (b) : (a))
#define min(a, b) ((a) < (b) ? (a) : (b))

// Hardware configuration

#define SRAM_BASE UINT64_C(0x0010000000)
#define DRAM_BASE UINT64_C(0xC000000000)
#define TUC_SFR_BASE UINT64_C(0x000E000000)
#define TDMA_BASE UINT64_C(0x0000C00000)
#define REMOTE_PE_DATA_BASE UINT64_C(0x8000000000)
#define REMOTE_ENTRY_SIZE UINT64_C(0x100000000)
#define REMOTE_ENTRY_SIZE_PER_PE UINT64_C(0x20000000)

#define NUM_SLICES 64
#define SLICE_SPACE_SIZE UINT64_C(0x400000) /* 4MB */

#define TDMA_DESC_QUEUE_SUB_HEAD ((volatile uint64_t *)(TDMA_BASE + 0x040))
#define TDMA_DESC_QUEUE_SUB_TAIL ((volatile uint64_t *)(TDMA_BASE + 0x048))
#define TDMA_DESC_QUEUE_COMP_HEAD ((volatile uint64_t *)(TDMA_BASE + 0x050))
#define TDMA_DESC_QUEUE_COMP_TAIL ((volatile uint64_t *)(TDMA_BASE + 0x058))
#define TDMA_DESC_QUEUE_ENTRY ((volatile uint64_t *)(TDMA_BASE + 0x100))
#define TDMA_DESC_QUEUE_SIZE 32

#define TDMA_WORD_SIZE 32

/**
 * Flushes data cache to scratchpad memory.
 *
 * Given memory range, begin..begin + size must be accessed under volatile
 * qualifier to ensure stores to the range are written to the physical memory
 * (either cache or main data memory).
 *
 * This function only ensures modifications on cache will be applied to the data
 * memory.
 */
static inline void flush_cache(volatile void *begin, size_t size) {
  /**
   * Arm Cortex-A55 has L1 data cache which has 64-byte cache line length.
   * Refer to:
   * https://developer.arm.com/documentation/100442/0100/functional-description/level-1-memory-system/about-the-l1-memory-system
   */
  const size_t cache_line_size = 64;

  uintptr_t begin_aligned = (uintptr_t)begin & ~(cache_line_size - 1);

  /* inclusive */
  uintptr_t end_aligned =
      ((uintptr_t)begin + size - 1) & ~(cache_line_size - 1);

  /* Before flushing, ensure all previous stores are committed to cache. */
  __asm__ __volatile__("dsb sy");

  for (uintptr_t address = begin_aligned; address <= end_aligned;
       address += cache_line_size) {
    __asm__ __volatile__("dc cvac, %0" : : "r"(address));
  }

  /* Ensure that flushing cache completes. */
  __asm__ __volatile__("dsb sy");
}

/**
 * Invalidates data cache.
 *
 * Given memory range, begin..begin + size must be accessed under volatile
 * qualifier to ensure loads from the range are read from the physical memory
 * (either cache or main data memory).
 *
 * This function only ensures modifications on data memory will be applited to
 * cache.
 */
static inline void invalidate_cache(volatile void *begin, size_t size) {
  /**
   * Arm Cortex-A55 has L1 data cache which has 64-byte cache line length.
   * Refer to:
   * https://developer.arm.com/documentation/100442/0100/functional-description/level-1-memory-system/about-the-l1-memory-system
   */
  const size_t cache_line_size = 64;

  uintptr_t begin_aligned = (uintptr_t)begin & ~(cache_line_size - 1);

  /* inclusive */
  uintptr_t end_aligned =
      ((uintptr_t)begin + size - 1) & ~(cache_line_size - 1);

  /* Ensure that data invalidation of cache starts after all memory accesses are
   * finished. */
  __asm__ __volatile__("dsb sy");

  for (uintptr_t address = begin_aligned; address <= end_aligned;
       address += cache_line_size) {
    __asm__ __volatile__("dc ivac, %0" : : "r"(address));
  }

  /* Ensure that invalidation of cache completes. */
  __asm__ __volatile__("dsb sy");
}

static inline uint64_t get_base_raw(uint64_t npu_id, uint64_t pe_id) {
  return REMOTE_PE_DATA_BASE + REMOTE_ENTRY_SIZE * npu_id +
         REMOTE_ENTRY_SIZE_PER_PE * pe_id;
}

static inline uint64_t get_base_raw_dram(uint64_t npu_id, uint64_t num_npu) {
  return REMOTE_PE_DATA_BASE +
         REMOTE_ENTRY_SIZE * (num_npu + npu_id * (16 - num_npu) / num_npu);
}

/**
 * Return the base address of PE in self cluster. The `pe_offset` is offset in
 * self cluster, not the physical PE id.
 */
static inline uint64_t self_cluster_pe_base(uint64_t pe_offset) {
  // ensure that pe_id is in [0, 3]
  uint64_t pe_id = pe_offset & 0b11;
  uint64_t base = get_base_raw(shared->npu_vid, pe_id);
  return base;
}

/**
 * Return the base address of PE in other cluster in same NPU. The `pe_offset`
 * is offset in cluster, not the physical PE id.
 */
static inline uint64_t other_cluster_pe_base(uint64_t pe_offset) {
  // ensure that pe_id in [4, 7]
  uint64_t pe_id = 4 + (pe_offset & 0b11);
  uint64_t base = get_base_raw(shared->npu_vid, pe_id);
  return base;
}

/**
 * Return the base address of PE in other NPU.
 * - `npu_vid` should be different from shared->npu_vid
 * - `cluster_id` should be 0 or 1
 * - `pe_offset` should be in [0, 3]
 */
static inline uint64_t remote_pe_base(uint64_t npu_vid, uint64_t cluster_id,
                                      uint64_t pe_offset) {
  // ensure that pe_id in [0, 7]
  uint64_t pe_id = ((cluster_id & 0b1) * 4) + (pe_offset & 0b11);
  uint64_t base = get_base_raw(npu_vid, pe_id);
  return base;
}

/**
 * Enqueue a Tensor DMA request (a set of descriptors for each DMA engine) to
 * the queue and wait for its completion.
 *
 * Note: the descriptor must not be overwritten before its DMA starts. If it is
 * on stack or any other short-lived storages, `tuc_wait` should be called after
 * this function.
 */
static inline void run_tdma_bulk(volatile struct dma_desc_t *desc0,
                                 volatile struct dma_desc_t *desc1,
                                 volatile struct dma_desc_t *desc2,
                                 volatile struct dma_desc_t *desc3,
                                 uint8_t completion_id, bool profile,
                                 uint16_t profile_id) {
  tuc_register_t operands[4];
  tuc_command_t command;

  tuc_alloc_registers(operands, 4);

  /* Trigger DMA (only PE 0) */
  command.op.opcode = TUC_COMMAND_DMA_W;
  command.op.operand0 = operands[0];
  command.op.operand1 = operands[1];
  command.op.operand2 = operands[2];
  command.op.operand3 = operands[3];

  dma_w_operand0_t operand0;
  operand0.raw = 0;

  operand0.pe0_desc_addr = (uint64_t)desc0;
  operand0.dma_tag_id = completion_id;
  operand0.profile = profile;
  operand0.profile_id = profile_id;

  /* Remote base address of myself to make other fused PEs can read data from me
   */
  const uint64_t self_remote_base = self_cluster_pe_base(shared->pe_id);

  TUC_GENERAL_REGISTERS[operands[0]] = operand0.raw;
  TUC_GENERAL_REGISTERS[operands[1]] =
      ((uint64_t)desc1 == 0xffffffff) ? 0xffffffff
                                      : (self_remote_base + (uint64_t)desc1);
  TUC_GENERAL_REGISTERS[operands[2]] =
      ((uint64_t)desc2 == 0xffffffff) ? 0xffffffff
                                      : (self_remote_base + (uint64_t)desc2);
  TUC_GENERAL_REGISTERS[operands[3]] =
      ((uint64_t)desc3 == 0xffffffff) ? 0xffffffff
                                      : (self_remote_base + (uint64_t)desc3);
  tuc_push(command);

  tuc_free_registers(operands, 4);
}

/**
 * Enqueue a descriptor into Tensor DMA queue and wait for its completion.
 *
 * Note: the descriptor must not be overwritten before its DMA starts. If it is
 * on stack or any other short-lived storages, `tuc_wait` should be called after
 * this function.
 */
static inline void run_tdma_and_wait(volatile struct dma_desc_t *desc) {
  volatile struct dma_desc_t *invalid_dma_address =
      (volatile struct dma_desc_t *)0xffffffff;

  flush_cache(desc, sizeof(*desc));
  run_tdma_bulk(desc, invalid_dma_address, invalid_dma_address,
                invalid_dma_address, 0, false, 0);

  /* Wait for the DMA */
  tuc_command_t command;
  command.raw = 1 << 16; /* Wait for DMA id = 0 */
  command.op.opcode = TUC_COMMAND_WAIT;
  tuc_push(command);
}

/**
 * Initializes slices.
 */
static inline void configure_slice(uint8_t invalid_slice_id) {
  volatile uint8_t *address = (volatile uint8_t *)TUC_SFR_BASE;
  uint8_t logical_slice_id = 0;

  for (uint32_t index = 0; index < NUM_SLICES + 1; index++) {
    if (index == invalid_slice_id) {
      *address = 0x40;
    } else {
      *address = logical_slice_id++;
    }
    address += 0x10000;
  }
}

/* Transpiler support */

static inline int32_t add(int32_t a, int32_t b,
                          __attribute__((unused)) int option) {
  return a + b;
}

static inline int32_t and
    (int32_t a, int32_t b, __attribute__((unused)) int option) {
  return a & b;
}

static inline int32_t or
    (int32_t a, int32_t b, __attribute__((unused)) int option) {
  return a | b;
}

static inline int32_t xor
    (int32_t a, int32_t b, __attribute__((unused)) int option) { return a ^ b; }

    static inline int32_t shl(int32_t a, int32_t b, int option) {
  if (option == 0) {
    return a << b;
  } else {
    int32_t bit_len = 8 * sizeof(int32_t);
    int32_t threshold = (1 << (bit_len - 1 - b)) - 1;
    if (a > threshold) {
      return INT32_MAX;
    } else if (a < -threshold) {
      return INT32_MIN;
    } else {
      return a << b;
    }
  }
}

static inline int32_t shrl(int32_t a, int32_t b,
                           __attribute__((unused)) int option) {
  return (int32_t)((uint32_t)a >> b);
}

static inline int32_t shra(int32_t a, int32_t b, int option) {
  if (option == 0) {
    return a >> b;
  } else {
    int32_t mask = (1 << b) - 1;
    int32_t remainder = a & mask;

    int32_t threshhold = (mask >> 1) + (a < 0);
    return (a >> b) + (remainder > threshhold);
  }
}

static inline int32_t clz(int32_t a, __attribute__((unused)) int32_t b,
                          __attribute__((unused)) int option) {
  return __builtin_clz(a);
}

static inline int32_t sub(int32_t a, int32_t b,
                          __attribute__((unused)) int option) {
  return a - b;
}

static inline int32_t mul(int32_t a, int32_t b, int option) {
  int integer_multiply = (option & 2) == 2;
  int saturate_round = (option & 1) == 1;

  int nudge = saturate_round;

  if (integer_multiply) {
    return a * b;
  } else {
    // check overflow
    if (a == b && a == INT32_MIN) {
      return INT32_MAX;
    }
    int64_t input_product = (int64_t)a * (int64_t)b;
    return (((input_product >> 30) + nudge) >> 1);
  }
}

static inline int32_t avg(int32_t a, int32_t b,
                          __attribute__((unused)) int option) {
  int64_t sum = (int64_t)a + (int64_t)b;
  int64_t sign = sum > 0 ? 1 : -1;
  return (int32_t)((sum + sign) / 2);
}

typedef uint64_t fetchword_t;

static inline void run_parallel_dma(uint32_t sram_addr_begin,
                                    uint32_t sram_addr_end,
                                    uint32_t slice_space_size,
                                    uint64_t dram_addr_begin, bool is_stod) {
  uint32_t slice_index_begin = sram_addr_begin / slice_space_size;
  uint32_t slice_offset_begin = sram_addr_begin % slice_space_size;
  /* slice_index_end and slice_offset_end are inclusive */
  uint32_t slice_index_end = sram_addr_end / slice_space_size;
  uint32_t slice_offset_end = sram_addr_end % slice_space_size;
  uint32_t in_slice_bytes =
      (slice_offset_end - slice_offset_begin) + sizeof(fetchword_t);
  uint32_t height = in_slice_bytes / TDMA_WORD_SIZE;
  uint32_t width = slice_index_end - slice_index_begin + 1;

  volatile struct dma_desc_t desc[2];
  volatile struct dma_desc_t *desc_ptr[2];

  for (uint32_t pe_index = 0; pe_index < 2; pe_index++) {
    volatile struct dma_desc_t *pe_desc = &desc[pe_index];
    desc_ptr[pe_index] = pe_desc;

    uint32_t in_pe_slice_index_begin =
        max(pe_index * NUM_SLICES, slice_index_begin) % NUM_SLICES;
    uint32_t in_pe_slice_index_end =
        min((pe_index + 1) * NUM_SLICES - 1, slice_index_end) % NUM_SLICES;
    uint32_t in_pe_width = in_pe_slice_index_end - in_pe_slice_index_begin + 1;

    if (in_pe_width == 0) {
      desc_ptr[pe_index] = (volatile struct dma_desc_t *)0xffffffff;
      continue;
    }

    dma_desc_init(pe_desc);
    pe_desc->metadata = 0;

    uint64_t sram_base = SRAM_BASE + slice_offset_begin +
                         in_pe_slice_index_begin * SLICE_SPACE_SIZE;
    uint64_t dram_base =
        DRAM_BASE + dram_addr_begin +
        (in_pe_slice_index_begin + pe_index * NUM_SLICES - slice_index_begin) *
            in_slice_bytes;

    pe_desc->source_limits[0] = height;
    pe_desc->source_limits[1] = in_pe_width;
    pe_desc->source_strides[0] = TDMA_WORD_SIZE;

    pe_desc->destination_limits[0] = height;
    pe_desc->destination_limits[1] = in_pe_width;
    pe_desc->destination_strides[0] = TDMA_WORD_SIZE;

    if (is_stod) {
      pe_desc->source_base = sram_base;
      pe_desc->destination_base = dram_base;
      pe_desc->source_strides[1] = SLICE_SPACE_SIZE;
      pe_desc->destination_strides[1] = in_slice_bytes;
    } else {
      pe_desc->source_base = dram_base;
      pe_desc->destination_base = sram_base;
      pe_desc->source_strides[1] = in_slice_bytes;
      pe_desc->destination_strides[1] = SLICE_SPACE_SIZE;
    }

    flush_cache(pe_desc, sizeof(*pe_desc));
  }

  run_tdma_bulk(desc_ptr[0], desc_ptr[1],
                (volatile struct dma_desc_t *)0xffffffff,
                (volatile struct dma_desc_t *)0xffffffff, 0, false, 0);

  /* Wait for the DMA */
  tuc_command_t command;
  command.raw = 1 << 16; /* Wait for DMA id = 0 */
  command.op.opcode = TUC_COMMAND_WAIT;
  tuc_push(command);
  tuc_wait();

  if (is_stod) {
    uint32_t commit_size = width * in_slice_bytes;
    commit(dram_addr_begin, commit_size);
  }
}

/**
 * Copies memory from sram to dram using tensor DMA.
 * Assumes in-slice bytes are multiples of TDMA_WORD_SIZE (32).
 */
static inline void stod_p_dma(uint32_t source_begin, uint32_t source_end,
                              uint32_t slice_space_size, uint64_t destination) {
  run_parallel_dma(source_begin, source_end, slice_space_size, destination,
                   true);
}

/**
 * Copies memory from dram to sram using tensor DMA.
 * Assumes in-slice bytes are multiples of TDMA_WORD_SIZE (32).
 */
static inline void dtos_p_dma(uint32_t destination_begin,
                              uint32_t destination_end,
                              uint32_t slice_space_size, uint64_t source) {
  run_parallel_dma(destination_begin, destination_end, slice_space_size, source,
                   false);
}

#endif // _PE_H
