/* Tensor Unit Controller Interface */
#ifndef _TUC_H
#define _TUC_H

#include <assert.h>
#include <stdalign.h>

#include "span.h"

#define TRAMPOLINE_EXIT (0 << 8)
#define TRAMPOLINE_RECV_MESSAGE (1 << 8)
#define TRAMPOLINE_WAIT_FOR_TUC (2 << 8)

#define NUM_ITERATOR_INDEXERS 8

#define TUC_BASE UINT64_C(0x000C000000)

#define TUC_COMMAND_QUEUE_HEAD ((volatile uint64_t *)(TUC_BASE + 0x020))
#define TUC_COMMAND_QUEUE_TAIL ((volatile uint64_t *)(TUC_BASE + 0x028))
#define TUC_COMMAND_QUEUE_ENTRY ((volatile uint32_t *)(TUC_BASE + 0x100))
#define TUC_GENERAL_REGISTERS ((volatile uint64_t *)(TUC_BASE + 0x200))

#define TUC_COMMAND_QUEUE_SIZE 64
#define TUC_REGISTER_COUNT 64

#define TUC_COMMAND_WAIT 17
#define TUC_COMMAND_INTERRUPT 18
#define TUC_COMMAND_DMA 19
#define TUC_COMMAND_DMA_W 22

/* How often interrupts are enqueued. */
#define TUC_INTERRUPT_PERIOD 16

/* How many descriptors can be in one tensor dma request (= maximal number of
 * fusioned PE) */
#define TDMA_DESC_PER_REQUEST 4

#define SHARED_FIELDS_ADDR 0xEB000

#define SHARED_AREA_SIZE (1 << 12)

#define NUM_MAX_CLUSTERS 8

/**
 * Tensor DMA Descriptor
 */
struct dma_desc_t {
  /**
   * [1:0] : opcode (0: TensorLoop, 1: SourceIndirect, 2: DestinationIndirect)
   */
  alignas(256) uint64_t metadata;

  /**
   * [2:0] : dimension
   * [8:8] : indirect entry type
   *   (0: indirect address, 1: indirect list address)
   */
  uint64_t indirect_access;

  uint64_t source_base;
  uint64_t destination_base;

  uint16_t source_limits[NUM_ITERATOR_INDEXERS];
  int32_t source_strides[NUM_ITERATOR_INDEXERS];

  uint16_t destination_limits[NUM_ITERATOR_INDEXERS];
  int32_t destination_strides[NUM_ITERATOR_INDEXERS];

  int32_t indirect_indices[32];
};
static_assert(sizeof(struct dma_desc_t) == 256, "dma descriptor size mismatch");

typedef uint64_t tuc_dma_desc_bitmap_t;
static_assert(sizeof(tuc_dma_desc_bitmap_t) * 8 == TUC_COMMAND_QUEUE_SIZE,
              "dma desc count mismatch");

typedef uint64_t tuc_register_bitmap_t;
static_assert(sizeof(tuc_register_bitmap_t) * 8 == TUC_REGISTER_COUNT,
              "register count mismatch");

typedef uint32_t tuc_timestamp_t;

struct __attribute((packed)) commit_info_t {
  uint64_t address;
  uint32_t size;
};
static_assert(sizeof(struct commit_info_t) == 12);

/**
 * A set of TUC resources (registers and DMA descriptors), which are reserved as
 * operands of scheduled commands. They will be freed (i.e. commands get
 * completed) after the queue head reaches `interrupt_index`.
 *
 * You should enqueue `interrupt` command at `interrupt_index` to wait for
 * blocks being released.
 */
struct tuc_block_t {
  uint32_t interrupt_index;
  tuc_register_bitmap_t used_registers;
  tuc_dma_desc_bitmap_t used_descs;
  tuc_timestamp_t current_timestamp;

  struct commit_info_t commits[TUC_INTERRUPT_PERIOD];
  uint32_t num_commits;
};
static_assert(sizeof(struct tuc_block_t) == 224);

/**
 * Shared data structure between PERT and task, and between tasks (libpe)
 */
struct shared_field_t {
  // shared fields between tasks
  alignas(256) volatile uint8_t shared_area[SHARED_AREA_SIZE];
  struct dma_desc_t dma_desc_arena[TUC_COMMAND_QUEUE_SIZE]
                                  [TDMA_DESC_PER_REQUEST];
  struct tuc_block_t blocks[TUC_COMMAND_QUEUE_SIZE];
  struct commit_info_t commits_to_be_sent[TUC_INTERRUPT_PERIOD];
  uint64_t cluster_timestamps[NUM_MAX_CLUSTERS];
  uint64_t dummy[211];
  // shared fields between task and PERT
  void (*msg_send)(uint64_t, uint64_t, uint64_t);
  uint64_t tuc_profile_level;
  // `panic_message_ptr`, `panic_lr` is considered unstable implementation
  // detail
  const char *panic_message_ptr;
  uint64_t panic_lr;
  __attribute__((noreturn)) void (*panic)(const char *);
  uint64_t (*trampoline)(uint64_t, uint64_t, uint64_t);
  struct span_handle_t (*span_enter)(const char *, uint64_t);
  void (*span_leave)(struct span_handle_t);
  uint64_t npu_id;                       // F_FFD8
  uint64_t npu_vid;                      // F_FFE0
  uint64_t pe_id;                        // F_FFE8
  void (*commit_fn)(uint64_t, uint32_t); // F_FFF0
  void (*pert_isr)(uint32_t);            // F_FFF8
};
static_assert(sizeof(struct shared_field_t) == 0x15000,
              "size of shared fields is 0x15000");

/* Shared data structure */
static volatile struct shared_field_t *const shared =
    (struct shared_field_t *)SHARED_FIELDS_ADDR;

static inline void dma_desc_init(volatile struct dma_desc_t *desc) {
  for (int i = 0; i < NUM_ITERATOR_INDEXERS; i++) {
    desc->source_limits[i] = 1;
    desc->destination_limits[i] = 1;
    desc->source_strides[i] = 0;
    desc->destination_strides[i] = 0;
  }
}

static inline uint64_t trampoline(uint64_t arg0, uint64_t arg1, uint64_t arg2) {
  return shared->trampoline(arg0, arg1, arg2);
}

static inline void commit(uint64_t dram_address, uint32_t size) {
  shared->commit_fn(dram_address, size);
}

static inline struct span_handle_t span_enter(const char *name,
                                              uint64_t profile_uid) {
  return shared->span_enter(name, profile_uid);
}

static inline void span_leave(struct span_handle_t span) {
  shared->span_leave(span);
}

__attribute__((noreturn)) static inline void panic(const char *message) {
  shared->panic(message);
}

static inline int tuc_profile_enabled(uint64_t level) {
  return shared->tuc_profile_level >= level;
}

static inline void msg_send(uint64_t to_npu_id, uint64_t to_pe_id,
                            uint64_t payload) {
  shared->msg_send(to_npu_id, to_pe_id, payload);
}

struct peer_message_t {
  uint64_t core_id;
  uint64_t payload;
};

static inline struct peer_message_t msg_recv(int block) {
  struct peer_message_t (*msg_recv_trampoline)(uint64_t);
  msg_recv_trampoline = (void *)shared->trampoline;
  return msg_recv_trampoline(TRAMPOLINE_RECV_MESSAGE | (uint64_t)block);
}

typedef uint32_t tuc_register_t;

typedef union {
  struct {
    unsigned option : 1;
    unsigned operand3 : 6;
    unsigned operand2 : 6;
    unsigned operand1 : 6;
    unsigned operand0 : 6;
    unsigned opcode : 7;
  } op;

  uint32_t raw;
} tuc_command_t;

typedef union {
  uint64_t raw;
  struct {
    uint64_t pe0_desc_addr : 40;
    uint64_t dma_tag_id : 6;
    uint64_t _reserved : 1;
    uint64_t profile : 1;
    uint64_t profile_id : 16;
  };
} dma_w_operand0_t;

#define len(array) (sizeof(array) / sizeof(array[0]))

void tuc_init(void);

void tuc_alloc_registers(tuc_register_t *registers, int num_registers);
uint32_t tuc_alloc_resources(tuc_register_t *registers, int num_registers,
                             volatile struct dma_desc_t **descs, int num_descs,
                             int num_commands);

void tuc_free_register(tuc_register_t register_index);
void tuc_free_registers(tuc_register_t *registers, int num_registers);
void tuc_free_resources(tuc_register_t *registers, int num_registers,
                        uint32_t new_tail);

volatile struct dma_desc_t *tuc_alloc_dma_desc(void);
void tuc_free_dma_desc(volatile struct dma_desc_t *desc);

void tuc_sync_timestamp(tuc_timestamp_t tuc_timestamp);
tuc_timestamp_t tuc_mark_timestamp(void);

void tuc_commit_on_next_interrupt(uint64_t addr, uint32_t size);

/**
 * Enqueues a command into TUC command queue.
 */
void tuc_push(tuc_command_t command);

/**
 * Waits all of enqueued commands.
 */
void tuc_wait(void);

#endif // _TUC_H
