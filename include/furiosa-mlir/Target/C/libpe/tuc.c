/**
 * Tensor Unit Controller Interface.
 * TODO: Rewrite in Rust (on PERT)
 */

#include <stdint.h>
#include <stdbool.h>

#include "printf.h"
#include "tuc.h"

#define SPM_END 0x380000

/* Invariant: at the start and end of every public API, max_usage <= TUC_COMMAND_QUEUE_SIZE - 2 */

static uint32_t max_usage(void);
static uint32_t distance_from_tail(uint32_t head);
static bool is_allocatable(int num_registers, int num_descs, int num_commands);
static void alloc_registers(tuc_register_t *registers, int num_registers);
static void commit_if_period_reached(int old_state);
static void wait_one_block(int old_state);
static void free_one_block(void);
static void commit_block(int old_state);
static void push_inner(tuc_command_t instruction);
static void interrupt_handler(uint32_t interrupt_index);
static void restore_interrupt(int old_state);
static int disable_interrupt(void);
static void check(bool cond, const char *cond_expr, const char *filename, int line);

#define CHECK(cond) check(cond, #cond, __FILE__, __LINE__)

/* Handcrafted built-in functions for freestanding binary. */
__attribute__((noreturn)) static inline void exit(uint8_t retcode)
{
    trampoline(((uint64_t)retcode) | TRAMPOLINE_EXIT, 0, 0);
    while (1)
    {
    }
}

static void check(bool cond, const char *cond_expr, const char *filename, int line)
{
    if (cond)
    {
        return;
    }

    char buf[2048], *payload = buf, *tail = buf;

    /* TODO: handle buffer overrun. */
    tail += sprintf(tail, "%s:%d: Assertion `%s` failed.\n", filename, line, cond_expr);
    tail += sprintf(tail, "Backtraces:\n");

    uint64_t link_register;
    __asm__ __volatile__("mov %[lr], x30" : [lr] "=r"(link_register));
    tail += sprintf(tail, "lr: 0x%lx\n", link_register);

    uint64_t *frame;
    __asm__ __volatile__("mov %[fp], x29" : [fp] "=r"(frame));

    for (int depth = 0; ((uint64_t) frame) < SPM_END && depth < 5; depth++)
    {
        link_register = *(frame + 1);
        tail += sprintf(tail, "[%d] lr: 0x%lx\n", depth, link_register);

        frame = (uint64_t *) *frame;
    }

    panic(payload);
}

/**
 * Laziest bound of queue head.
 *
 * Unlike tail, we cannot statically infer the exact value of head index. This is approximated,
 * by holding latest awaitted block's completion index.
 **/
static uint32_t min_head;
static uint32_t tail;

static tuc_register_bitmap_t free_registers, registers_to_be_freed;
static tuc_dma_desc_bitmap_t free_descs, descs_to_be_freed;
static tuc_timestamp_t synced_timestamp, timestamp_to_be_marked;

static uint32_t num_commits_to_be_sent;

static int block_head, block_count;

void tuc_init()
{
    min_head = *TUC_COMMAND_QUEUE_HEAD;
    tail = *TUC_COMMAND_QUEUE_TAIL;

    for (unsigned index = 0; index < TUC_COMMAND_QUEUE_SIZE; index++)
    {
        shared->blocks[index].interrupt_index = 0;
        shared->blocks[index].used_registers = 0;
        shared->blocks[index].used_descs = 0;
        shared->blocks[index].current_timestamp = 0;
    }

    block_head = 0;
    block_count = 0;

    free_registers = -1;
    registers_to_be_freed = 0;

    free_descs = -1;
    descs_to_be_freed = 0;

    synced_timestamp = 0;
    timestamp_to_be_marked = 0;

    for (unsigned index = 0; index < TUC_INTERRUPT_PERIOD; index++)
    {
        shared->commits_to_be_sent[index].address = 0;
        shared->commits_to_be_sent[index].size = 0;
    }
    num_commits_to_be_sent = 0;

    shared->pert_isr = interrupt_handler;
}

uint32_t tuc_alloc_resources(tuc_register_t *registers, int num_registers, volatile struct dma_desc_t **descs, int num_descs, int num_commands)
{
    int old_state = disable_interrupt();

    if (!is_allocatable(num_registers, num_descs, num_commands))
    {
        commit_block(old_state);

        do
        {
            wait_one_block(old_state);
        } while (!is_allocatable(num_registers, num_descs, num_commands));
    }

    alloc_registers(registers, num_registers);

    for (int index = 0; index < num_descs; index++)
    {
        int desc_index = __builtin_ctzl(free_descs);
        free_descs &= ~(1ul << desc_index);
        descs[index] = &shared->dma_desc_arena[desc_index][0];
    }

    restore_interrupt(old_state);
    return tail;
}

void tuc_alloc_registers(tuc_register_t *registers, int num_registers)
{
    int old_state = disable_interrupt();

    if (__builtin_popcountl(free_registers) < num_registers)
    {
        commit_block(old_state);

        do
        {
            wait_one_block(old_state);
        } while (__builtin_popcountl(free_registers) < num_registers);
    }

    alloc_registers(registers, num_registers);
    restore_interrupt(old_state);
}

void tuc_sync_timestamp(tuc_timestamp_t tuc_timestamp)
{
    int old_state = disable_interrupt();

    if (synced_timestamp < tuc_timestamp)
    {
        commit_block(old_state);

        do
        {
            wait_one_block(old_state);
        } while (synced_timestamp < tuc_timestamp);
    }

    restore_interrupt(old_state);
}

tuc_timestamp_t tuc_mark_timestamp()
{
    int old_state = disable_interrupt();
    tuc_timestamp_t tuc_timestamp = ++timestamp_to_be_marked;
    commit_block(old_state);
    restore_interrupt(old_state);

    return tuc_timestamp;
}

void tuc_free_register(tuc_register_t register_index)
{
    registers_to_be_freed |= 1ul << register_index;
}

void tuc_free_registers(tuc_register_t *registers, int num_registers)
{
    for (int index = 0; index < num_registers; index++)
    {
        registers_to_be_freed |= 1ul << registers[index];
    }
}

void tuc_free_resources(tuc_register_t *registers, int num_registers, uint32_t new_tail)
{
    tuc_free_registers(registers, num_registers);
    *TUC_COMMAND_QUEUE_TAIL = tail = new_tail;

    int old_state = disable_interrupt();
    commit_if_period_reached(old_state);

    restore_interrupt(old_state);
}

volatile struct dma_desc_t *tuc_alloc_dma_desc()
{
    int old_state = disable_interrupt();

    if (__builtin_popcountl(free_descs) == 0)
    {
        commit_block(old_state);

        do
        {
            wait_one_block(old_state);
        } while (__builtin_popcountl(free_descs) == 0);
    }

    int desc_index = __builtin_ctzl(free_descs);
    free_descs &= ~(1ul << desc_index);

    restore_interrupt(old_state);
    return &shared->dma_desc_arena[desc_index][0];
}

void tuc_free_dma_desc(volatile struct dma_desc_t *desc)
{
    int desc_index = (desc - &shared->dma_desc_arena[0][0]) / TDMA_DESC_PER_REQUEST;
    descs_to_be_freed |= 1ul << desc_index;
}

void tuc_commit_on_next_interrupt(uint64_t address, uint32_t size)
{
    shared->commits_to_be_sent[num_commits_to_be_sent].address = address;
    shared->commits_to_be_sent[num_commits_to_be_sent].size = size;
    num_commits_to_be_sent++;
}

/**
 * Enqueues a command into TUC command queue.
 */
void tuc_push(tuc_command_t command)
{
    int old_state = disable_interrupt();

    /**
     * -1 for a room of possible `interrupt`.
     * -1 to distinguish queue full from empty.
     */
    if (max_usage() >= TUC_COMMAND_QUEUE_SIZE - 2)
    {
        wait_one_block(old_state);
    }

    push_inner(command);
    commit_if_period_reached(old_state);

    restore_interrupt(old_state);
}

/* Waits all of enqueued commands. */
void tuc_wait()
{
    int old_state = disable_interrupt();
    if (max_usage() == 0)
    {
        restore_interrupt(old_state);
        return;
    }

    commit_block(old_state);

    while (block_count > 0)
    {
        wait_one_block(old_state);
    }

    restore_interrupt(old_state);
}

static uint32_t max_usage()
{
    return distance_from_tail(min_head);
}

static uint32_t distance_from_tail(uint32_t head)
{
    return (tail + TUC_COMMAND_QUEUE_SIZE - head) % TUC_COMMAND_QUEUE_SIZE;
}

static bool is_allocatable(int num_registers, int num_descs, int num_commands)
{
    return __builtin_popcountl(free_registers) >= num_registers && __builtin_popcountl(free_descs) >= num_descs && max_usage() + num_commands < TUC_COMMAND_QUEUE_SIZE - 1;
}

/**
 * Interrupt must be disabled before calling this function.
 */
static void alloc_registers(tuc_register_t *registers, int num_registers)
{
    for (int index = 0; index < num_registers; index++)
    {
        tuc_register_t register_index = __builtin_ctzl(free_registers);
        free_registers &= ~(1ul << register_index);
        registers[index] = register_index;
    }
}

/**
 * Interrupt must be disabled before calling this function.
 */
static void commit_if_period_reached(int old_state)
{
    uint32_t last_interrupt_index = (block_count == 0)
                                        ? min_head
                                        : shared->blocks[(block_head + block_count - 1) % TUC_COMMAND_QUEUE_SIZE].interrupt_index;

    if (distance_from_tail(last_interrupt_index) >= TUC_INTERRUPT_PERIOD)
    {
        commit_block(old_state);
    }
}

/**
 * Wait for interrupt handler to free at least one TUC block.
 * Interrupt must be disabled before calling this function.
 */
static void wait_one_block(int old_state)
{
    int old_block_count = block_count;
    CHECK(old_block_count > 0);

    do
    {
        // Restore interrrupt as we're leaving Task to PERT.
        restore_interrupt(old_state);

        // We're still waiting for TUC. Return TRAMPOLINE_WAIT_FOR_TUC to make PERT async future.
        trampoline(TRAMPOLINE_WAIT_FOR_TUC, 0, 0);

        // Disable interrupt here to make critical section for old_block_count and block_count.
        disable_interrupt();
    } while (old_block_count == block_count);
}

/**
 * Free one TUC block. Must not be called by other than interrupt handler.
 * Interrupt must be disabled before calling this function.
 */
static void free_one_block()
{
    free_registers |= shared->blocks[block_head].used_registers;
    free_descs |= shared->blocks[block_head].used_descs;
    tuc_timestamp_t current_timestamp = shared->blocks[block_head].current_timestamp;
    synced_timestamp = current_timestamp > synced_timestamp ? current_timestamp : synced_timestamp;
    min_head = (shared->blocks[block_head].interrupt_index + 1) % TUC_COMMAND_QUEUE_SIZE;

    for (uint32_t i = 0; i < shared->blocks[block_head].num_commits; i++)
    {
        commit(shared->blocks[block_head].commits[i].address, shared->blocks[block_head].commits[i].size);
    }

    block_head = (block_head + 1) % TUC_COMMAND_QUEUE_SIZE;
    block_count -= 1;
}

/**
 * Flush all reserved deallocations and commits into a TUC block.
 * Interrupt must be disabled before calling this function.
 */
static void commit_block(int old_state)
{
    /* If queue is full, committing a block invalidates queue state (tail crosses head). */
    while (max_usage() >= TUC_COMMAND_QUEUE_SIZE - 1)
    {
        wait_one_block(old_state);
    }

    CHECK((uint64_t) block_count < len(shared->blocks));

    int block_index = (block_head + block_count) % TUC_COMMAND_QUEUE_SIZE;

    shared->blocks[block_index].interrupt_index = tail;
    shared->blocks[block_index].used_registers = registers_to_be_freed;
    shared->blocks[block_index].used_descs = descs_to_be_freed;
    shared->blocks[block_index].current_timestamp = timestamp_to_be_marked;

    for (uint32_t i = 0; i < num_commits_to_be_sent; i++)
    {
        shared->blocks[block_index].commits[i] = shared->commits_to_be_sent[i];
    }
    shared->blocks[block_index].num_commits = num_commits_to_be_sent;

    registers_to_be_freed = 0;
    descs_to_be_freed = 0;
    num_commits_to_be_sent = 0;

    tuc_command_t interrupt;
    interrupt.op.opcode = TUC_COMMAND_INTERRUPT;
    push_inner(interrupt);

    block_count += 1;
}

/* ========================================= H/W Access ========================================= */

static void push_inner(tuc_command_t command)
{
    TUC_COMMAND_QUEUE_ENTRY[tail] = command.raw;
    tail = (tail + 1) % TUC_COMMAND_QUEUE_SIZE;
    *TUC_COMMAND_QUEUE_TAIL = tail;
}

static void interrupt_handler(__attribute__((unused)) uint32_t interrupt_index)
{
    /**
     * TODO: handle with lost interrupts.
     *
     * Under current design, interrupt cannot be enqueued more than 16 (interrupt queue depth),
     * since current tuc command does not exhaust all registers and descriptors in one command.
     *
     * While it's impractical to miss an interrupt, we should program defensively.
     */
    free_one_block();
}

/**
 * Disables interrupt and returns old interrupt state.
 */
static int disable_interrupt()
{
    int old_state = 0;
    __asm__ __volatile__("mrs %[old], daif" : [old] "=r"(old_state));

    int new_state = old_state | (1 << 6) | (1 << 7); /* disable IRQ and FIQ */
    __asm__ __volatile__("msr daif, %[new]" : : [new] "r"(new_state));

    return old_state;
}

static void restore_interrupt(int old_state)
{
    __asm__ __volatile__("msr daif, %[new]" : : [new] "r"(old_state));
}
