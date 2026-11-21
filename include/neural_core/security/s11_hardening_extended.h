#ifndef SNEPPX_S11_HARDENING_EXTENDED_H
#define SNEPPX_S11_HARDENING_EXTENDED_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_CFI_MAX_INDIRECT_CALLS 65536
#define SNEPPX_CET_BITMAP_SIZE 4096
#define SNEPPX_BTI_MAX_TARGETS 16384
#define SNEPPX_MEMTAG_TAG_SIZE 4
#define SNEPPX_MEMTAG_GRANULE_SIZE 16

/*
 * SNEPPX - S11 Hardening Extended
 *
 * WHAT
 *   S11 Hardening Extended.
 *
 * CONCEPT
 *   Provides the S11 Hardening Extended.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_CFI_SHARED,
    SNEPPX_CFI_PURE_VCALL,
    SNEPPX_CFI_NON_VIRTUAL_CALL,
    SNEPPX_CFI_VIRTUAL_CALL,
    SNEPPX_CFI_NULL_POINTER
} SNEPPXCFICheckType;

typedef struct {
    uint64_t target_address;
    uint64_t expected_type_id;
    SNEPPXCFICheckType check_type;
    uint32_t vtable_offset;
    uint8_t validated : 1;
    uint8_t passthrough : 1;
} SNEPPXCFICheck;

typedef struct {
    uint32_t num_checks;
    SNEPPXCFICheck checks[SNEPPX_CFI_MAX_INDIRECT_CALLS];
    uint8_t shadow_table[SNEPPX_CFI_MAX_INDIRECT_CALLS * sizeof(uint64_t)];
    uint64_t cfi_hash_seed;
} SNEPPXCFIState;

typedef struct {
    uint64_t shadow_stack[4096];
    uint32_t shadow_stack_pointer;
    uint64_t return_address_cache[1024];
    uint32_t ra_cache_count;
    uint64_t last_known_ret;
} SNEPPXShadowStack;

typedef struct {
    uint8_t* tag_table;
    size_t tag_table_size;
    uint64_t tag_generation_key;
    uint32_t granule_shift;
    uint8_t fault_on_mismatch : 1;
    uint8_t async_tag_clear : 1;
} SNEPPXMemtagState;

typedef struct {
    uint64_t ibt_enabled : 1;
    uint64_t shstk_enabled : 1;
    uint64_t endbr64_present : 1;
    uint64_t legacy_compat : 1;
    uint64_t indirect_branch_tracking : 1;
    uint64_t suppress_rd_ssp : 1;
    uint64_t reserved : 58;
} SNEPPXCetState;

int snepx_cfi_init(SNEPPXCFIState* state, uint64_t seed);
int snepx_cfi_register_target(SNEPPXCFIState* state, uint64_t target, uint64_t type_id, SNEPPXCFICheckType check_type);
int snepx_cfi_validate_call(SNEPPXCFIState* state, uint64_t target, uint64_t type_id);
int snepx_cfi_validate_vtable_call(SNEPPXCFIState* state, void* obj, uint64_t vtable_offset);

int snepx_shadow_stack_init(SNEPPXShadowStack* sstack);
int snepx_shadow_stack_push(SNEPPXShadowStack* sstack, uint64_t return_addr);
int snepx_shadow_stack_pop(SNEPPXShadowStack* sstack, uint64_t* return_addr);
int snepx_shadow_stack_verify(SNEPPXShadowStack* sstack, uint64_t actual_return_addr);

int snepx_memtag_init(SNEPPXMemtagState* mtag, size_t heap_size);
uint8_t snepx_memtag_generate(SNEPPXMemtagState* mtag, const void* ptr);
int snepx_memtag_check(SNEPPXMemtagState* mtag, const void* ptr, uint8_t expected_tag);
int snepx_memtag_clear_async(SNEPPXMemtagState* mtag);
int snepx_memtag_set_fault_mode(SNEPPXMemtagState* mtag, uint8_t fault_on_mismatch);

int snepx_cet_enable(SNEPPXCetState* cet);
int snepx_cet_disable(SNEPPXCetState* cet);
int snepx_cet_verify_endbr(const void* target);
int snepx_cet_set_ibt_policy(SNEPPXCetState* cet, uint8_t track_indirect);

// Pointer authentication extensions
typedef struct {
    uint64_t pac_key_a[2];
    uint64_t pac_key_b[2];
    uint64_t pac_key_g[2];
    uint64_t pac_key_ga[2];
} SNEPPXPointerAuth;

int snepx_pac_init(SNEPPXPointerAuth* pa, const uint8_t* seed, size_t seed_len);
uint64_t snepx_pac_sign_a(SNEPPXPointerAuth* pa, uint64_t ptr, uint64_t context);
uint64_t snepx_pac_sign_b(SNEPPXPointerAuth* pa, uint64_t ptr, uint64_t context);
uint64_t snepx_pac_auth(SNEPPXPointerAuth* pa, uint64_t ptr, uint64_t context);
int snepx_pac_strip(SNEPPXPointerAuth* pa, uint64_t ptr);

// Branch Target Identification
int snepx_bti_register_landing(SNEPPXBTIState* bti, const void* target);
int snepx_bti_check_landing(SNEPPXBTIState* bti, const void* target);

// Intel PT / Branch tracing
typedef struct {
    uint64_t config_bits;
    size_t buffer_size;
    uint8_t* trace_buffer;
    uint64_t cycle_threshold;
    uint32_t addr_filter_a;
    uint32_t addr_filter_b;
} SNEPPXPtraceState;

int snepx_ptrace_init(SNEPPXPtraceState* pt, size_t buf_size);
int snepx_ptrace_enable(SNEPPXPtraceState* pt);
int snepx_ptrace_disable(SNEPPXPtraceState* pt, uint8_t* trace_out, size_t* trace_len);
int snepx_ptrace_analyze(SNEPPXPtraceState* pt, const uint8_t* trace, size_t trace_len);

#endif
