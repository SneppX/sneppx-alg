#ifndef SNEPPX_MEMORY_HARDENING_H
#define SNEPPX_MEMORY_HARDENING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
/*
 * SNEPPX - Memory Hardening
 *
 * WHAT
 *   Memory Hardening.
 *
 * CONCEPT
 *   Provides memory management.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

#define SNEPPX_MEM_GUARD_PAGE_SIZE 4096
#define SNEPPX_MEM_QUARANTINE_SIZE 128
#define SNEPPX_MAX_PAC_KEYS 16

/* Double-free / use-after-free detection */
typedef struct {
    void* entries[SNEPPX_MEM_QUARANTINE_SIZE];
    size_t sizes[SNEPPX_MEM_QUARANTINE_SIZE];
    uint64_t canaries[SNEPPX_MEM_QUARANTINE_SIZE];
    int count;
    int index;
} SNEPPXMemQuarantine;

/**
 * @brief Initialize Mem Quarantine.
 *
 * @param q [out] Q value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_mem_quarantine_init(SNEPPXMemQuarantine* q);
/**
 * @brief Add Mem Quarantine.
 *
 * @param q [out] Q value.
 * @param ptr [out] Ptr value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_mem_quarantine_add(SNEPPXMemQuarantine* q, void* ptr, size_t size);
/**
 * @brief Perform Mem Quarantine Check.
 *
 * @param q [out] Q value.
 * @param ptr [in] Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_mem_quarantine_check(SNEPPXMemQuarantine* q, const void* ptr);

/* Heap metadata encryption */
typedef struct {
    uint64_t xor_key;
    int enabled;
} SNEPPXHeapMetadataEncrypt;

/**
 * @brief Initialize Heap Metadata.
 *
 * @param hme [out] Hme value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_heap_metadata_init(SNEPPXHeapMetadataEncrypt* hme);
/**
 * @brief Encrypt Heap Metadata.
 *
 * @param hme [out] Hme value.
 * @param metadata [out] Metadata value.
 * @param len [in] Len value.
 */
void SNEPPX_heap_metadata_encrypt(SNEPPXHeapMetadataEncrypt* hme, void* metadata, size_t len);
/**
 * @brief Decrypt Heap Metadata.
 *
 * @param hme [out] Hme value.
 * @param metadata [out] Metadata value.
 * @param len [in] Len value.
 */
void SNEPPX_heap_metadata_decrypt(SNEPPXHeapMetadataEncrypt* hme, void* metadata, size_t len);

/* W^X enforcement */
/**
 * @brief Perform Mem Enforce Wx.
 *
 * @param addr [out] Addr value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_mem_enforce_wx(void* addr, size_t size);
/**
 * @brief Perform Mem Set Rx.
 *
 * @param addr [out] Addr value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_mem_set_rx(void* addr, size_t size);
/**
 * @brief Perform Mem Set Rw.
 *
 * @param addr [out] Addr value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_mem_set_rw(void* addr, size_t size);

/* Seccomp-BPF sandbox */
typedef struct {
    int enabled;
    int allow_read;
    int allow_write;
    int allow_open;
    int allow_socket;
    int allow_exec;
} SNEPPXSeccompConfig;

/**
 * @brief Initialize Seccomp.
 *
 * @param cfg [out] Cfg value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_seccomp_init(SNEPPXSeccompConfig* cfg);
/**
 * @brief Apply Seccomp.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_seccomp_apply(void);

/* Pointer authentication */
typedef struct {
    uint64_t pac_keys[SNEPPX_MAX_PAC_KEYS];
    int key_count;
} SNEPPXPAC;

/**
 * @brief Initialize Pac.
 *
 * @param pac [out] Pac value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_pac_init(SNEPPXPAC* pac);
/**
 * @brief Sign Pac.
 *
 * @param pac [out] Pac value.
 * @param pointer [in] Pointer value.
 * @param key_idx [in] Key Idx value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_pac_sign(SNEPPXPAC* pac, const void* pointer, int key_idx);
/**
 * @brief Verify Pac.
 *
 * @param pac [out] Pac value.
 * @param pointer [in] Pointer value.
 * @param signature [in] Signature value.
 * @param key_idx [in] Key Idx value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_pac_verify(SNEPPXPAC* pac, const void* pointer, uint64_t signature, int key_idx);

/* Control Flow Guard */
typedef struct {
    uintptr_t valid_targets[1024];
    int target_count;
} SNEPPXCFG;

/**
 * @brief Initialize Cfg.
 *
 * @param cfg [out] Cfg value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cfg_init(SNEPPXCFG* cfg);
/**
 * @brief Get Cfg Add Tar.
 *
 * @param cfg [out] Cfg value.
 * @param target [out] Target value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cfg_add_target(SNEPPXCFG* cfg, void* target);
/**
 * @brief Perform Cfg Validate.
 *
 * @param cfg [out] Cfg value.
 * @param target [out] Target value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cfg_validate(SNEPPXCFG* cfg, void* target);

/* Safe stack (shadow call stack) */
#define SNEPPX_SHADOW_STACK_DEPTH 256

typedef struct {
    uintptr_t stack[SNEPPX_SHADOW_STACK_DEPTH];
    int sp;
    int overflow_detected;
} SNEPPXShadowStack;

/**
 * @brief Initialize Shadow Stack.
 *
 * @param ss [out] Ss value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_shadow_stack_init(SNEPPXShadowStack* ss);
/**
 * @brief Perform Shadow Stack Push.
 *
 * @param ss [out] Ss value.
 * @param return_addr [in] Return Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_shadow_stack_push(SNEPPXShadowStack* ss, uintptr_t return_addr);
/**
 * @brief Perform Shadow Stack Pop.
 *
 * @param ss [out] Ss value.
 * @param return_addr [out] Return Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_shadow_stack_pop(SNEPPXShadowStack* ss, uintptr_t* return_addr);

/* Thread-local canary pool */
typedef struct {
    uint64_t canaries[64];
    int count;
} SNEPPXThreadCanaryPool;

/**
 * @brief Initialize Tls Canary Pool.
 *
 * @param pool [out] Pool value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_tls_canary_pool_init(SNEPPXThreadCanaryPool* pool);
/**
 * @brief Perform Tls Canary Alloc.
 *
 * @param pool [out] Pool value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_tls_canary_alloc(SNEPPXThreadCanaryPool* pool);
/**
 * @brief Perform Tls Canary Check.
 *
 * @param pool [out] Pool value.
 * @param canary [in] Canary value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_tls_canary_check(SNEPPXThreadCanaryPool* pool, uint64_t canary);

/* Guard page pool */
typedef struct {
    void* pages[64];
    size_t sizes[64];
    int count;
} SNEPPXGuardPagePool;

/**
 * @brief Initialize Guard Pool.
 *
 * @param pool [out] Pool value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_guard_pool_init(SNEPPXGuardPagePool* pool);
/**
 * @brief Perform Guard Pool Alloc.
 *
 * @param pool [out] Pool value.
 * @param size [in] Size value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_guard_pool_alloc(SNEPPXGuardPagePool* pool, size_t size);
/**
 * @brief Free Guard Pool.
 *
 * @param pool [out] Pool value.
 * @param ptr [out] Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_guard_pool_free(SNEPPXGuardPagePool* pool, void* ptr);

/* Memory pressure detection */
typedef struct {
    size_t total_allocated;
    size_t peak_allocated;
    size_t allocation_count;
    size_t allocation_limit;
    uint64_t last_warning_time;
    int pressure_level;
} SNEPPXMemPressure;

/**
 * @brief Initialize Mem Pressure.
 *
 * @param mp [out] Mp value.
 * @param limit [in] Limit value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_mem_pressure_init(SNEPPXMemPressure* mp, size_t limit);
/**
 * @brief Perform Mem Pressure Track.
 *
 * @param mp [out] Mp value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_mem_pressure_track(SNEPPXMemPressure* mp, size_t size);
/**
 * @brief Perform Mem Pressure Release.
 *
 * @param mp [out] Mp value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_mem_pressure_release(SNEPPXMemPressure* mp, size_t size);
/**
 * @brief Perform Mem Pressure Check.
 *
 * @param mp [out] Mp value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_mem_pressure_check(SNEPPXMemPressure* mp);

#ifdef __cplusplus
}
#endif
#endif
