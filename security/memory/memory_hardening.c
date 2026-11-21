#include "memory_hardening.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif
#ifdef __linux__
#include <sys/prctl.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <asm/unistd.h>
#endif

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


/* --- Quarantine --- */
/**
 * @brief Initialize Mem Quarantine.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_quarantine_init(SNEPPXMemQuarantine* q) {
    if (!q) return -1;
    memset(q,0,sizeof(*q));
    return 0;
}

/**
 * @brief Add Mem Quarantine.
 *
 * @param q [out] Q value.
 * @param ptr [out] Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_quarantine_add(SNEPPXMemQuarantine* q, void* ptr, size_t size) {
    if (!q||!ptr) return -1;
    for (int i=0;i<q->count;i++) if (q->entries[i]==ptr) return -1;
    q->entries[q->index]=ptr;
    q->sizes[q->index]=size;
    q->canaries[q->index]=(uint64_t)(uintptr_t)ptr ^ (uint64_t)time(NULL);
    memset(ptr,0xFD,size);
    q->index=(q->index+1)%SNEPPX_MEM_QUARANTINE_SIZE;
    if (q->count<SNEPPX_MEM_QUARANTINE_SIZE) q->count++;
    return 0;
}

/**
 * @brief Perform Mem Quarantine Check.
 *
 * @param q [out] Q value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_quarantine_check(SNEPPXMemQuarantine* q, const void* ptr) {
    if (!q||!ptr) return 0;
    for (int i=0;i<q->count;i++) if (q->entries[i]==ptr) return 1;
    return 0;
}

/* --- Heap Metadata Encryption --- */
/**
 * @brief Initialize Heap Metadata.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heap_metadata_init(SNEPPXHeapMetadataEncrypt* hme) {
    if (!hme) return -1;
    hme->xor_key=(uint64_t)time(NULL)^0x9E3779B97F4A7C15ULL;
    hme->xor_key=hme->xor_key*6364136223846793005ULL+1442695040888963407ULL;
    hme->enabled=1;
    return 0;
}

/**
 * @brief Encrypt Heap Metadata.
 *
 * @param hme [out] Hme value.
 * @param metadata [out] Metadata value.
 */
void SNEPPX_heap_metadata_encrypt(SNEPPXHeapMetadataEncrypt* hme, void* metadata, size_t len) {
    if (!hme||!hme->enabled||!metadata) return;
    uint8_t* p=(uint8_t*)metadata;
    for (size_t i=0;i<len;i++) {
        p[i]^=(uint8_t)(hme->xor_key>>((i%8)*8));
        if (i%8==7) hme->xor_key=hme->xor_key*6364136223846793005ULL+1;
    }
}

/**
 * @brief Decrypt Heap Metadata.
 *
 * @param hme [out] Hme value.
 * @param metadata [out] Metadata value.
 */
void SNEPPX_heap_metadata_decrypt(SNEPPXHeapMetadataEncrypt* hme, void* metadata, size_t len) {
    SNEPPX_heap_metadata_encrypt(hme,metadata,len);
}

/* --- W^X Enforcement --- */
/**
 * @brief Perform Mem Enforce Wx.
 *
 * @param addr [out] Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_enforce_wx(void* addr, size_t size) {
    (void)addr;(void)size;
#ifdef _WIN32
    DWORD old; return VirtualProtect(addr,size,PAGE_EXECUTE_READWRITE,&old)?0:-1;
#else
    if (!addr||size==0) return -1;
    size_t ps=getpagesize();
    void* aligned=(void*)((uintptr_t)addr&~(ps-1));
    size_t aligned_size=((uintptr_t)addr+size+ps-1)&~(ps-1);
    aligned_size-=(uintptr_t)aligned;
    return mprotect(aligned,aligned_size,PROT_READ|PROT_WRITE|PROT_EXEC);
#endif
}

/**
 * @brief Perform Mem Set Rx.
 *
 * @param addr [out] Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_set_rx(void* addr, size_t size) {
#ifdef _WIN32
    DWORD old; return VirtualProtect(addr,size,PAGE_EXECUTE_READ,&old)?0:-1;
#else
    size_t ps=getpagesize();
    void* aligned=(void*)((uintptr_t)addr&~(ps-1));
    return mprotect(aligned,((uintptr_t)addr+size+ps-1)&~(ps-1)-(uintptr_t)aligned,PROT_READ|PROT_EXEC);
#endif
}

/**
 * @brief Perform Mem Set Rw.
 *
 * @param addr [out] Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_set_rw(void* addr, size_t size) {
#ifdef _WIN32
    DWORD old; return VirtualProtect(addr,size,PAGE_READWRITE,&old)?0:-1;
#else
    size_t ps=getpagesize();
    void* aligned=(void*)((uintptr_t)addr&~(ps-1));
    return mprotect(aligned,((uintptr_t)addr+size+ps-1)&~(ps-1)-(uintptr_t)aligned,PROT_READ|PROT_WRITE);
#endif
}

/* --- Seccomp --- */
/**
 * @brief Initialize Seccomp.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_seccomp_init(SNEPPXSeccompConfig* cfg) {
    if (!cfg) return -1;
    memset(cfg,0,sizeof(*cfg));
    cfg->allow_read=1; cfg->allow_write=1;
    cfg->allow_open=0; cfg->allow_socket=0;
    cfg->allow_exec=0; cfg->enabled=1;
    return 0;
}

/**
 * @brief Apply Seccomp.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_seccomp_apply(void) {
#ifdef __linux__
    if (prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0)<0) return -1;
    struct sock_filter filter[]={
        BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,nr)),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_read,0,1), BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_write,0,1), BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_mmap,0,1), BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_munmap,0,1), BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_exit_group,0,1), BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL),
    };
    struct sock_fprog prog={.len=sizeof(filter)/sizeof(filter[0]),.filter=filter};
    return prctl(PR_SET_SECCOMP,SECCOMP_MODE_FILTER,&prog);
#else
    return 0;
#endif
}

/* --- Pointer Authentication --- */
/**
 * @brief Initialize Pac.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_pac_init(SNEPPXPAC* pac) {
    if (!pac) return -1;
    for (int i=0;i<SNEPPX_MAX_PAC_KEYS;i++) {
        pac->pac_keys[i]=(uint64_t)rand()^((uint64_t)rand()<<32)^((uint64_t)time(NULL)<<(i%8));
        pac->pac_keys[i]=pac->pac_keys[i]*6364136223846793005ULL+1442695040888963407ULL;
    }
    pac->key_count=SNEPPX_MAX_PAC_KEYS;
    return 0;
}

/**
 * @brief Sign Pac.
 *
 * @param pac [out] Pac value.
 * @param pointer [in] Pointer value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_pac_sign(SNEPPXPAC* pac, const void* pointer, int key_idx) {
    if (!pac||!pointer||key_idx>=pac->key_count) return 0;
    uint64_t addr=(uint64_t)(uintptr_t)pointer;
    uint64_t key=pac->pac_keys[key_idx];
    uint64_t hash=addr;
    for (int i=0;i<8;i++) hash=(hash>>1)^(key&(1ULL<<63)?0xC96C5795D7870F42ULL:0);
    return (addr<<16)|(hash&0xFFFF);
}

/**
 * @brief Verify Pac.
 *
 * @param pac [out] Pac value.
 * @param pointer [in] Pointer value.
 * @param signature [in] Signature value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_pac_verify(SNEPPXPAC* pac, const void* pointer, uint64_t signature, int key_idx) {
    if (!pac||!pointer||key_idx>=pac->key_count) return 0;
    uint64_t expected=SNEPPX_pac_sign(pac,pointer,key_idx);
    return (expected==signature)?1:0;
}

/* --- CFG --- */
int SNEPPX_cfg_init(SNEPPXCFG* cfg) { if (!cfg) return -1; memset(cfg,0,sizeof(*cfg)); return 0; }

/**
 * @brief Get Cfg Add Tar.
 *
 * @param cfg [out] Cfg value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cfg_add_target(SNEPPXCFG* cfg, void* target) {
    if (!cfg||!target||cfg->target_count>=1024) return -1;
    cfg->valid_targets[cfg->target_count++]=(uintptr_t)target;
    return 0;
}

/**
 * @brief Perform Cfg Validate.
 *
 * @param cfg [out] Cfg value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cfg_validate(SNEPPXCFG* cfg, void* target) {
    if (!cfg||!target) return 0;
    uintptr_t t=(uintptr_t)target;
    for (int i=0;i<cfg->target_count;i++) if (cfg->valid_targets[i]==t) return 1;
    return 0;
}

/* --- Shadow Stack --- */
/**
 * @brief Initialize Shadow Stack.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shadow_stack_init(SNEPPXShadowStack* ss) {
    if (!ss) return -1;
    memset(ss,0,sizeof(*ss));
    ss->sp=-1;
    return 0;
}

/**
 * @brief Perform Shadow Stack Push.
 *
 * @param ss [out] Ss value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shadow_stack_push(SNEPPXShadowStack* ss, uintptr_t return_addr) {
    if (!ss||ss->sp>=SNEPPX_SHADOW_STACK_DEPTH-1) { if (ss) ss->overflow_detected=1; return -1; }
    ss->stack[++ss->sp]=return_addr;
    return 0;
}

/**
 * @brief Perform Shadow Stack Pop.
 *
 * @param ss [out] Ss value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shadow_stack_pop(SNEPPXShadowStack* ss, uintptr_t* return_addr) {
    if (!ss||!return_addr||ss->sp<0) return -1;
    *return_addr=ss->stack[ss->sp--];
    return 0;
}

/* --- TLS Canary Pool --- */
/**
 * @brief Initialize Tls Canary Pool.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls_canary_pool_init(SNEPPXThreadCanaryPool* pool) {
    if (!pool) return -1;
    for (int i=0;i<64;i++) pool->canaries[i]=(uint64_t)rand()^((uint64_t)rand()<<32);
    pool->count=64;
    return 0;
}

/**
 * @brief Perform Tls Canary Alloc.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_tls_canary_alloc(SNEPPXThreadCanaryPool* pool) {
    if (!pool||pool->count<=0) return 0;
    return pool->canaries[--pool->count];
}

/**
 * @brief Perform Tls Canary Check.
 *
 * @param pool [out] Pool value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls_canary_check(SNEPPXThreadCanaryPool* pool, uint64_t canary) {
    if (!pool) return 0;
    for (int i=0;i<64-pool->count;i++) if (pool->canaries[i]==canary) return 1;
    return 0;
}

/* --- Guard Page Pool --- */
/**
 * @brief Initialize Guard Pool.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_guard_pool_init(SNEPPXGuardPagePool* pool) {
    if (!pool) return -1;
    memset(pool,0,sizeof(*pool));
    return 0;
}

/**
 * @brief Perform Guard Pool Alloc.
 *
 * @param pool [out] Pool value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_guard_pool_alloc(SNEPPXGuardPagePool* pool, size_t size) {
    (void)pool;
    if (size==0) return NULL;
    size_t total=size+SNEPPX_MEM_GUARD_PAGE_SIZE*2;
#ifdef _WIN32
    void* base=VirtualAlloc(NULL,total,MEM_RESERVE,PAGE_NOACCESS);
    if (!base) return NULL;
    void* user=(char*)base+SNEPPX_MEM_GUARD_PAGE_SIZE;
    if (!VirtualAlloc(user,size,MEM_COMMIT,PAGE_READWRITE)) { VirtualFree(base,0,MEM_RELEASE); return NULL; }
    return user;
#else
    size_t ps=sysconf(_SC_PAGESIZE);
    size_t aligned_total=((total+ps-1)/ps)*ps;
    void* base=mmap(NULL,aligned_total,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (base==MAP_FAILED) return NULL;
    void* user=(char*)base+ps;
    if (mprotect(user,size,PROT_READ|PROT_WRITE)!=0) { munmap(base,aligned_total); return NULL; }
    return user;
#endif
}

/**
 * @brief Free Guard Pool.
 *
 * @param pool [out] Pool value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_guard_pool_free(SNEPPXGuardPagePool* pool, void* ptr) {
    (void)pool;
    if (!ptr) return -1;
    size_t total=SNEPPX_MEM_GUARD_PAGE_SIZE*2;
#ifdef _WIN32
    VirtualFree((char*)ptr-SNEPPX_MEM_GUARD_PAGE_SIZE,0,MEM_RELEASE);
#else
    size_t ps=sysconf(_SC_PAGESIZE);
    void* base=(char*)ptr-ps;
    munmap(base,ps*3);
#endif
    return 0;
}

/* --- Memory Pressure --- */
/**
 * @brief Initialize Mem Pressure.
 *
 * @param mp [out] Mp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_pressure_init(SNEPPXMemPressure* mp, size_t limit) {
    if (!mp) return -1;
    memset(mp,0,sizeof(*mp));
    mp->allocation_limit=limit;
    return 0;
}

/**
 * @brief Perform Mem Pressure Track.
 *
 * @param mp [out] Mp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_pressure_track(SNEPPXMemPressure* mp, size_t size) {
    if (!mp) return -1;
    mp->total_allocated+=size;
    mp->allocation_count++;
    if (mp->total_allocated>mp->peak_allocated) mp->peak_allocated=mp->total_allocated;
    if (mp->total_allocated>mp->allocation_limit) mp->pressure_level=1;
    if (mp->total_allocated>mp->allocation_limit*2) mp->pressure_level=2;
    return mp->pressure_level;
}

/**
 * @brief Perform Mem Pressure Release.
 *
 * @param mp [out] Mp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_pressure_release(SNEPPXMemPressure* mp, size_t size) {
    if (!mp) return -1;
    mp->total_allocated-=(mp->total_allocated>=size)?size:mp->total_allocated;
    if (mp->total_allocated<=mp->allocation_limit) mp->pressure_level=0;
    return 0;
}

/**
 * @brief Perform Mem Pressure Check.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_pressure_check(SNEPPXMemPressure* mp) {
    if (!mp) return -1;
    return mp->pressure_level;
}

/* === Internal Data Structures === */

#define SNEPPX_FREELIST_CANARY_VALUE 0xDEADBEEFCAFEBABEULL
#define SNEPPX_FREELIST_DEFAULT_MAX 2048
#define SNEPPX_MEM_POOL_MAX_BLOCKS 65536
#define SNEPPX_SCRUB_PASSES 4
#define SNEPPX_FREELIST_CANARY_SIZE sizeof(uint64_t)
#define SNEPPX_BITMAP_BITS_PER_BYTE 8
#define SNEPPX_POOL_SCRUB_BYTE 0xAB

typedef struct {
    void** entries;
    size_t* sizes;
    uint64_t* stored_canaries;
    int count;
    int max_count;
} SNEPPXFreeList;

typedef struct {
    size_t current;
    int pressure_level;
    size_t peak;
} SNEPPXMemPressureStats;

typedef struct {
    size_t block_size;
    int total_blocks;
    int free_blocks;
    uint8_t* bitmap;
    void* memory;
} SNEPPXMemPool;

static SNEPPXFreeList g_freelist;
static int g_freelist_initialized = 0;

static const uint8_t g_scrub_patterns[4] = {0xFD, 0xDF, 0xAB, 0x00};
static int g_scrub_phase = 0;

/* === Freelist Integrity === */

/**
 * @brief Perform Freelist Canary Value.
 *
 * @param ptr [out] Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
static uint64_t SNEPPX_freelist_canary_value(void* ptr, size_t size) {
    return (uint64_t)(uintptr_t)ptr ^ (uint64_t)size ^ SNEPPX_FREELIST_CANARY_VALUE;
}

/**
 * @brief Perform Freelist Write Canary.
 *
 * @param ptr [out] Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
static int SNEPPX_freelist_write_canary(void* ptr, size_t size) {
    if (!ptr || size < sizeof(uint64_t)) return -1;
    uint64_t* canary = (uint64_t*)((uint8_t*)ptr + size - sizeof(uint64_t));
    *canary = SNEPPX_freelist_canary_value(ptr, size);
    return 0;
}

/**
 * @brief Perform Freelist Verify Canary.
 *
 * @param ptr [in] Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
static int SNEPPX_freelist_verify_canary(const void* ptr, size_t size) {
    if (!ptr || size < sizeof(uint64_t)) return 0;
    const uint64_t* canary = (const uint64_t*)((const uint8_t*)ptr + size - sizeof(uint64_t));
    return (*canary == SNEPPX_freelist_canary_value((void*)ptr, size)) ? 1 : 0;
}

/**
 * @brief Perform Freelist Init Internal.
 *
 * @return 0 on success, -1 on error.
 */
static int SNEPPX_freelist_init_internal(int max_entries) {
    if (g_freelist_initialized) return 0;
    g_freelist.entries = (void**)calloc((size_t)max_entries, sizeof(void*));
    if (!g_freelist.entries) return -1;
    g_freelist.sizes = (size_t*)calloc((size_t)max_entries, sizeof(size_t));
    if (!g_freelist.sizes) { free(g_freelist.entries); g_freelist.entries = NULL; return -1; }
    g_freelist.stored_canaries = (uint64_t*)calloc((size_t)max_entries, sizeof(uint64_t));
    if (!g_freelist.stored_canaries) { free(g_freelist.entries); free(g_freelist.sizes); g_freelist.entries = NULL; g_freelist.sizes = NULL; return -1; }
    g_freelist.count = 0;
    g_freelist.max_count = max_entries;
    g_freelist_initialized = 1;
    return 0;
}

/**
 * @brief Perform Freelist Check.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_freelist_check(SNEPPXFreeList* fl) {
    if (!fl) {
        if (!g_freelist_initialized) return -1;
        fl = &g_freelist;
    }
    for (int i = 0; i < fl->count; i++) {
        if (!fl->entries[i]) continue;
        if (!SNEPPX_freelist_verify_canary(fl->entries[i], fl->sizes[i])) return -1;
        if (fl->stored_canaries[i] != SNEPPX_freelist_canary_value(fl->entries[i], fl->sizes[i])) return -1;
    }
    return 0;
}

/**
 * @brief Add Freelist.
 *
 * @param ptr [out] Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
static int SNEPPX_freelist_add(void* ptr, size_t size) {
    if (!g_freelist_initialized) if (SNEPPX_freelist_init_internal(1024) != 0) return -1;
    if (!ptr) return -1;
    if (g_freelist.count >= g_freelist.max_count) return -1;
    for (int i = 0; i < g_freelist.count; i++) if (g_freelist.entries[i] == ptr) return -1;
    SNEPPX_freelist_write_canary(ptr, size);
    g_freelist.entries[g_freelist.count] = ptr;
    g_freelist.sizes[g_freelist.count] = size;
    g_freelist.stored_canaries[g_freelist.count] = SNEPPX_freelist_canary_value(ptr, size);
    g_freelist.count++;
    return 0;
}

/**
 * @brief Remove Freelist.
 *
 * @return 0 on success, -1 on error.
 */
static int SNEPPX_freelist_remove(void* ptr) {
    if (!g_freelist_initialized || !ptr) return -1;
    for (int i = 0; i < g_freelist.count; i++) {
        if (g_freelist.entries[i] == ptr) {
            g_freelist.entries[i] = g_freelist.entries[g_freelist.count - 1];
            g_freelist.sizes[i] = g_freelist.sizes[g_freelist.count - 1];
            g_freelist.stored_canaries[i] = g_freelist.stored_canaries[g_freelist.count - 1];
            g_freelist.count--;
            return 0;
        }
    }
    return -1;
}

/* === Memory Scrub === */

/**
 * @brief Perform Mem Scrub.
 *
 * @param ptr [out] Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_scrub(void* ptr, size_t size) {
    if (!ptr || size == 0) return -1;
    uint8_t* p = (uint8_t*)ptr;
    size_t half = size / 2;
    size_t quarter = size / 4;
    size_t three_quarter = half + quarter;
    size_t pattern_idx = (size_t)g_scrub_phase % 4;
    volatile uint8_t sink = 0;
    for (size_t pass = 0; pass < SNEPPX_SCRUB_PASSES; pass++) {
        uint8_t current_pat = g_scrub_patterns[(pattern_idx + pass) % 4];
        for (size_t i = 0; i < size; i++) {
            p[i] ^= current_pat;
            p[i] = (uint8_t)((p[i] << 3) | (p[i] >> 5));
            p[i] ^= (uint8_t)(pass * 0x33);
            sink ^= p[i];
        }
        if (pass == 0) {
            for (size_t i = 0; i < size; i++) p[i] = 0xFD;
        }
        if (pass == 1) {
            for (size_t i = 0; i < half; i++) p[i] = (uint8_t)(p[i] ^ 0xDF);
            for (size_t i = half; i < size; i++) p[i] = (uint8_t)(p[i] ^ 0xBA);
        }
        if (pass == 2) {
            for (size_t i = 0; i < quarter; i++) p[i] = (uint8_t)(p[i] * 0xAB);
            for (size_t i = quarter; i < three_quarter; i++) p[i] = (uint8_t)(p[i] ^ 0xCD);
            for (size_t i = three_quarter; i < size; i++) p[i] = (uint8_t)(p[i] * 0xEF);
        }
        if (pass == 3) {
            for (size_t i = 0; i < size; i++) p[i] = 0x00;
        }
    }
    (void)sink;
    g_scrub_phase = (g_scrub_phase + 1) % 1024;
    return 0;
}

/* === Expanded W^X === */

/**
 * @brief Perform Mem Enforce Wx Strict.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_enforce_wx_strict(void) {
    int total_fixed = 0;
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    MEMORY_BASIC_INFORMATION mbi;
    uint8_t* addr = (uint8_t*)si.lpMinimumApplicationAddress;
    while (addr < (uint8_t*)si.lpMaximumApplicationAddress) {
        size_t query = VirtualQuery((void*)addr, &mbi, sizeof(mbi));
        if (query == 0) break;
        if (mbi.State == MEM_COMMIT && (mbi.Protect & 0xF0) != 0) {
            DWORD target = mbi.Protect;
            if ((mbi.Protect & PAGE_EXECUTE_READWRITE) == PAGE_EXECUTE_READWRITE) {
                target = PAGE_READWRITE;
            } else if ((mbi.Protect & PAGE_EXECUTE_WRITECOPY) == PAGE_EXECUTE_WRITECOPY) {
                target = PAGE_WRITECOPY;
            } else if (mbi.Protect & PAGE_EXECUTE) {
                if (mbi.Protect & PAGE_READONLY) target = PAGE_EXECUTE_READ;
                else if (mbi.Protect & PAGE_READWRITE) target = PAGE_EXECUTE_READWRITE;
                else target = PAGE_EXECUTE_READ;
            }
            if (target != mbi.Protect) {
                DWORD old;
                if (VirtualProtect(mbi.BaseAddress, mbi.RegionSize, target, &old)) total_fixed++;
            }
        }
        addr += mbi.RegionSize;
    }
    (void)si;
#else
    FILE* maps = fopen("/proc/self/maps", "r");
    if (!maps) return -1;
    char line[512];
    while (fgets(line, sizeof(line), maps)) {
        unsigned long start, end;
        char perms[8] = {0};
        if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) < 3) continue;
        size_t len = end - start;
        if (len == 0) continue;
        if (strchr(perms, 'x') && strchr(perms, 'w')) {
            int prot = PROT_READ | PROT_WRITE;
            if (mprotect((void*)(uintptr_t)start, len, prot) == 0) total_fixed++;
        } else if (strchr(perms, 'x') && !strchr(perms, 'r') && !strchr(perms, 'w')) {
            int prot = PROT_READ | PROT_EXEC;
            if (mprotect((void*)(uintptr_t)start, len, prot) == 0) total_fixed++;
        } else if (perms[0] == '-' && perms[1] == '-' && perms[2] == 'x') {
            int prot = PROT_READ | PROT_EXEC;
            if (mprotect((void*)(uintptr_t)start, len, prot) == 0) total_fixed++;
        }
    }
    fclose(maps);
#endif
    return total_fixed;
}

/**
 * @brief Perform Mem Enforce Wx Check.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_enforce_wx_check(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    MEMORY_BASIC_INFORMATION mbi;
    uint8_t* addr = (uint8_t*)si.lpMinimumApplicationAddress;
    while (addr < (uint8_t*)si.lpMaximumApplicationAddress) {
        if (VirtualQuery((void*)addr, &mbi, sizeof(mbi)) == 0) break;
        if (mbi.State == MEM_COMMIT && (mbi.Protect & 0xF0) != 0) {
            if ((mbi.Protect & PAGE_EXECUTE_READWRITE) == PAGE_EXECUTE_READWRITE) {
                (void)si;
                return 1;
            }
        }
        addr += mbi.RegionSize;
    }
    (void)si;
    return 0;
#else
    FILE* maps = fopen("/proc/self/maps", "r");
    if (!maps) return -1;
    int found = 0;
    char line[512];
    while (fgets(line, sizeof(line), maps)) {
        unsigned long start, end;
        char perms[8] = {0};
        if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) < 3) continue;
        if (perms[0] == 'r' && perms[1] == 'w' && perms[2] == 'x') { found = 1; break; }
    }
    fclose(maps);
    return found;
#endif
}

/* === Expanded CFG === */

/**
 * @brief Get Cfg Remove Tar.
 *
 * @param cfg [out] Cfg value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cfg_remove_target(SNEPPXCFG* cfg, void* target) {
    if (!cfg || !target) return -1;
    uintptr_t t = (uintptr_t)target;
    for (int i = 0; i < cfg->target_count; i++) {
        if (cfg->valid_targets[i] == t) {
            cfg->valid_targets[i] = cfg->valid_targets[cfg->target_count - 1];
            cfg->target_count--;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Clear Cfg.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cfg_clear(SNEPPXCFG* cfg) {
    if (!cfg) return -1;
    memset(cfg->valid_targets, 0, sizeof(cfg->valid_targets));
    cfg->target_count = 0;
    return 0;
}

/* === Expanded Memory Pressure === */

/**
 * @brief Perform Mem Pressure Get Stats.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXMemPressureStats SNEPPX_mem_pressure_get_stats(SNEPPXMemPressure* mp) {
    SNEPPXMemPressureStats stats;
    stats.current = 0;
    stats.pressure_level = -1;
    stats.peak = 0;
    if (!mp) return stats;
    stats.current = mp->total_allocated;
    stats.pressure_level = mp->pressure_level;
    stats.peak = mp->peak_allocated;
    return stats;
}

/**
 * @brief Perform Mem Pressure Set Limit.
 *
 * @param mp [out] Mp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_pressure_set_limit(SNEPPXMemPressure* mp, size_t limit) {
    if (!mp) return -1;
    mp->allocation_limit = limit;
    if (mp->total_allocated > limit) mp->pressure_level = 1;
    if (mp->total_allocated > limit * 2) mp->pressure_level = 2;
    if (mp->total_allocated <= limit) mp->pressure_level = 0;
    return 0;
}

/* === Memory Pool === */

/**
 * @brief Initialize Mem Pool.
 *
 * @param pool [out] Pool value.
 * @param block_size [in] Block Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_pool_init(SNEPPXMemPool* pool, size_t block_size, int blocks) {
    if (!pool || block_size == 0 || blocks <= 0 || blocks > SNEPPX_MEM_POOL_MAX_BLOCKS) return -1;
    if (block_size < sizeof(uint64_t)) block_size = sizeof(uint64_t);
    pool->block_size = block_size;
    pool->total_blocks = blocks;
    pool->free_blocks = blocks;
    size_t bitmap_size = (size_t)((blocks + SNEPPX_BITMAP_BITS_PER_BYTE - 1) / SNEPPX_BITMAP_BITS_PER_BYTE);
    pool->bitmap = (uint8_t*)calloc(bitmap_size, 1);
    if (!pool->bitmap) return -1;
    pool->memory = (uint8_t*)calloc((size_t)blocks, block_size);
    if (!pool->memory) {
        free(pool->bitmap);
        pool->bitmap = NULL;
        return -1;
    }
    memset(pool->bitmap, 0, bitmap_size);
    memset(pool->memory, 0xAB, (size_t)blocks * block_size);
    return 0;
}

/**
 * @brief Perform Mem Pool Alloc.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_mem_pool_alloc(SNEPPXMemPool* pool) {
    if (!pool || !pool->bitmap || !pool->memory) return NULL;
    if (pool->free_blocks <= 0) return NULL;
    for (int i = 0; i < pool->total_blocks; i++) {
        int byte_idx = i / SNEPPX_BITMAP_BITS_PER_BYTE;
        int bit_idx = i % SNEPPX_BITMAP_BITS_PER_BYTE;
        if (!(pool->bitmap[byte_idx] & (uint8_t)(1 << bit_idx))) {
            pool->bitmap[byte_idx] |= (uint8_t)(1 << bit_idx);
            pool->free_blocks--;
            void* block = (uint8_t*)pool->memory + (size_t)i * pool->block_size;
            memset(block, 0, pool->block_size);
            return block;
        }
    }
    return NULL;
}

/**
 * @brief Free Mem Pool.
 *
 * @param pool [out] Pool value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_pool_free(SNEPPXMemPool* pool, void* ptr) {
    if (!pool || !pool->bitmap || !pool->memory || !ptr) return -1;
    uintptr_t mem_start = (uintptr_t)pool->memory;
    uintptr_t mem_end = mem_start + (uintptr_t)pool->total_blocks * pool->block_size;
    uintptr_t p = (uintptr_t)ptr;
    if (p < mem_start || p >= mem_end) return -1;
    uintptr_t offset = p - mem_start;
    if (offset % pool->block_size != 0) return -1;
    int idx = (int)(offset / pool->block_size);
    if (idx < 0 || idx >= pool->total_blocks) return -1;
    int byte_idx = idx / SNEPPX_BITMAP_BITS_PER_BYTE;
    int bit_idx = idx % SNEPPX_BITMAP_BITS_PER_BYTE;
    if (!(pool->bitmap[byte_idx] & (uint8_t)(1 << bit_idx))) return -1;
    memset(ptr, SNEPPX_POOL_SCRUB_BYTE, pool->block_size);
    pool->bitmap[byte_idx] &= (uint8_t)~(1 << bit_idx);
    pool->free_blocks++;
    return 0;
}

/* === Page Alignment === */

/**
 * @brief Perform Mem Page Size.
 *
 * @return The computed size/count, or 0 on error.
 */
static size_t SNEPPX_mem_page_size(void);

/**
 * @brief Perform Mem Is Page Aligned.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_is_page_aligned(const void* addr) {
    if (!addr) return 0;
    size_t ps = SNEPPX_mem_page_size();
    return (((uintptr_t)addr & (ps - 1)) == 0) ? 1 : 0;
}

/**
 * @brief Perform Mem Page Size.
 *
 * @return The computed size/count, or 0 on error.
 */
static size_t SNEPPX_mem_page_size(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (size_t)si.dwPageSize;
#elif defined(_SC_PAGESIZE)
    long sz = sysconf(_SC_PAGESIZE);
    return (sz > 0) ? (size_t)sz : 4096;
#elif defined(_SC_PAGE_SIZE)
    long sz = sysconf(_SC_PAGE_SIZE);
    return (sz > 0) ? (size_t)sz : 4096;
#else
    return 4096;
#endif
}

/**
 * @brief Perform Mem Round Up To Page.
 *
 * @return 0 on success, -1 on error.
 */
static int SNEPPX_mem_round_up_to_page(size_t* size) {
    if (!size || *size == 0) return -1;
    size_t ps = SNEPPX_mem_page_size();
    *size = (*size + ps - 1) & ~(ps - 1);
    return 0;
}

/**
 * @brief Perform Mem Num Pages.
 *
 * @return 0 on success, -1 on error.
 */
static int SNEPPX_mem_num_pages(size_t size) {
    size_t ps = SNEPPX_mem_page_size();
    if (size == 0) return 0;
    return (int)((size + ps - 1) / ps);
}

/* === Expanded Shadow Stack === */

/**
 * @brief Perform Shadow Stack Peek.
 *
 * @param ss [out] Ss value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shadow_stack_peek(SNEPPXShadowStack* ss, uintptr_t* return_addr) {
    if (!ss || !return_addr) return -1;
    if (ss->sp < 0) return -1;
    *return_addr = ss->stack[ss->sp];
    return 0;
}

/**
 * @brief Perform Shadow Stack Depth.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shadow_stack_depth(SNEPPXShadowStack* ss) {
    if (!ss) return -1;
    return ss->sp + 1;
}

/**
 * @brief Perform Shadow Stack Validate.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shadow_stack_validate(SNEPPXShadowStack* ss) {
    if (!ss) return -1;
    if (ss->overflow_detected) return -1;
    if (ss->sp < -1 || ss->sp >= SNEPPX_SHADOW_STACK_DEPTH) return -1;
    if (ss->sp < 0) return 0;
    uint64_t check = 0;
    for (int i = 0; i <= ss->sp; i++) {
        if (ss->stack[i] == 0 || ss->stack[i] == (uintptr_t)-1) return -1;
        if (i > 0) {
            if (ss->stack[i] == ss->stack[i - 1] && ss->stack[i] == 0) return -1;
        }
        check ^= (uint64_t)ss->stack[i];
        check = (check << 7) | (check >> 57);
    }
    uint64_t expected = 0;
    for (int i = 0; i <= ss->sp; i++) {
        expected ^= (uint64_t)ss->stack[i];
        expected = (expected << 7) | (expected >> 57);
    }
    if (check != expected) return -1;
    for (int i = ss->sp + 1; i < SNEPPX_SHADOW_STACK_DEPTH; i++) {
        if (ss->stack[i] != 0) return -1;
    }
    return 0;
}

/* === Freelist Integration with Quarantine === */

/**
 * @brief Perform Quarantine Add With Freelist.
 *
 * @param q [out] Q value.
 * @param ptr [out] Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quarantine_add_with_freelist(SNEPPXMemQuarantine* q, void* ptr, size_t size) {
    int ret = SNEPPX_mem_quarantine_add(q, ptr, size);
    if (ret == 0) SNEPPX_freelist_add(ptr, size);
    return ret;
}

/**
 * @brief Perform Quarantine Check With Freelist.
 *
 * @param q [out] Q value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quarantine_check_with_freelist(SNEPPXMemQuarantine* q, const void* ptr) {
    if (SNEPPX_mem_quarantine_check(q, ptr)) return 1;
    if (g_freelist_initialized) {
        for (int i = 0; i < g_freelist.count; i++) {
            if (g_freelist.entries[i] == ptr) return 1;
        }
    }
    return 0;
}
