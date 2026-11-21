#include "secure_allocator.h"
#include "neural_core/security/cryptographic_random_generator.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#define SNEPPX_SECURE_PAGE_SIZE 4096
#define SNEPPX_SECURE_CANARY_SIZE 8
#define SNEPPX_SECURE_MAX_RECORDS 1024
#define SNEPPX_SECURE_OVERHEAD_SIZE 16
#define SNEPPX_SECURE_GUARD_MAGIC 0x414E58
#define SNEPPX_SECURE_QUARANTINE_SIZE 64
#define SNEPPX_SECURE_FREED_SET_SIZE 1024
#define SNEPPX_SECURE_FREELIST_CANARY ((uint64_t)0xDEADFEEE)

#define SNEPPX_SMALL_OBJECT_MAX 64
#define SNEPPX_SMALL_BINS 7
#define SNEPPX_SMALL_BIN_CAPACITY 256
#define SNEPPX_SMALL_BITMAP_WORDS 8

#define SNEPPX_SCRUB_PATTERN_ZERO 0
#define SNEPPX_SCRUB_PATTERN_ONE 1
#define SNEPPX_SCRUB_PATTERN_RANDOM 2

/*
 * SNEPPX - Secure Allocator
 *
 * WHAT
 *   Secure Allocator.
 *
 * CONCEPT
 *   Provides the Secure Allocator.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int g_scrub_pattern = SNEPPX_SCRUB_PATTERN_ZERO;
static int g_quarantine_max = SNEPPX_SECURE_QUARANTINE_SIZE;

typedef struct {
    size_t actual_size;
    uint32_t magic;
    uint32_t reserved;
} SNEPPXSecureAllocHeader;

typedef struct SNEPPXSecureFreeNode {
    void* base;
    size_t total;
    uint64_t canary;
    struct SNEPPXSecureFreeNode* next;
    struct SNEPPXSecureFreeNode* prev;
} SNEPPXSecureFreeNode;

typedef struct {
    void* base;
    size_t total_with_guards;
} SNEPPXSecureQuarantineEntry;

typedef struct {
    void* blocks[SNEPPX_SMALL_BIN_CAPACITY];
    uint32_t bitmap[SNEPPX_SMALL_BITMAP_WORDS];
    int count;
    int bin_size;
} SNEPPXSmallObjectBin;

static SNEPPXSecureAllocRecord g_records[SNEPPX_SECURE_MAX_RECORDS];
static int g_record_count = 0;
static int g_initialized = 0;
static SNEPPXSecureFreeNode* g_freelist_head = NULL;
static void* g_freed_set[SNEPPX_SECURE_FREED_SET_SIZE];
static SNEPPXSecureQuarantineEntry g_quarantine[SNEPPX_SECURE_QUARANTINE_SIZE];
static int g_quarantine_count = 0;
static size_t g_stats_num_frees = 0;
static size_t g_stats_num_double_free = 0;
static size_t g_stats_num_canary_violations = 0;

static SNEPPXSmallObjectBin g_small_bins[SNEPPX_SMALL_BINS];
static int g_small_bins_initialized = 0;

static int g_small_sizes[SNEPPX_SMALL_BINS] = {8, 16, 24, 32, 40, 48, 64};

static void secure_zero(void* ptr, size_t len) {
    volatile unsigned char* p = (volatile unsigned char*)ptr;
    for (size_t i = 0; i < len; i++) p[i] = 0;
}

static void secure_scrub(void* ptr, size_t len) {
    volatile unsigned char* p = (volatile unsigned char*)ptr;
    if (g_scrub_pattern == SNEPPX_SCRUB_PATTERN_ZERO) {
        for (size_t i = 0; i < len; i++) p[i] = 0;
    } else if (g_scrub_pattern == SNEPPX_SCRUB_PATTERN_ONE) {
        for (size_t i = 0; i < len; i++) p[i] = 0xFF;
    } else {
        SNEPPX_random_bytes((uint8_t*)p, len);
    }
}

static size_t round_up_page(size_t bytes) {
    return (bytes + SNEPPX_SECURE_PAGE_SIZE - 1) & ~(SNEPPX_SECURE_PAGE_SIZE - 1);
}

static uint32_t freed_set_hash(void* ptr) {
    return (uint32_t)(((uintptr_t)ptr >> 4) % SNEPPX_SECURE_FREED_SET_SIZE);
}

static int freed_set_check(void* ptr) {
    uint32_t idx = freed_set_hash(ptr);
    for (size_t i = 0; i < SNEPPX_SECURE_FREED_SET_SIZE; i++) {
        size_t j = (idx + (uint32_t)i) % SNEPPX_SECURE_FREED_SET_SIZE;
        if (g_freed_set[j] == ptr) return 1;
        if (g_freed_set[j] == NULL) return 0;
    }
    return 0;
}

static void freed_set_add(void* ptr) {
    uint32_t idx = freed_set_hash(ptr);
    for (size_t i = 0; i < SNEPPX_SECURE_FREED_SET_SIZE; i++) {
        size_t j = (idx + (uint32_t)i) % SNEPPX_SECURE_FREED_SET_SIZE;
        if (g_freed_set[j] == NULL || g_freed_set[j] == ptr) {
            g_freed_set[j] = ptr;
            return;
        }
    }
}

static void freed_set_remove(void* ptr) {
    uint32_t idx = freed_set_hash(ptr);
    for (size_t i = 0; i < SNEPPX_SECURE_FREED_SET_SIZE; i++) {
        size_t j = (idx + (uint32_t)i) % SNEPPX_SECURE_FREED_SET_SIZE;
        if (g_freed_set[j] == ptr) {
            g_freed_set[j] = NULL;
            return;
        }
        if (g_freed_set[j] == NULL) return;
    }
}

static void freed_set_clear(void) {
    memset(g_freed_set, 0, sizeof(g_freed_set));
}

static int add_record(void* addr, size_t size, size_t guard_f, size_t guard_b, uint64_t canary) {
    if (g_record_count >= SNEPPX_SECURE_MAX_RECORDS) return -1;
    SNEPPXSecureAllocRecord* r = &g_records[g_record_count++];
    r->addr = addr; r->size = size; r->guard_front = guard_f;
    r->guard_back = guard_b; r->canary = canary; r->is_freed = 0;
    return 0;
}

static SNEPPXSecureFreeNode* freelist_node_alloc(void* base, size_t total) {
    SNEPPXSecureFreeNode* node = (SNEPPXSecureFreeNode*)malloc(sizeof(SNEPPXSecureFreeNode));
    if (!node) return NULL;
    node->base = base;
    node->total = total;
    node->canary = SNEPPX_SECURE_FREELIST_CANARY;
    node->next = NULL;
    node->prev = NULL;
    return node;
}

static void freelist_insert(SNEPPXSecureFreeNode* node) {
    if (!g_freelist_head) {
        g_freelist_head = node;
        return;
    }
    SNEPPXSecureFreeNode* cur = g_freelist_head;
    SNEPPXSecureFreeNode* prv = NULL;
    while (cur && cur->base < node->base) {
        prv = cur;
        cur = cur->next;
    }
    if (prv) {
        char* prv_end = (char*)prv->base + prv->total;
        if (prv_end == node->base) {
            prv->total += node->total;
            prv->canary = SNEPPX_SECURE_FREELIST_CANARY;
            prv->next = node->next;
            if (node->next) node->next->prev = prv;
            free(node);
            node = prv;
        }
    }
    if (cur) {
        char* node_end = (char*)node->base + node->total;
        if (node_end == cur->base) {
            node->total += cur->total;
            node->canary = SNEPPX_SECURE_FREELIST_CANARY;
            node->next = cur->next;
            if (cur->next) cur->next->prev = node;
            if (g_freelist_head == cur) g_freelist_head = node;
            free(cur);
        }
    }
    if (node != prv && node->prev != prv) {
        node->next = cur;
        node->prev = prv;
        if (prv) prv->next = node;
        else g_freelist_head = node;
        if (cur) cur->prev = node;
    }
}

static void freelist_remove(SNEPPXSecureFreeNode* node) {
    if (!node) return;
    if (node->prev) node->prev->next = node->next;
    else g_freelist_head = node->next;
    if (node->next) node->next->prev = node->prev;
    free(node);
}

static SNEPPXSecureFreeNode* freelist_find_by_base(void* base) {
    SNEPPXSecureFreeNode* cur = g_freelist_head;
    while (cur) {
        if (cur->base == base) return cur;
        if (cur->base > base) return NULL;
        cur = cur->next;
    }
    return NULL;
}

static void freelist_destroy(void) {
    SNEPPXSecureFreeNode* cur = g_freelist_head;
    while (cur) {
        SNEPPXSecureFreeNode* next = cur->next;
        free(cur);
        cur = next;
    }
    g_freelist_head = NULL;
}

static int find_small_bin(size_t size) {
    for (int i = 0; i < SNEPPX_SMALL_BINS; i++) {
        if (size <= (size_t)g_small_sizes[i]) return i;
    }
    return -1;
}

static void small_bin_mark_used(int bin_idx, int slot) {
    if (bin_idx < 0 || bin_idx >= SNEPPX_SMALL_BINS) return;
    int word = slot / 32;
    int bit = slot % 32;
    g_small_bins[bin_idx].bitmap[word] |= (1u << bit);
}

static void small_bin_mark_free(int bin_idx, int slot) {
    if (bin_idx < 0 || bin_idx >= SNEPPX_SMALL_BINS) return;
    int word = slot / 32;
    int bit = slot % 32;
    g_small_bins[bin_idx].bitmap[word] &= ~(1u << bit);
}

static int small_bin_find_free(int bin_idx) {
    if (bin_idx < 0 || bin_idx >= SNEPPX_SMALL_BINS) return -1;
    for (int w = 0; w < SNEPPX_SMALL_BITMAP_WORDS; w++) {
        uint32_t bits = g_small_bins[bin_idx].bitmap[w];
        if (bits != 0xFFFFFFFF) {
            for (int b = 0; b < 32; b++) {
                if (!(bits & (1u << b))) {
                    return w * 32 + b;
                }
            }
        }
    }
    return -1;
}

static int init_small_bins(void) {
    if (g_small_bins_initialized) return 0;
    for (int i = 0; i < SNEPPX_SMALL_BINS; i++) {
        memset(&g_small_bins[i], 0, sizeof(SNEPPXSmallObjectBin));
        g_small_bins[i].bin_size = g_small_sizes[i];
        g_small_bins[i].count = 0;
    }
    g_small_bins_initialized = 1;
    return 0;
}

/**
 * @brief Initialize Secure Allocator.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_secure_allocator_init(SNEPPXSecureAllocator* alloc) {
    if (!alloc) return -1;
    memset(alloc, 0, sizeof(*alloc));
    alloc->use_guard_pages = 1;
    alloc->use_canaries = 1;
    g_initialized = 1;
    g_record_count = 0;
    g_quarantine_count = 0;
    g_stats_num_frees = 0;
    g_stats_num_double_free = 0;
    g_stats_num_canary_violations = 0;
    g_scrub_pattern = SNEPPX_SCRUB_PATTERN_ZERO;
    g_quarantine_max = SNEPPX_SECURE_QUARANTINE_SIZE;
    freed_set_clear();
    freelist_destroy();
    init_small_bins();
    return 0;
}

/**
 * @brief Destroy Secure Allocator.
 */
void SNEPPX_secure_allocator_destroy(SNEPPXSecureAllocator* alloc) {
    if (!alloc) return;
    for (int i = 0; i < g_record_count; i++) {
        if (!g_records[i].is_freed && g_records[i].addr) {
            SNEPPX_secure_free(alloc, g_records[i].addr);
        }
    }
    memset(g_records, 0, sizeof(g_records));
    g_record_count = 0;
    memset(alloc, 0, sizeof(*alloc));
    freed_set_clear();
    freelist_destroy();
    g_quarantine_count = 0;
    g_stats_num_frees = 0;
    g_stats_num_double_free = 0;
    g_stats_num_canary_violations = 0;
    memset(g_small_bins, 0, sizeof(g_small_bins));
    g_small_bins_initialized = 0;
}

/**
 * @brief Perform Secure Alloc.
 *
 * @param alloc [out] Alloc value.
 * @param bytes [in] Bytes value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_secure_alloc(SNEPPXSecureAllocator* alloc, size_t bytes, size_t alignment) {
    (void)alignment;
    if (!alloc || bytes == 0) return NULL;

    if (bytes <= SNEPPX_SMALL_OBJECT_MAX) {
        int bin_idx = find_small_bin(bytes);
        if (bin_idx >= 0) {
            int slot = small_bin_find_free(bin_idx);
            if (slot >= 0) {
                small_bin_mark_used(bin_idx, slot);
                void* ptr = g_small_bins[bin_idx].blocks[slot];
                if (ptr) {
                    g_small_bins[bin_idx].count++;
                    add_record(ptr, g_small_bins[bin_idx].bin_size, 0, 0, 0);
                    alloc->total_allocated += g_small_bins[bin_idx].bin_size;
                    alloc->num_allocations++;
                    if (alloc->total_allocated > alloc->peak_allocated)
                        alloc->peak_allocated = alloc->total_allocated;
                    return ptr;
                }
            }
        }
    }

    size_t alloc_bytes = bytes + SNEPPX_SECURE_OVERHEAD_SIZE;
    size_t total = round_up_page(alloc_bytes + SNEPPX_SECURE_CANARY_SIZE);
    size_t total_with_guards = total + SNEPPX_SECURE_PAGE_SIZE * 2;

    void* base = NULL;
#ifdef _WIN32
    base = VirtualAlloc(NULL, total_with_guards, MEM_RESERVE, PAGE_NOACCESS);
    if (!base) return NULL;
    void* user_region = (char*)base + SNEPPX_SECURE_PAGE_SIZE;
    void* user_committed = VirtualAlloc(user_region, total, MEM_COMMIT, PAGE_READWRITE);
    if (!user_committed) { VirtualFree(base, 0, MEM_RELEASE); return NULL; }
#else
    base = mmap(NULL, total_with_guards, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) return NULL;
    void* user_region = (char*)base + SNEPPX_SECURE_PAGE_SIZE;
    if (mprotect(user_region, total, PROT_READ | PROT_WRITE) != 0) {
        munmap(base, total_with_guards); return NULL;
    }
#endif

    SNEPPXSecureAllocHeader* hdr = (SNEPPXSecureAllocHeader*)user_region;
    hdr->magic = SNEPPX_SECURE_GUARD_MAGIC;
    hdr->actual_size = bytes;
    hdr->reserved = 0;

    void* ptr = (char*)user_region + SNEPPX_SECURE_OVERHEAD_SIZE;

    uint64_t canary_val = SNEPPX_secure_canary_generate();
    uint64_t* canary_ptr = (uint64_t*)((char*)ptr + total - SNEPPX_SECURE_OVERHEAD_SIZE - SNEPPX_SECURE_CANARY_SIZE);
    *canary_ptr = canary_val;

    if (alloc->use_guard_pages) {
        alloc->total_allocated += bytes;
        alloc->num_allocations++;
        if (alloc->total_allocated > alloc->peak_allocated)
            alloc->peak_allocated = alloc->total_allocated;
    }

    add_record(ptr, bytes, SNEPPX_SECURE_PAGE_SIZE, SNEPPX_SECURE_PAGE_SIZE, canary_val);
    return ptr;
}

/**
 * @brief Free Secure.
 *
 * @param alloc [out] Alloc value.
 */
void SNEPPX_secure_free(SNEPPXSecureAllocator* alloc, void* ptr) {
    if (!alloc || !ptr) return;

    for (int i = 0; i < g_small_bins_initialized && i < SNEPPX_SMALL_BINS; i++) {
        for (int j = 0; j < SNEPPX_SMALL_BIN_CAPACITY; j++) {
            if (g_small_bins[i].blocks[j] == ptr) {
                small_bin_mark_free(i, j);
                g_small_bins[i].count--;
                alloc->total_allocated -= (alloc->total_allocated >= (size_t)g_small_bins[i].bin_size) ? (size_t)g_small_bins[i].bin_size : alloc->total_allocated;
                g_stats_num_frees++;
                return;
            }
        }
    }

    if (freed_set_check(ptr)) {
        g_stats_num_double_free++;
        if (alloc->on_overflow) {
            for (int i = 0; i < g_record_count; i++) {
                if (g_records[i].addr == ptr) {
                    alloc->on_overflow(&g_records[i]);
                    break;
                }
            }
        }
        return;
    }

    for (int i = 0; i < g_record_count; i++) {
        if (g_records[i].addr == ptr && !g_records[i].is_freed) {
            SNEPPXSecureAllocRecord* r = &g_records[i];
            size_t alloc_bytes = r->size + SNEPPX_SECURE_OVERHEAD_SIZE;
            size_t total = round_up_page(alloc_bytes + SNEPPX_SECURE_CANARY_SIZE);
            size_t total_with_guards = total + r->guard_front + r->guard_back;
            void* base = (char*)ptr - r->guard_front - SNEPPX_SECURE_OVERHEAD_SIZE;
            void* user_region = (char*)base + r->guard_front;

            SNEPPXSecureAllocHeader* hdr = (SNEPPXSecureAllocHeader*)user_region;
            if (hdr->magic != SNEPPX_SECURE_GUARD_MAGIC) {
                g_stats_num_canary_violations++;
                if (alloc->on_overflow) alloc->on_overflow(r);
            }

            uint64_t* canary_ptr = (uint64_t*)((char*)ptr + total - SNEPPX_SECURE_OVERHEAD_SIZE - SNEPPX_SECURE_CANARY_SIZE);
            if (*canary_ptr != r->canary) {
                g_stats_num_canary_violations++;
                if (alloc->on_overflow) alloc->on_overflow(r);
            }

            secure_scrub(ptr, r->size);
            r->is_freed = 1;
            alloc->total_allocated -= (alloc->total_allocated >= r->size) ? r->size : alloc->total_allocated;
            g_stats_num_frees++;
            freed_set_add(ptr);

            SNEPPXSecureFreeNode* node = freelist_node_alloc(base, total_with_guards);
            if (node) freelist_insert(node);

#ifdef _WIN32
            VirtualFree(base, 0, MEM_RELEASE);
#else
            munmap(base, total_with_guards);
#endif
            return;
        }
    }
}

/**
 * @brief Perform Secure Audit.
 */
void SNEPPX_secure_audit(SNEPPXSecureAllocator* alloc) {
    if (!alloc) return;
    printf("=== SNEPPX Secure Allocator Audit ===\n");
    printf("Total allocated: %zu bytes\n", alloc->total_allocated);
    printf("Peak allocated: %zu bytes\n", alloc->peak_allocated);
    printf("Active allocations: %zu\n", alloc->num_allocations);
    int violations = 0;
    for (int i = 0; i < g_record_count; i++) {
        if (g_records[i].is_freed) continue;
        size_t alloc_bytes = g_records[i].size + SNEPPX_SECURE_OVERHEAD_SIZE;
        size_t total = round_up_page(alloc_bytes + SNEPPX_SECURE_CANARY_SIZE);
        uint64_t* canary_ptr = (uint64_t*)((char*)g_records[i].addr + total - SNEPPX_SECURE_OVERHEAD_SIZE - SNEPPX_SECURE_CANARY_SIZE);
        if (*canary_ptr != g_records[i].canary) {
            printf("VIOLATION: canary corrupted at %p\n", g_records[i].addr);
            violations++;
        }
        void* user_region = (char*)g_records[i].addr - SNEPPX_SECURE_OVERHEAD_SIZE;
        SNEPPXSecureAllocHeader* hdr = (SNEPPXSecureAllocHeader*)user_region;
        if (hdr->magic != SNEPPX_SECURE_GUARD_MAGIC) {
            printf("VIOLATION: header magic corrupted at %p\n", g_records[i].addr);
            violations++;
        }
    }
    printf("Canary violations: %d\n", violations);
    printf("Total frees: %zu\n", g_stats_num_frees);
    printf("Double-free detections: %zu\n", g_stats_num_double_free);
    printf("Canary violations (total): %zu\n", g_stats_num_canary_violations);
    printf("Freelist blocks: ");
    int fcount = 0;
    for (SNEPPXSecureFreeNode* n = g_freelist_head; n; n = n->next) fcount++;
    printf("%d\n", fcount);
    printf("Quarantine entries: %d/%d\n", g_quarantine_count, g_quarantine_max);
    printf("Scrub pattern: %d\n", g_scrub_pattern);
    int small_total = 0;
    for (int i = 0; i < SNEPPX_SMALL_BINS; i++)
        small_total += g_small_bins[i].count;
    printf("Small object cache entries: %d\n", small_total);
    printf("================================\n");
}

/**
 * @brief Perform Secure Canary Generate.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_secure_canary_generate(void) {
    static uint64_t counter = 0;
    counter = (counter + 1) * 0x9E3779B97F4A7C15ULL ^ 0xDEADBEEFCAFEBABEULL;
    return counter;
}

/**
 * @brief Perform Secure Canary Check.
 *
 * @param ptr [out] Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_secure_canary_check(void* ptr, uint64_t canary) {
    if (!ptr) return -1;
    for (int i = 0; i < g_record_count; i++) {
        if (g_records[i].addr == ptr && !g_records[i].is_freed) {
            size_t alloc_bytes = g_records[i].size + SNEPPX_SECURE_OVERHEAD_SIZE;
            size_t total = round_up_page(alloc_bytes + SNEPPX_SECURE_CANARY_SIZE);
            uint64_t* canary_ptr = (uint64_t*)((char*)ptr + total - SNEPPX_SECURE_OVERHEAD_SIZE - SNEPPX_SECURE_CANARY_SIZE);
            return (*canary_ptr == canary && g_records[i].canary == canary) ? 0 : -1;
        }
    }
    return -1;
}

/**
 * @brief Perform Secure Freelist Check.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_secure_freelist_check(SNEPPXSecureAllocator* alloc) {
    (void)alloc;
    if (!g_initialized) return -1;
    int errors = 0;
    SNEPPXSecureFreeNode* cur = g_freelist_head;
    while (cur) {
        if (cur->canary != SNEPPX_SECURE_FREELIST_CANARY) {
            fprintf(stderr, "FREELIST ERROR: canary corrupted at node %p (base %p)\n", (void*)cur, cur->base);
            errors++;
        }
        if (cur->prev && cur->prev->next != cur) {
            fprintf(stderr, "FREELIST ERROR: broken prev link at node base %p\n", cur->base);
            errors++;
        }
        if (cur->next && cur->next->prev != cur) {
            fprintf(stderr, "FREELIST ERROR: broken next link at node base %p\n", cur->base);
            errors++;
        }
        if (cur->next && (char*)cur->base + cur->total > (char*)cur->next->base) {
            fprintf(stderr, "FREELIST ERROR: overlapping or out-of-order blocks at base %p and %p\n", cur->base, cur->next->base);
            errors++;
        }
        if (cur->next && (char*)cur->base + cur->total == (char*)cur->next->base) {
            fprintf(stderr, "FREELIST ERROR: adjacent blocks should be merged at base %p\n", cur->base);
            errors++;
        }
        cur = cur->next;
    }
    return (errors == 0) ? 0 : -1;
}

/**
 * @brief Perform Secure Free Quarantine.
 *
 * @param alloc [out] Alloc value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_secure_free_quarantine(SNEPPXSecureAllocator* alloc, void* ptr) {
    if (!alloc || !ptr) return -1;
    if (g_quarantine_count >= g_quarantine_max) {
        SNEPPX_secure_free_flush_quarantine(alloc);
    }
    if (g_quarantine_count >= g_quarantine_max) return -1;

    if (freed_set_check(ptr)) {
        g_stats_num_double_free++;
        return -1;
    }

    for (int i = 0; i < g_record_count; i++) {
        if (g_records[i].addr == ptr && !g_records[i].is_freed) {
            SNEPPXSecureAllocRecord* r = &g_records[i];
            size_t alloc_bytes = r->size + SNEPPX_SECURE_OVERHEAD_SIZE;
            size_t total = round_up_page(alloc_bytes + SNEPPX_SECURE_CANARY_SIZE);
            size_t total_with_guards = total + r->guard_front + r->guard_back;
            void* base = (char*)ptr - r->guard_front - SNEPPX_SECURE_OVERHEAD_SIZE;
            void* user_region = (char*)base + r->guard_front;

            SNEPPXSecureAllocHeader* hdr = (SNEPPXSecureAllocHeader*)user_region;
            if (hdr->magic != SNEPPX_SECURE_GUARD_MAGIC) {
                g_stats_num_canary_violations++;
                if (alloc->on_overflow) alloc->on_overflow(r);
            }

            uint64_t* canary_ptr = (uint64_t*)((char*)ptr + total - SNEPPX_SECURE_OVERHEAD_SIZE - SNEPPX_SECURE_CANARY_SIZE);
            if (*canary_ptr != r->canary) {
                g_stats_num_canary_violations++;
                if (alloc->on_overflow) alloc->on_overflow(r);
            }

            secure_scrub(ptr, r->size);
            r->is_freed = 1;
            alloc->total_allocated -= (alloc->total_allocated >= r->size) ? r->size : alloc->total_allocated;
            g_stats_num_frees++;
            freed_set_add(ptr);

            SNEPPXSecureFreeNode* node = freelist_node_alloc(base, total_with_guards);
            if (node) freelist_insert(node);

            g_quarantine[g_quarantine_count].base = base;
            g_quarantine[g_quarantine_count].total_with_guards = total_with_guards;
            g_quarantine_count++;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Secure Free Flush Quarantine.
 */
void SNEPPX_secure_free_flush_quarantine(SNEPPXSecureAllocator* alloc) {
    (void)alloc;
    for (int i = 0; i < g_quarantine_count; i++) {
        if (g_quarantine[i].base) {
#ifdef _WIN32
            VirtualFree(g_quarantine[i].base, 0, MEM_RELEASE);
#else
            munmap(g_quarantine[i].base, g_quarantine[i].total_with_guards);
#endif
        }
    }
    g_quarantine_count = 0;
    memset(g_quarantine, 0, sizeof(g_quarantine));
}

/**
 * @brief Perform Secure Allocator Get Stats.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXSecureAllocStats SNEPPX_secure_allocator_get_stats(SNEPPXSecureAllocator* alloc) {
    SNEPPXSecureAllocStats stats;
    memset(&stats, 0, sizeof(stats));
    if (!alloc) return stats;
    stats.total_allocated = alloc->total_allocated;
    stats.peak_allocated = alloc->peak_allocated;
    stats.num_allocations = alloc->num_allocations;
    stats.num_frees = g_stats_num_frees;
    stats.num_double_free_detected = g_stats_num_double_free;
    stats.num_canary_violations = g_stats_num_canary_violations;
    return stats;
}

/**
 * @brief Perform Secure Allocator Set Scrub Pattern.
 */
void SNEPPX_secure_allocator_set_scrub_pattern(int pattern_id) {
    if (pattern_id >= 0 && pattern_id <= 2)
        g_scrub_pattern = pattern_id;
}

/**
 * @brief Perform Secure Allocator Get Fragmentation.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_secure_allocator_get_fragmentation(void) {
    if (!g_freelist_head) return 0.0;
    size_t total_free = 0;
    size_t largest_free = 0;
    int block_count = 0;
    SNEPPXSecureFreeNode* cur = g_freelist_head;
    while (cur) {
        total_free += cur->total;
        if (cur->total > largest_free) largest_free = cur->total;
        block_count++;
        cur = cur->next;
    }
    if (total_free == 0 || block_count <= 1) return 0.0;
    return 1.0 - ((double)largest_free / (double)total_free);
}

/**
 * @brief Perform Secure Allocator Defragment.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_secure_allocator_defragment(void) {
    if (!g_freelist_head) return 0;
    int merged = 0;
    SNEPPXSecureFreeNode* cur = g_freelist_head;
    while (cur && cur->next) {
        char* cur_end = (char*)cur->base + cur->total;
        if (cur_end == cur->next->base) {
            cur->total += cur->next->total;
            SNEPPXSecureFreeNode* victim = cur->next;
            cur->next = victim->next;
            if (victim->next) victim->next->prev = cur;
            free(victim);
            merged++;
        } else {
            cur = cur->next;
        }
    }
    return merged;
}

int SNEPPX_secure_allocator_walk(void (*walk_callback)(const SNEPPXSecureAllocRecord* record)) {
    if (!walk_callback) return -1;
    int walked = 0;
    for (int i = 0; i < g_record_count; i++) {
        if (!g_records[i].is_freed && g_records[i].addr) {
            walk_callback(&g_records[i]);
            walked++;
        }
    }
    return walked;
}

/**
 * @brief Perform Secure Allocator Set Quarantine Max.
 */
void SNEPPX_secure_allocator_set_quarantine_max(int max_entries) {
    if (max_entries > 0 && max_entries <= 4096)
        g_quarantine_max = max_entries;
}

/**
 * @brief Perform Secure Allocator Get Quarantine Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_secure_allocator_get_quarantine_count(void) {
    return g_quarantine_count;
}

static int small_bin_populate(int bin_idx) {
    if (bin_idx < 0 || bin_idx >= SNEPPX_SMALL_BINS) return -1;
    SNEPPXSmallObjectBin* bin = &g_small_bins[bin_idx];
    int size = bin->bin_size;
    int needed = 0;
    for (int w = 0; w < SNEPPX_SMALL_BITMAP_WORDS; w++) {
        uint32_t bits = bin->bitmap[w];
        for (int b = 0; b < 32; b++) {
            if (!(bits & (1u << b)) && bin->blocks[w * 32 + b] == NULL) {
                needed++;
            }
        }
    }
    if (needed == 0) return 0;
    size_t block_total = round_up_page((size_t)size * needed + SNEPPX_SECURE_OVERHEAD_SIZE + SNEPPX_SECURE_CANARY_SIZE);
    size_t total_with_guards = block_total + SNEPPX_SECURE_PAGE_SIZE * 2;
    void* base = NULL;
#ifdef _WIN32
    base = VirtualAlloc(NULL, total_with_guards, MEM_RESERVE, PAGE_NOACCESS);
    if (!base) return -1;
    void* user_region = (char*)base + SNEPPX_SECURE_PAGE_SIZE;
    void* user_committed = VirtualAlloc(user_region, block_total, MEM_COMMIT, PAGE_READWRITE);
    if (!user_committed) { VirtualFree(base, 0, MEM_RELEASE); return -1; }
#else
    base = mmap(NULL, total_with_guards, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) return -1;
    void* user_region = (char*)base + SNEPPX_SECURE_PAGE_SIZE;
    if (mprotect(user_region, block_total, PROT_READ | PROT_WRITE) != 0) {
        munmap(base, total_with_guards); return -1;
    }
#endif
    int slot = 0;
    for (int w = 0; w < SNEPPX_SMALL_BITMAP_WORDS && slot < needed; w++) {
        uint32_t bits = bin->bitmap[w];
        for (int b = 0; b < 32 && slot < needed; b++) {
            int idx = w * 32 + b;
            if (!(bits & (1u << b)) && bin->blocks[idx] == NULL) {
                bin->blocks[idx] = (char*)user_region + (size_t)slot * size;
                SNEPPXSecureAllocHeader* hdr = (SNEPPXSecureAllocHeader*)bin->blocks[idx];
                hdr->magic = SNEPPX_SECURE_GUARD_MAGIC;
                hdr->actual_size = (size_t)size;
                hdr->reserved = 0;
                bin->blocks[idx] = (char*)bin->blocks[idx] + SNEPPX_SECURE_OVERHEAD_SIZE;
                slot++;
            }
        }
    }
    return slot;
}
static int freelist_split_block(SNEPPXSecureFreeNode* node, size_t needed) {
    if (!node || node->total < needed) return -1;
    size_t remaining = node->total - needed;
    if (remaining < SNEPPX_SECURE_PAGE_SIZE) return -1;
    void* split_base = (char*)node->base + needed;
    SNEPPXSecureFreeNode* split = freelist_node_alloc(split_base, remaining);
    if (!split) return -1;
    node->total = needed;
    split->next = node->next;
    split->prev = node;
    if (node->next) node->next->prev = split;
    node->next = split;
    return 0;
}

static int freelist_count_blocks(void) {
    int count = 0;
    SNEPPXSecureFreeNode* cur = g_freelist_head;
    while (cur) { count++; cur = cur->next; }
    return count;
}

static size_t freelist_total_free(void) {
    size_t total = 0;
    SNEPPXSecureFreeNode* cur = g_freelist_head;
    while (cur) { total += cur->total; cur = cur->next; }
    return total;
}

static void record_remove_by_addr(void* addr) {
    for (int i = 0; i < g_record_count; i++) {
        if (g_records[i].addr == addr) {
            g_records[i].is_freed = 1;
            g_records[i].addr = NULL;
            return;
        }
    }
}

static int record_find_by_addr(void* addr) {
    for (int i = 0; i < g_record_count; i++) {
        if (g_records[i].addr == addr && !g_records[i].is_freed)
            return i;
    }
    return -1;
}

static void small_bin_destroy(void) {
    for (int i = 0; i < SNEPPX_SMALL_BINS; i++) {
        memset(g_small_bins[i].blocks, 0, sizeof(g_small_bins[i].blocks));
        memset(g_small_bins[i].bitmap, 0, sizeof(g_small_bins[i].bitmap));
        g_small_bins[i].count = 0;
    }
    g_small_bins_initialized = 0;
}

/**
 * @brief Perform Secure Allocator Get Total Allocated.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_secure_allocator_get_total_allocated(SNEPPXSecureAllocator* alloc) {
    if (!alloc) return 0;
    return alloc->total_allocated;
}

/**
 * @brief Perform Secure Allocator Get Peak Allocated.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_secure_allocator_get_peak_allocated(SNEPPXSecureAllocator* alloc) {
    if (!alloc) return 0;
    return alloc->peak_allocated;
}

/**
 * @brief Perform Secure Allocator Get Num Allocations.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_secure_allocator_get_num_allocations(SNEPPXSecureAllocator* alloc) {
    if (!alloc) return 0;
    return alloc->num_allocations;
}

/**
 * @brief Perform Secure Allocator Get Freelist Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_secure_allocator_get_freelist_count(void) {
    return freelist_count_blocks();
}

/**
 * @brief Perform Secure Allocator Get Freelist Total.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_secure_allocator_get_freelist_total(void) {
    return freelist_total_free();
}

/**
 * @brief Perform Secure Allocator Enable Guard Pages.
 *
 * @param alloc [out] Alloc value.
 */
void SNEPPX_secure_allocator_enable_guard_pages(SNEPPXSecureAllocator* alloc, int enable) {
    if (alloc) alloc->use_guard_pages = enable;
}

/**
 * @brief Perform Secure Allocator Enable Canaries.
 *
 * @param alloc [out] Alloc value.
 */
void SNEPPX_secure_allocator_enable_canaries(SNEPPXSecureAllocator* alloc, int enable) {
    if (alloc) alloc->use_canaries = enable;
}
static void freelist_validate_all(void) {
    SNEPPXSecureFreeNode* cur = g_freelist_head;
    while (cur) {
        if (cur->total < SNEPPX_SECURE_PAGE_SIZE) {
            cur = cur->next;
            continue;
        }
        size_t half = cur->total / 2;
        if (half < SNEPPX_SECURE_PAGE_SIZE) { cur = cur->next; continue; }
        void* split_base = (char*)cur->base + half;
        SNEPPXSecureFreeNode* split = freelist_node_alloc(split_base, cur->total - half);
        if (split) {
            cur->total = half;
            split->next = cur->next;
            split->prev = cur;
            if (cur->next) cur->next->prev = split;
            cur->next = split;
        }
        cur = cur->next;
    }
}

/**
 * @brief Perform Secure Allocator Trim Pool.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_secure_allocator_trim_pool(SNEPPXSecureAllocator* alloc) {
    (void)alloc;
    int removed = 0;
    SNEPPXSecureFreeNode* cur = g_freelist_head;
    while (cur) {
        SNEPPXSecureFreeNode* next = cur->next;
        if (cur->total >= SNEPPX_SECURE_PAGE_SIZE * 4) {
            size_t trim = cur->total - SNEPPX_SECURE_PAGE_SIZE * 2;
            if (freelist_split_block(cur, cur->total - trim) == 0) {
                SNEPPXSecureFreeNode* to_free = cur->next;
                if (to_free) {
                    if (to_free->prev) to_free->prev->next = to_free->next;
                    if (to_free->next) to_free->next->prev = to_free->prev;
                    free(to_free);
                    removed++;
                }
            }
        }
        cur = next;
    }
    return removed;
}

/**
 * @brief Perform Secure Allocator Get Small Bin Count.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_secure_allocator_get_small_bin_count(int bin_idx) {
    if (bin_idx < 0 || bin_idx >= SNEPPX_SMALL_BINS) return 0;
    return (size_t)g_small_bins[bin_idx].count;
}

/**
 * @brief Perform Secure Allocator Get Small Bin Size.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_secure_allocator_get_small_bin_size(int bin_idx) {
    if (bin_idx < 0 || bin_idx >= SNEPPX_SMALL_BINS) return 0;
    return g_small_bins[bin_idx].bin_size;
}
/**
 * @brief Perform Secure Allocator Get Record Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_secure_allocator_get_record_count(void) { return g_record_count; }
/**
 * @brief Perform Secure Allocator Get Initialized.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_secure_allocator_get_initialized(void) { return g_initialized; }
/**
 * @brief Perform Secure Allocator Set Initialized.
 */
void SNEPPX_secure_allocator_set_initialized(int v) { g_initialized = v; }
/**
 * @brief Perform Secure Allocator Reset Stats.
 */
void SNEPPX_secure_allocator_reset_stats(void) { g_stats_num_frees = 0; g_stats_num_double_free = 0; g_stats_num_canary_violations = 0; }
