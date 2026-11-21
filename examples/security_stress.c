#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "protected_memory_manager.h"
#include "stack_canary_protection.h"
#include "timing_attack_countermeasure.h"

/*
 * SNEPPX - Security Stress
 *
 * WHAT
 *   Security Stress.
 *
 * CONCEPT
 *   Provides the Security Stress.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


int main(void) {
    printf("=== SNEPPX S1 Secure Memory Stress Test ===\n\n");

    SNEPPXSecureAllocConfig cfg = {1, 1, 0, 1};
    SNEPPXSecurePool* pool = SNEPPX_secure_pool_create(100 * 1024 * 1024, &cfg);
    if (!pool) {
        printf("FAIL: could not create 100MB pool\n");
        return 1;
    }
    printf("Pool created: 100 MB with guard pages, canaries, ASLR\n\n");

    srand(42);
    void** ptrs = (void**)malloc(10000 * sizeof(void*));
    size_t* sizes = (size_t*)malloc(10000 * sizeof(size_t));
    if (!ptrs || !sizes) {
        printf("FAIL: malloc for tracking\n");
        return 1;
    }
    memset(ptrs, 0, 10000 * sizeof(void*));

    uint64_t total_time = 0, total_alloc_time = 0, total_free_time = 0;
    int alloc_ok = 0, corrupt_detected = 0;

    printf("Allocating 10,000 random-sized blocks...\n");
    for (int i = 0; i < 10000; i++) {
        size_t sz = (size_t)((rand() % 4096) + 1);
        sizes[i] = sz;
        uint64_t t0 = SNEPPX_timing_start();
        ptrs[i] = SNEPPX_secure_malloc(pool, sz, 16);
        uint64_t t1 = SNEPPX_timing_end();
        total_alloc_time += (t1 - t0);
        if (ptrs[i]) {
            alloc_ok++;
            memset(ptrs[i], i & 0xFF, sz);
        }
    }
    printf("  Allocations succeeded: %d / 10000\n", alloc_ok);

    printf("Corrupting 1%% of canaries and freeing all...\n");
    for (int i = 0; i < 10000; i++) {
        if (!ptrs[i]) continue;
        if (rand() % 100 == 0 && sizes[i] >= 8) {
            uint8_t* p = (uint8_t*)ptrs[i];
            p[sizes[i] - 1] ^= 0xFF;
        }
    }

    for (int i = 0; i < 10000; i++) {
        if (!ptrs[i]) continue;
        uint64_t t0 = SNEPPX_timing_start();
        SNEPPX_secure_pool_free(pool, ptrs[i], sizes[i]);
        uint64_t t1 = SNEPPX_timing_end();
        total_free_time += (t1 - t0);
    }

    size_t total, used, peak;
    SNEPPX_secure_pool_stats(pool, &total, &used, &peak);
    printf("\n  Peak usage: %zu bytes (%.1f%% of %zu)\n",
           peak, (double)peak/total*100, total);

    double avg_alloc = (alloc_ok > 0) ?
        (double)(total_alloc_time / 2400) / alloc_ok : 0;
    double avg_free = (double)(total_free_time / 2400) / 10000;
    printf("  Avg alloc latency: %.0f ns\n", avg_alloc);
    printf("  Avg free latency:  %.0f ns\n", avg_free);

    SNEPPX_secure_pool_destroy(pool);
    free(ptrs);
    free(sizes);

    printf("\n=== Stress test complete ===\n");
    return 0;
}
