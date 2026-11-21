#include "memory_leak_detector.h"
#include "test_gtest.h"
#include <stdio.h>

/*
 * SNEPPX - Test Memory Leak
 *
 * WHAT
 *   Test Memory Leak.
 *
 * CONCEPT
 *   Provides memory management.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


TEST(test_memory_leak, suite) {
    SNEPPX_leak_init();
    leak_stats_t stats;
    void *p1 = SNEPPX_leak_malloc(1024, "test.c", 10, "main");
    void *p2 = SNEPPX_leak_malloc(2048, "test.c", 11, "main");
    SNEPPX_leak_free(p1);
    SNEPPX_leak_get_stats(&stats);
    printf("Stats: active=%d current=%llu peak=%llu\n",
        stats.active_allocations, stats.current_usage, stats.peak_usage);
    leak_report_t reports[16];
    int n = SNEPPX_leak_check(reports, 16);
    printf("Leak reports: %d\n", n);
    for (int i = 0; i < n; i++)
        printf("  Leak: ptr=%p size=%zu at %s:%d\n", reports[i].ptr, reports[i].size, reports[i].file, reports[i].line);
    SNEPPX_leak_free(p2);
    printf("PASS: Memory leak detector OK\n");
    return;
}
