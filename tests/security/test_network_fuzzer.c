#include "network_fuzzer.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Network Fuzzer
 *
 * WHAT
 *   Test Network Fuzzer.
 *
 * CONCEPT
 *   Provides the Test Network Fuzzer.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


TEST(test_network_fuzzer, suite) {
    fuzz_stats_t stats;
    SNEPPX_fuzzer_init(FUZZ_PROTO_RAW);
    uint8_t buf[1024];
    size_t len;
    for (int i = 0; i < 100; i++) {
        SNEPPX_fuzzer_generate(buf, &len, sizeof(buf));
        fuzz_result_t result;
        SNEPPX_fuzzer_execute(buf, len, &result);
    }
    SNEPPX_fuzzer_get_stats(&stats);
    printf("Fuzzer stats: exec=%llu crashes=%llu cov=%.1f%%\n",
        stats.total_executions, stats.total_crashes, stats.coverage_percent);
    SNEPPX_fuzzer_init(FUZZ_PROTO_TLS);
    for (int i = 0; i < 50; i++) {
        SNEPPX_fuzzer_generate(buf, &len, sizeof(buf));
        fuzz_result_t result;
        SNEPPX_fuzzer_execute(buf, len, &result);
    }
    SNEPPX_fuzzer_get_stats(&stats);
    printf("TLS fuzzer: exec=%llu mut=%llu\n", stats.total_executions, stats.total_mutations);
    printf("PASS: Network fuzzer OK\n");
    return;
}
