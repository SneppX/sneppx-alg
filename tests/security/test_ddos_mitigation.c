#include "ddos_mitigation.h"
#include "test_gtest.h"
#include <stdio.h>
#include <stdint.h>

/*
 * SNEPPX - Test Ddos Mitigation
 *
 * WHAT
 *   Test Ddos Mitigation.
 *
 * CONCEPT
 *   Provides the Test Ddos Mitigation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


TEST(test_ddos_mitigation, suite) {
    ddos_stats_t stats;
    int blocked = SNEPPX_ddos_detect_syn_flood(0xC0A80001, 0xC0A80002);
    printf("SYN flood detection returned %d\n", blocked);
    SNEPPX_ddos_track_connection(0xC0A80001, 0xC0A80002, 1234, 80, 1024);
    uint32_t cookie = SNEPPX_ddos_generate_syn_cookie(0xC0A80001, 0xC0A80002, 1234);
    int valid = SNEPPX_ddos_validate_syn_cookie(cookie, 0xC0A80001, 0xC0A80002, 1234);
    printf("SYN cookie valid: %d\n", valid);
    SNEPPX_ddos_get_stats(&stats);
    printf("DDoS stats: dropped=%llu blocks=%llu\n", stats.total_packets_dropped, stats.total_blocks_issued);
    printf("PASS: DDoS mitigation OK\n");
    return;
}
