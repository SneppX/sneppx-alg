#include "container_breakout.h"
#include "test_gtest.h"
#include <stdio.h>

/*
 * SNEPPX - Test Container Breakout
 *
 * WHAT
 *   Test Container Breakout.
 *
 * CONCEPT
 *   Provides the Test Container Breakout.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


TEST(test_container_breakout, suite) {
    breakout_stats_t stats;
    int r = SNEPPX_breakout_detect_ns_change(1234, "malicious");
    printf("NS change detect: %d\n", r);
    r = SNEPPX_breakout_detect_mount(1234, "/etc/passwd", "malicious");
    printf("Mount detect (suspicious path): %d\n", r);
    r = SNEPPX_breakout_detect_capability(1234, 0xFFFFFFFF, "malicious");
    printf("Capability detect: %d\n", r);
    r = SNEPPX_breakout_detect_syscall(1234, "mount", "malicious");
    printf("Syscall detect: %d\n", r);
    r = SNEPPX_breakout_check_file_access("/etc/shadow");
    printf("File access detect: %d\n", r);
    SNEPPX_breakout_add_watch_file("/var/run/docker.sock");
    SNEPPX_breakout_get_stats(&stats);
    printf("Breakout stats: attempts=%llu blocks=%llu suspicious=%d\n",
        stats.total_breakout_attempts, stats.total_blocks, stats.suspicious_processes);
    printf("PASS: Container breakout detection OK\n");
    return;
}
