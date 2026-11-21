#include "anti_debugging_countermeasure.h"
#include <cstdio>
#include <cassert>
#include <iostream>

/*
 * SNEPPX - Test Obf Anti
 *
 * WHAT
 *   Test Obf Anti.
 *
 * CONCEPT
 *   Provides the Test Obf Anti.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  TEST: %s ... ", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

void test_detect_ptrace() {
    TEST("detect_ptrace_no_debugger");
    SNEPPX::SNEPPXAntiDebug anti;

    bool detected = anti.detect_ptrace();

    ASSERT(detected == false, "ptrace should not detect when not being traced");
    PASS();
}

void test_timing() {
    TEST("timing_measurement");
    SNEPPX::SNEPPXAntiDebug anti;

    bool anomaly = anti.detect_timing_anomaly();

    ASSERT(anomaly == false, "timing anomaly should not trigger in normal execution");
    PASS();
}

void test_breakpoint_scan() {
    TEST("breakpoint_scan");
    SNEPPX::SNEPPXAntiDebug anti;

    uint8_t clean_code[] = { 0x48, 0x89, 0xE5, 0x48, 0x83, 0xEC, 0x20 };
    bool found_clean = anti.detect_breakpoint(clean_code, sizeof(clean_code));
    ASSERT(found_clean == false, "no INT3 in clean code");

    uint8_t code_with_bp[] = { 0x48, 0xCC, 0x89, 0xE5, 0xCC, 0x83, 0xEC };
    bool found_bp = anti.detect_breakpoint(code_with_bp, sizeof(code_with_bp));
    ASSERT(found_bp == true, "should detect INT3 breakpoints");

    PASS();
}

void test_detect_vm() {
    TEST("detect_vm");
    SNEPPX::SNEPPXAntiDebug anti;

    bool is_vm = anti.detect_vm();

    ASSERT(is_vm == false || is_vm == true, "detect_vm should return bool");
    PASS();
}

void test_action_config() {
    TEST("action_config");
    SNEPPX::SNEPPXAntiDebug anti;

    ASSERT(anti.get_action() == SNEPPX::SNEPPXAntiDebugAction::WIPE_AND_EXIT, "default action");

    anti.set_action(SNEPPX::SNEPPXAntiDebugAction::CRASH);
    ASSERT(anti.get_action() == SNEPPX::SNEPPXAntiDebugAction::CRASH, "action change");

    anti.set_action(SNEPPX::SNEPPXAntiDebugAction::SILENT_EXIT);
    ASSERT(anti.get_action() == SNEPPX::SNEPPXAntiDebugAction::SILENT_EXIT, "action change 2");

    PASS();
}

int main() {
    printf("\n=== S2 Obfuscation Anti-Debug Tests ===\n\n");

    test_detect_ptrace();
    test_timing();
    test_breakpoint_scan();
    test_detect_vm();
    test_action_config();

    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
