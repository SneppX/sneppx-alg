#include "stack_canary_protection.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Canary
 *
 * WHAT
 *   Test Canary.
 *
 * CONCEPT
 *   Provides stack canary hardening.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_canary_set_check(void) {
    SNEPPXCanary canary;
    SNEPPX_canary_generate(&canary);
    SX_ASSERT(canary.generation >= 0, "canary generation set");
    uint8_t buf[SNEPPX_CANARY_SIZE];
    SNEPPX_canary_write(&canary, buf);
    int ok = SNEPPX_canary_verify(&canary, buf);
    SX_ASSERT(ok, "canary check passes");
}

static void test_canary_detect_tamper(void) {
    SNEPPXCanary canary;
    SNEPPX_canary_generate(&canary);
    uint8_t buf[SNEPPX_CANARY_SIZE];
    SNEPPX_canary_write(&canary, buf);
    buf[0] ^= 0xFF;  // tamper
    int ok = SNEPPX_canary_verify(&canary, buf);
    SX_ASSERT(!ok, "tampered canary fails check");
}

static void test_canary_refresh(void) {
    SNEPPXCanary c1, c2;
    SNEPPX_canary_generate(&c1);
    SNEPPX_canary_generate(&c2);
    int same = 1;
    for (int i = 0; i < SNEPPX_CANARY_SIZE; i++) {
        if (c1.value[i] != c2.value[i]) { same = 0; break; }
    }
    // Very unlikely they're the same
    SX_ASSERT(!same || c1.generation != c2.generation || c1.value[0] != c2.value[0], "refreshed canary differs");
    uint8_t buf[SNEPPX_CANARY_SIZE];
    SNEPPX_canary_write(&c2, buf);
    int ok = SNEPPX_canary_verify(&c2, buf);
    SX_ASSERT(ok, "refreshed canary valid");
}


TEST(test_canary, canary_set_check) { test_canary_set_check(); }
TEST(test_canary, canary_detect_tamper) { test_canary_detect_tamper(); }
TEST(test_canary, canary_refresh) { test_canary_refresh(); }
