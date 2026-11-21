#include "adversarial_robustness_certification.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Arc Input Guard
 *
 * WHAT
 *   Test Arc Input Guard.
 *
 * CONCEPT
 *   Provides buffer overflow protection.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_guard_create(void) {
    SNEPPXInputGuard* g = SNEPPX_input_guard_create(64, 64, 42);
    SX_ASSERT(g != NULL, "guard not null");
    SX_ASSERT(g->projection_matrix->shape[0] == 64, "proj rows 64");
    SX_ASSERT(g->projection_matrix->shape[1] == 64, "proj cols 64");
    SNEPPX_input_guard_destroy(g);
}

static void test_guard_normal_input(void) {
    SNEPPXInputGuard* g = SNEPPX_input_guard_create(64, 64, 42);
    SX_ASSERT(g != NULL, "guard not null");

    size_t shape_in[] = {8, 64};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    unsigned long s = 123;
    for (size_t i = 0; i < 8 * 64; i++) {
        s = s * 1103515245UL + 12345UL;
        d[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 0.2f;
    }

    SNEPPXTensor* sanitized = NULL;
    float score = 0.0f;
    SNEPPX_arc_input_guard_forward(g, input, &sanitized, &score);
    SX_ASSERT(sanitized != NULL, "sanitized not null");
    SX_ASSERT(score < 0.1f, "anomaly_score < 0.1");
    SX_ASSERT(sanitized->shape[0] == 8, "batch ok");
    SX_ASSERT(sanitized->shape[1] == 64, "dim ok");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(sanitized);
    SNEPPX_input_guard_destroy(g);
}

static void test_guard_anomaly_input(void) {
    SNEPPXInputGuard* g = SNEPPX_input_guard_create(64, 64, 42);
    SX_ASSERT(g != NULL, "guard not null");

    size_t shape_in[] = {8, 64};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    unsigned long s = 123;
    for (size_t i = 0; i < 8 * 64; i++) {
        s = s * 1103515245UL + 12345UL;
        d[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 20.0f;
    }

    SNEPPXTensor* sanitized = NULL;
    float score = 0.0f;
    SNEPPX_arc_input_guard_forward(g, input, &sanitized, &score);
    SX_ASSERT(sanitized != NULL, "sanitized not null");
    SX_ASSERT(score > 0.5f, "anomaly_score > 0.5");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(sanitized);
    SNEPPX_input_guard_destroy(g);
}


TEST(test_arc_input_guard, test_guard_create) { test_guard_create(); }
TEST(test_arc_input_guard, test_guard_normal_input) { test_guard_normal_input(); }
TEST(test_arc_input_guard, test_guard_anomaly_input) { test_guard_anomaly_input(); }
