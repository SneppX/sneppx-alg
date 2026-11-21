#include "automatic_differentiation_framework.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Activation
 *
 * WHAT
 *   Test Activation.
 *
 * CONCEPT
 *   Provides activation functions.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






static void test_relu_forward(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    size_t shape[] = {3};
    SNEPPXTensor* data = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    float* d = (float*)data->data;
    d[0] = -2.0f; d[1] = 0.0f; d[2] = 3.0f;
    SNEPPXVariable* v = SNEPPX_variable_create(data, 1);
    SNEPPXVariable* out = SNEPPX_relu(tape, v);
    SX_ASSERT(out != NULL, "relu output");
    float* od = (float*)out->data->data;
    SX_ASSERT_NEAR(od[0], 0.0f, 1e-6f, "relu negative clamped");
    SX_ASSERT_NEAR(od[1], 0.0f, 1e-6f, "relu zero");
    SX_ASSERT_NEAR(od[2], 3.0f, 1e-6f, "relu positive unchanged");
    SNEPPX_tape_destroy(tape);
}

static void test_gelu_forward(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    size_t shape[] = {2};
    SNEPPXTensor* data = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    float* d = (float*)data->data;
    d[0] = 0.0f; d[1] = 1.0f;
    SNEPPXVariable* v = SNEPPX_variable_create(data, 1);
    SNEPPXVariable* out = SNEPPX_gelu(tape, v);
    SX_ASSERT(out != NULL, "gelu output");
    float* od = (float*)out->data->data;
    SX_ASSERT_NEAR(od[0], 0.0f, 1e-2f, "gelu(0) ~ 0");
    SX_ASSERT_NEAR(od[1], 0.841f, 1e-2f, "gelu(1) ~ 0.841");
    SNEPPX_tape_destroy(tape);
}

static void test_silu_forward(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    size_t shape[] = {2};
    SNEPPXTensor* data = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    float* d = (float*)data->data;
    d[0] = 0.0f; d[1] = 2.0f;
    SNEPPXVariable* v = SNEPPX_variable_create(data, 1);
    SNEPPXVariable* out = SNEPPX_silu(tape, v);
    SX_ASSERT(out != NULL, "silu output");
    float* od = (float*)out->data->data;
    SX_ASSERT_NEAR(od[0], 0.0f, 1e-2f, "silu(0) ~ 0");
    SX_ASSERT_NEAR(od[1], 1.761f, 1e-2f, "silu(2) ~ 1.761");
    SNEPPX_tape_destroy(tape);
}


TEST(test_activation, relu_forward) { test_relu_forward(); }
TEST(test_activation, gelu_forward) { test_gelu_forward(); }
TEST(test_activation, silu_forward) { test_silu_forward(); }
