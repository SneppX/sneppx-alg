#include "automatic_differentiation_framework.h"
#include "test_gtest.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <math.h>






static void test_fake_quant_forward_clamp(void) {
    size_t shape[] = {4};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    SX_ASSERT(t != NULL, "tensor created");
    float* d = (float*)t->data;
    d[0] = 1.4f;   /* rounds to 1 */
    d[1] = -0.6f;  /* rounds to -1 */
    d[2] = 200.0f; /* clamps to qmax */
    d[3] = -200.0f;/* clamps to qmin */
    SNEPPXVariable* v = SNEPPX_variable_create(t, 0);
    SX_ASSERT(v != NULL, "variable created");

    /* scale=1.0, bits=8 -> q in [-128,127], y = round(x) */
    SNEPPXTensor* q = SNEPPX_tensor_fake_quant(v->data, 1.0f, 8);
    SX_ASSERT(q != NULL, "fake_quant tensor");
    float* qd = (float*)q->data;
    SX_ASSERT_NEAR(qd[0], 1.0f, 1e-6f, "1.4 -> 1");
    SX_ASSERT_NEAR(qd[1], -1.0f, 1e-6f, "-0.6 -> -1");
    SX_ASSERT_NEAR(qd[2], 127.0f, 1e-6f, "200 clamps to 127");
    SX_ASSERT_NEAR(qd[3], -128.0f, 1e-6f, "-200 clamps to -128");

    SNEPPX_tensor_destroy(q);
    SNEPPX_variable_destroy(v);
}

static void test_fake_quant_ste_backward(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SX_ASSERT(tape != NULL, "tape created");
    size_t shape[] = {3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT32);
    SX_ASSERT(t != NULL, "tensor created");
    float* d = (float*)t->data;
    d[0] = 50.0f;  /* in-range with scale 0.5, bits 8 -> q=100 -> 50.0 */
    d[1] = 1.0f;
    d[2] = -1.0f;
    SNEPPXVariable* v = SNEPPX_variable_create(t, 1);
    SX_ASSERT(v != NULL, "variable created (requires_grad)");

    SNEPPXVariable* q = SNEPPX_fake_quant(tape, v, 0.5f, 8);
    SX_ASSERT(q != NULL, "fake_quant op created");
    SX_ASSERT(q->backward_fn != NULL, "backward_fn set for fake_quant");

    /* seed gradient = ones on output -> STE: input grad = ones (identity) */
    SNEPPX_tape_backward(tape, q);
    SX_ASSERT(v->grad != NULL, "gradient computed for input (STE)");
    float* gd = (float*)v->grad->data;
    SX_ASSERT_NEAR(gd[0], 1.0f, 1e-6f, "STE grad identity for elem 0");
    SX_ASSERT_NEAR(gd[1], 1.0f, 1e-6f, "STE grad identity for elem 1");
    SX_ASSERT_NEAR(gd[2], 1.0f, 1e-6f, "STE grad identity for elem 2");

    /* tape owns q (recorded by the op); v is caller-owned */
    SNEPPX_tape_destroy(tape);
    SNEPPX_variable_destroy(v);
}

static void test_fake_quant_invalid_args(void) {
    size_t shape[] = {2};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* v = SNEPPX_variable_create(t, 1);
    SX_ASSERT(v != NULL, "variable created");
    /* zero scale rejected */
    SX_ASSERT(SNEPPX_tensor_fake_quant(v->data, 0.0f, 8) == NULL, "zero scale rejected");
    /* too-small bits rejected */
    SX_ASSERT(SNEPPX_tensor_fake_quant(v->data, 1.0f, 1) == NULL, "bits<2 rejected");
    /* op-level guard: null input rejected */
    SNEPPXVariable* bad = SNEPPX_fake_quant(NULL, NULL, 1.0f, 8);
    SX_ASSERT(bad == NULL, "null input rejected");
    SNEPPX_variable_destroy(v);
}


TEST(test_fake_quant, fake_quant_forward_clamp) { test_fake_quant_forward_clamp(); }
TEST(test_fake_quant, fake_quant_ste_backward) { test_fake_quant_ste_backward(); }
TEST(test_fake_quant, fake_quant_invalid_args) { test_fake_quant_invalid_args(); }
