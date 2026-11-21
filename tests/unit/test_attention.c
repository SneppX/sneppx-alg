#include "multidimensional_tensor_engine.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Attention
 *
 * WHAT
 *   Test Attention.
 *
 * CONCEPT
 *   Provides attention mechanisms.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






static void test_tensor_eye(void) {
    SNEPPXTensor* t = SNEPPX_tensor_eye(4, SNEPPX_FLOAT32);
    SX_ASSERT(t != NULL, "eye created");
    SX_ASSERT(t->shape[0] == 4, "eye rows");
    SX_ASSERT(t->shape[1] == 4, "eye cols");
    float* d = (float*)t->data;
    for (size_t i = 0; i < 4; i++)
        for (size_t j = 0; j < 4; j++)
            SX_ASSERT_NEAR(d[i * 4 + j], (i == j) ? 1.0f : 0.0f, 1e-6f, "eye values");
    SNEPPX_tensor_destroy(t);
}

static void test_tensor_arange(void) {
    SNEPPXTensor* t = SNEPPX_tensor_arange(0.0f, 5.0f, 1.0f, SNEPPX_FLOAT32);
    SX_ASSERT(t != NULL, "arange created");
    float* d = (float*)t->data;
    for (size_t i = 0; i < 5; i++) SX_ASSERT_NEAR(d[i], (float)i, 1e-6f, "arange values");
    SNEPPX_tensor_destroy(t);
}

static void test_tensor_reshape(void) {
    size_t shape[] = {2, 6};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    float* d = (float*)t->data;
    for (size_t i = 0; i < 12; i++) d[i] = (float)i;
    size_t new_shape[] = {3, 4};
    SNEPPXTensor* r = SNEPPX_tensor_reshape(t, new_shape, 2);
    SX_ASSERT(r != NULL, "reshape created");
    SX_ASSERT(r->shape[0] == 3, "reshape rows");
    SX_ASSERT(r->shape[1] == 4, "reshape cols");
    SNEPPX_tensor_destroy(r);
    SNEPPX_tensor_destroy(t);
}

static void test_tensor_transpose(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    float* d = (float*)t->data;
    for (size_t i = 0; i < 6; i++) d[i] = (float)i;
    SNEPPXTensor* tp = SNEPPX_tensor_transpose(t, 0, 1);
    SX_ASSERT(tp != NULL, "transpose created");
    SX_ASSERT(tp->shape[0] == 3, "transpose rows");
    SX_ASSERT(tp->shape[1] == 2, "transpose cols");
    SNEPPX_tensor_destroy(tp);
    SNEPPX_tensor_destroy(t);
}

static void test_tensor_sum(void) {
    size_t shape[] = {2, 2};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* s = SNEPPX_tensor_sum(t, 1);
    SX_ASSERT(s != NULL, "sum created");
    float* sd = (float*)s->data;
    SX_ASSERT_NEAR(sd[0], 2.0f, 1e-6f, "sum row 0");
    SX_ASSERT_NEAR(sd[1], 2.0f, 1e-6f, "sum row 1");
    SNEPPX_tensor_destroy(s);
    SNEPPX_tensor_destroy(t);
}

static void test_tensor_concat(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* a = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    const SNEPPXTensor* tensors[2] = {a, b};
    SNEPPXTensor* c = SNEPPX_tensor_concat(tensors, 2, 0);
    SX_ASSERT(c != NULL, "concat created");
    SX_ASSERT(c->shape[0] == 4, "concat rows");
    SX_ASSERT(c->shape[1] == 3, "concat cols");
    SNEPPX_tensor_destroy(c);
    SNEPPX_tensor_destroy(b);
    SNEPPX_tensor_destroy(a);
}


TEST(test_attention, tensor_eye) { test_tensor_eye(); }
TEST(test_attention, tensor_arange) { test_tensor_arange(); }
TEST(test_attention, tensor_reshape) { test_tensor_reshape(); }
TEST(test_attention, tensor_transpose) { test_tensor_transpose(); }
TEST(test_attention, tensor_sum) { test_tensor_sum(); }
TEST(test_attention, tensor_concat) { test_tensor_concat(); }
