#include "multidimensional_tensor_engine.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>

/*
 * SNEPPX - Test Tensor Ops
 *
 * WHAT
 *   Test Tensor Ops.
 *
 * CONCEPT
 *   Provides tensor operations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






/* ---------- Element-wise Arithmetic ---------- */

static void test_add_basic(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* a = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* c = SNEPPX_tensor_add(a, b);
    SX_ASSERT(c != NULL, "add result not null");
    float* d = (float*)c->data;
    for (size_t i = 0; i < 6; i++) SX_ASSERT_NEAR(d[i], 2.0f, 1e-6f, "1+1==2");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b); SNEPPX_tensor_destroy(c);
}

static void test_add_broadcast(void) {
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3};
    SNEPPXTensor* a = SNEPPX_tensor_full(shape_a, 2, SNEPPX_FLOAT32, SX_ARR_C(float, 1, 2.0f));
    SNEPPXTensor* b = SNEPPX_tensor_ones(shape_b, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* c = SNEPPX_tensor_add(a, b);
    SX_ASSERT(c != NULL, "broadcast add not null");
    float* d = (float*)c->data;
    for (size_t i = 0; i < 6; i++) SX_ASSERT_NEAR(d[i], 3.0f, 1e-6f, "2+1==3");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b); SNEPPX_tensor_destroy(c);
}

static void test_sub_basic(void) {
    size_t shape[] = {3};
    SNEPPXTensor* a = SNEPPX_tensor_arange(5.0f, 0.0f, -1.0f, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* c = SNEPPX_tensor_sub(a, b);
    SX_ASSERT(c != NULL, "sub result not null");
    float* d = (float*)c->data;
    SX_ASSERT_NEAR(d[0], 4.0f, 1e-6f, "5-1==4");
    SX_ASSERT_NEAR(d[1], 3.0f, 1e-6f, "4-1==3");
    SX_ASSERT_NEAR(d[2], 2.0f, 1e-6f, "3-1==2");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b); SNEPPX_tensor_destroy(c);
}

static void test_mul_basic(void) {
    size_t shape[] = {3};
    float two = 2.0f;
    SNEPPXTensor* a = SNEPPX_tensor_arange(1.0f, 4.0f, 1.0f, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_full(shape, 1, SNEPPX_FLOAT32, &two);
    SNEPPXTensor* c = SNEPPX_tensor_mul(a, b);
    SX_ASSERT(c != NULL, "mul result not null");
    float* d = (float*)c->data;
    SX_ASSERT_NEAR(d[0], 2.0f, 1e-6f, "1*2==2");
    SX_ASSERT_NEAR(d[1], 4.0f, 1e-6f, "2*2==4");
    SX_ASSERT_NEAR(d[2], 6.0f, 1e-6f, "3*2==6");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b); SNEPPX_tensor_destroy(c);
}

static void test_div_basic(void) {
    size_t shape[] = {3};
    SNEPPXTensor* a = SNEPPX_tensor_arange(2.0f, 8.0f, 2.0f, SNEPPX_FLOAT32);
    float two = 2.0f;
    SNEPPXTensor* b = SNEPPX_tensor_full(shape, 1, SNEPPX_FLOAT32, &two);
    SNEPPXTensor* c = SNEPPX_tensor_div(a, b);
    SX_ASSERT(c != NULL, "div result not null");
    float* d = (float*)c->data;
    SX_ASSERT_NEAR(d[0], 1.0f, 1e-6f, "2/2==1");
    SX_ASSERT_NEAR(d[1], 2.0f, 1e-6f, "4/2==2");
    SX_ASSERT_NEAR(d[2], 3.0f, 1e-6f, "6/2==3");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b); SNEPPX_tensor_destroy(c);
}

static void test_pow_basic(void) {
    size_t shape[] = {3};
    SNEPPXTensor* a = SNEPPX_tensor_arange(1.0f, 4.0f, 1.0f, SNEPPX_FLOAT32);
    float two = 2.0f;
    SNEPPXTensor* b = SNEPPX_tensor_full(shape, 1, SNEPPX_FLOAT32, &two);
    SNEPPXTensor* c = SNEPPX_tensor_pow(a, b);
    SX_ASSERT(c != NULL, "pow result not null");
    float* d = (float*)c->data;
    SX_ASSERT_NEAR(d[0], 1.0f, 1e-5f, "1^2==1");
    SX_ASSERT_NEAR(d[1], 4.0f, 1e-5f, "2^2==4");
    SX_ASSERT_NEAR(d[2], 9.0f, 1e-5f, "3^2==9");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b); SNEPPX_tensor_destroy(c);
}

static void test_neg_basic(void) {
    size_t shape[] = {3};
    SNEPPXTensor* a = SNEPPX_tensor_arange(1.0f, 4.0f, 1.0f, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_neg(a);
    SX_ASSERT(b != NULL, "neg result not null");
    float* d = (float*)b->data;
    SX_ASSERT_NEAR(d[0], -1.0f, 1e-6f, "-1");
    SX_ASSERT_NEAR(d[1], -2.0f, 1e-6f, "-2");
    SX_ASSERT_NEAR(d[2], -3.0f, 1e-6f, "-3");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void test_abs_basic(void) {
    size_t shape[] = {3};
    float vals[] = {-3.0f, 0.0f, 4.0f};
    SNEPPXTensor* a = SNEPPX_tensor_empty(NULL, 0, SNEPPX_FLOAT32);
    a = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    memcpy(a->data, vals, 3 * sizeof(float));
    SNEPPXTensor* b = SNEPPX_tensor_abs(a);
    SX_ASSERT(b != NULL, "abs result not null");
    float* d = (float*)b->data;
    SX_ASSERT_NEAR(d[0], 3.0f, 1e-6f, "abs(-3)==3");
    SX_ASSERT_NEAR(d[1], 0.0f, 1e-6f, "abs(0)==0");
    SX_ASSERT_NEAR(d[2], 4.0f, 1e-6f, "abs(4)==4");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void test_sign_basic(void) {
    size_t shape[] = {3};
    float vals[] = {-5.0f, 0.0f, 3.0f};
    SNEPPXTensor* a = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    memcpy(a->data, vals, 3 * sizeof(float));
    SNEPPXTensor* b = SNEPPX_tensor_sign(a);
    SX_ASSERT(b != NULL, "sign result not null");
    float* d = (float*)b->data;
    SX_ASSERT_NEAR(d[0], -1.0f, 1e-6f, "sign(-5)==-1");
    SX_ASSERT_NEAR(d[1], 0.0f, 1e-6f, "sign(0)==0");
    SX_ASSERT_NEAR(d[2], 1.0f, 1e-6f, "sign(3)==1");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void test_exp_log_sqrt(void) {
    size_t shape[] = {3};
    SNEPPXTensor* a = SNEPPX_tensor_arange(1.0f, 4.0f, 1.0f, SNEPPX_FLOAT32);
    SNEPPXTensor* e = SNEPPX_tensor_exp(a);
    SX_ASSERT(e != NULL, "exp result not null");
    float* ed = (float*)e->data;
    SX_ASSERT_NEAR(ed[0], expf(1.0f), 1e-5f, "exp(1)");
    SX_ASSERT_NEAR(ed[1], expf(2.0f), 1e-5f, "exp(2)");
    SX_ASSERT_NEAR(ed[2], expf(3.0f), 1e-5f, "exp(3)");
    SNEPPXTensor* l = SNEPPX_tensor_log(e);
    SX_ASSERT(l != NULL, "log result not null");
    float* ld = (float*)l->data;
    SX_ASSERT_NEAR(ld[0], 1.0f, 1e-4f, "log(exp(1))==1");
    SX_ASSERT_NEAR(ld[1], 2.0f, 1e-4f, "log(exp(2))==2");
    SX_ASSERT_NEAR(ld[2], 3.0f, 1e-4f, "log(exp(3))==3");
    SNEPPXTensor* s = SNEPPX_tensor_sqrt(a);
    SX_ASSERT(s != NULL, "sqrt result not null");
    float* sd = (float*)s->data;
    SX_ASSERT_NEAR(sd[0], 1.0f, 1e-5f, "sqrt(1)==1");
    SX_ASSERT_NEAR(sd[1], sqrtf(2.0f), 1e-5f, "sqrt(2)");
    SX_ASSERT_NEAR(sd[2], sqrtf(3.0f), 1e-5f, "sqrt(3)");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(e); SNEPPX_tensor_destroy(l); SNEPPX_tensor_destroy(s);
}

static void test_round_ops(void) {
    size_t shape[] = {4};
    float vals[] = {1.4f, 1.6f, -1.4f, -1.6f};
    SNEPPXTensor* a = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    memcpy(a->data, vals, 4 * sizeof(float));
    SNEPPXTensor* f = SNEPPX_tensor_floor(a);
    float* fd = (float*)f->data;
    SX_ASSERT_NEAR(fd[0], 1.0f, 1e-6f, "floor(1.4)==1");
    SX_ASSERT_NEAR(fd[2], -2.0f, 1e-6f, "floor(-1.4)==-2");
    SNEPPXTensor* c = SNEPPX_tensor_ceil(a);
    float* cd = (float*)c->data;
    SX_ASSERT_NEAR(cd[1], 2.0f, 1e-6f, "ceil(1.6)==2");
    SX_ASSERT_NEAR(cd[2], -1.0f, 1e-6f, "ceil(-1.4)==-1");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(f); SNEPPX_tensor_destroy(c);
}

static void test_trig_ops(void) {
    size_t shape[] = {1};
    float pi = 3.14159265f;
    SNEPPXTensor* a = SNEPPX_tensor_full(NULL, 0, SNEPPX_FLOAT32, &pi);
    SNEPPXTensor* s = SNEPPX_tensor_sin(a);
    SX_ASSERT(s != NULL, "sin result not null");
    SX_ASSERT_NEAR(((float*)s->data)[0], 0.0f, 1e-5f, "sin(pi)==0");
    SNEPPXTensor* c = SNEPPX_tensor_cos(a);
    SX_ASSERT(c != NULL, "cos result not null");
    SX_ASSERT_NEAR(((float*)c->data)[0], -1.0f, 1e-5f, "cos(pi)==-1");
    float half_pi = pi / 2.0f;
    SNEPPXTensor* h = SNEPPX_tensor_full(NULL, 0, SNEPPX_FLOAT32, &half_pi);
    SNEPPXTensor* sn = SNEPPX_tensor_sin(h);
    SX_ASSERT_NEAR(((float*)sn->data)[0], 1.0f, 1e-5f, "sin(pi/2)==1");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(s); SNEPPX_tensor_destroy(c);
    SNEPPX_tensor_destroy(h); SNEPPX_tensor_destroy(sn);
}

static void test_tanh_sigmoid(void) {
    size_t shape[] = {2};
    float vals[] = {0.0f, 1.0f};
    SNEPPXTensor* a = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    memcpy(a->data, vals, 2 * sizeof(float));
    SNEPPXTensor* t = SNEPPX_tensor_tanh(a);
    SX_ASSERT(t != NULL, "tanh result not null");
    SX_ASSERT_NEAR(((float*)t->data)[0], 0.0f, 1e-5f, "tanh(0)==0");
    SX_ASSERT_NEAR(((float*)t->data)[1], tanhf(1.0f), 1e-5f, "tanh(1)");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(t);
}

/* ---------- Comparison Ops ---------- */

static void test_eq_basic(void) {
    size_t shape[] = {3};
    SNEPPXTensor* a = SNEPPX_tensor_arange(1.0f, 4.0f, 1.0f, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_eq(a, a);
    SX_ASSERT(r != NULL, "eq result not null");
    uint8_t* d = (uint8_t*)r->data;
    SX_ASSERT(d[0] == 1, "eq self true");
    SX_ASSERT(r->dtype == SNEPPX_BOOL, "eq dtype bool");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(r);
}

static void test_lt_basic(void) {
    size_t shape[] = {3};
    float vals[] = {3.0f, 3.0f, 3.0f};
    SNEPPXTensor* a = SNEPPX_tensor_arange(1.0f, 4.0f, 1.0f, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    memcpy(b->data, vals, 3 * sizeof(float));
    SNEPPXTensor* r = SNEPPX_tensor_lt(a, b);
    SX_ASSERT(r != NULL, "lt result not null");
    uint8_t* d = (uint8_t*)r->data;
    SX_ASSERT(d[0] == 1, "1<3 true");
    SX_ASSERT(d[2] == 0, "3<3 false");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b); SNEPPX_tensor_destroy(r);
}

static void test_gt_basic(void) {
    size_t shape[] = {3};
    float vals[] = {2.0f, 2.0f, 2.0f};
    SNEPPXTensor* a = SNEPPX_tensor_arange(1.0f, 4.0f, 1.0f, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    memcpy(b->data, vals, 3 * sizeof(float));
    SNEPPXTensor* r = SNEPPX_tensor_gt(a, b);
    SX_ASSERT(r != NULL, "gt result not null");
    uint8_t* d = (uint8_t*)r->data;
    SX_ASSERT(d[0] == 0, "1>2 false");
    SX_ASSERT(d[2] == 1, "3>2 true");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b); SNEPPX_tensor_destroy(r);
}

static void test_null_inputs(void) {
    SX_ASSERT(SNEPPX_tensor_add(NULL, NULL) == NULL, "add null returns NULL");
    SX_ASSERT(SNEPPX_tensor_neg(NULL) == NULL, "neg null returns NULL");
    SX_ASSERT(SNEPPX_tensor_abs(NULL) == NULL, "abs null returns NULL");
    SX_ASSERT(SNEPPX_tensor_sin(NULL) == NULL, "sin null returns NULL");
    SX_ASSERT(SNEPPX_tensor_eq(NULL, NULL) == NULL, "eq null returns NULL");
    SX_ASSERT(SNEPPX_tensor_div(NULL, NULL) == NULL, "div null returns NULL");
}


TEST(test_tensor_ops, test_add_basic) { test_add_basic(); }
TEST(test_tensor_ops, test_add_broadcast) { test_add_broadcast(); }
TEST(test_tensor_ops, test_sub_basic) { test_sub_basic(); }
TEST(test_tensor_ops, test_mul_basic) { test_mul_basic(); }
TEST(test_tensor_ops, test_div_basic) { test_div_basic(); }
TEST(test_tensor_ops, test_pow_basic) { test_pow_basic(); }
TEST(test_tensor_ops, test_neg_basic) { test_neg_basic(); }
TEST(test_tensor_ops, test_abs_basic) { test_abs_basic(); }
TEST(test_tensor_ops, test_sign_basic) { test_sign_basic(); }
TEST(test_tensor_ops, test_exp_log_sqrt) { test_exp_log_sqrt(); }
TEST(test_tensor_ops, test_round_ops) { test_round_ops(); }
TEST(test_tensor_ops, test_trig_ops) { test_trig_ops(); }
TEST(test_tensor_ops, test_tanh_sigmoid) { test_tanh_sigmoid(); }
TEST(test_tensor_ops, test_eq_basic) { test_eq_basic(); }
TEST(test_tensor_ops, test_lt_basic) { test_lt_basic(); }
TEST(test_tensor_ops, test_gt_basic) { test_gt_basic(); }
TEST(test_tensor_ops, test_null_inputs) { test_null_inputs(); }
