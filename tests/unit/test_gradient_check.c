#include "automatic_differentiation_framework.h"
#include "test_gtest.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Test Gradient Check
 *
 * WHAT
 *   Test Gradient Check.
 *
 * CONCEPT
 *   Provides the Test Gradient Check.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_grad_add(void) {
    size_t sh[] = {3};
    float ad[] = {1.0f, 2.0f, 3.0f};
    float bd[] = {4.0f, 5.0f, 6.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, ad, 12); memcpy(tb->data, bd, 12);
    SNEPPXVariable *a = SNEPPX_variable_create(ta, 1), *b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_add(tape, a, b);
    SNEPPX_tape_backward(tape, c);
    for (size_t i = 0; i < 3; i++) {
        SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[i], 1.0f), "add da = 1");
        SX_ASSERT(SX_FLOAT_CLOSE(((float*)b->grad->data)[i], 1.0f), "add db = 1");
    }
    SNEPPX_tape_destroy(tape);
}

static void test_grad_mul(void) {
    size_t sh[] = {2};
    float ad[] = {3.0f, 4.0f};
    float bd[] = {5.0f, 6.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, ad, 8); memcpy(tb->data, bd, 8);
    SNEPPXVariable *a = SNEPPX_variable_create(ta, 1), *b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_mul(tape, a, b);
    SNEPPX_tape_backward(tape, c);
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[0], 5.0f) && SX_FLOAT_CLOSE(((float*)a->grad->data)[1], 6.0f), "mul da = b");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)b->grad->data)[0], 3.0f) && SX_FLOAT_CLOSE(((float*)b->grad->data)[1], 4.0f), "mul db = a");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_matmul(void) {
    size_t sha[] = {2,3}, shb[] = {3,2};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sha, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(shb, 2, SNEPPX_FLOAT32);
    for (size_t i = 0; i < 6; i++) { ((float*)ta->data)[i] = (float)(i+1); ((float*)tb->data)[i] = (float)(i+7); }
    SNEPPXVariable *a = SNEPPX_variable_create(ta, 1), *b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_matmul(tape, a, b);
    SNEPPX_tape_backward(tape, c);
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[0], 15), "matmul da[0]=15");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[5], 23), "matmul da[5]=23");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)b->grad->data)[0], 5), "matmul db[0]=5");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)b->grad->data)[5], 9), "matmul db[5]=9");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_div(void) {
    size_t sh[] = {2};
    float ad[] = {10.0f, 20.0f}, bd[] = {2.0f, 5.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, ad, 8); memcpy(tb->data, bd, 8);
    SNEPPXVariable *a = SNEPPX_variable_create(ta, 1), *b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_div(tape, a, b);
    SNEPPX_tape_backward(tape, c);
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[0], 0.5f), "div da = 1/b");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)b->grad->data)[0], -2.5f), "div db = -a/b^2");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_relu(void) {
    size_t sh[] = {3};
    float ad[] = {-2.0f, 0.0f, 5.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, ad, 12);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* r = SNEPPX_relu(tape, a);
    SNEPPX_tape_backward(tape, r);
    float* ag = (float*)a->grad->data;
    SX_ASSERT(SX_FLOAT_CLOSE(ag[0], 0.0f) && SX_FLOAT_CLOSE(ag[1], 0.0f) && SX_FLOAT_CLOSE(ag[2], 1.0f), "relu grad");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_sigmoid(void) {
    size_t sh[] = {1};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    ((float*)ta->data)[0] = 0.0f;
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* s = SNEPPX_sigmoid(tape, a);
    SNEPPX_tape_backward(tape, s);
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[0], 0.25f), "sigmoid grad at 0 = 0.25");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_tanh(void) {
    size_t sh[] = {1};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    ((float*)ta->data)[0] = 0.0f;
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* h = SNEPPX_tanh(tape, a);
    SNEPPX_tape_backward(tape, h);
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[0], 1.0f), "tanh grad at 0 = 1");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_exp_log(void) {
    size_t sh[] = {2};
    float ad[] = {0.0f, 1.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, ad, 8);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* e = SNEPPX_exp(tape, a);
    SNEPPX_tape_backward(tape, e);
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[0], 1.0f) && SX_FLOAT_CLOSE(((float*)a->grad->data)[1], 2.71828f), "exp grad");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_pow(void) {
    size_t sh[] = {2};
    float ad[] = {2.0f, 3.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, ad, 8);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    ((float*)tb->data)[0] = 2.0f; ((float*)tb->data)[1] = 3.0f;
    SNEPPXVariable *a = SNEPPX_variable_create(ta, 1), *b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* p = SNEPPX_pow(tape, a, b);
    SNEPPX_tape_backward(tape, p);
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[0], 4.0f), "pow da = b*a^(b-1): 4");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[1], 27.0f), "pow da: 3*3^2=27");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_sin_cos(void) {
    size_t sh[] = {2};
    float ad[] = {0.0f, 1.570796f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, ad, 8);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* s = SNEPPX_sin(tape, a);
    SNEPPX_tape_backward(tape, s);
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[0], 1.0f), "sin grad at 0 = cos(0) = 1");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[1], 0.0f), "sin grad at pi/2 = cos(pi/2) = 0");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_chain_shared_input(void) {
    size_t sh[] = {2};
    float ad[] = {2.0f, 3.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, ad, 8);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_add(tape, a, b);
    SNEPPXVariable* d = SNEPPX_mul(tape, c, a);
    SNEPPXVariable* loss = SNEPPX_sum(tape, d, 0);
    SNEPPX_tape_backward(tape, loss);
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[0], 4.0f), "chain shared a[0]=4");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[1], 6.0f), "chain shared a[1]=6");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_mse(void) {
    size_t sh[] = {3};
    SNEPPXTensor* tp = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tt = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    ((float*)tp->data)[0] = 1; ((float*)tp->data)[1] = 2; ((float*)tp->data)[2] = 3;
    SNEPPXVariable* pred = SNEPPX_variable_create(tp, 1);
    SNEPPXVariable* target = SNEPPX_variable_create(tt, 0);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* loss = SNEPPX_mse_loss(tape, pred, target);
    SNEPPX_tape_backward(tape, loss);
    float* pg = (float*)pred->grad->data;
    SX_ASSERT(SX_FLOAT_CLOSE(pg[0], 2.0f/3.0f) && SX_FLOAT_CLOSE(pg[1], 4.0f/3.0f) && SX_FLOAT_CLOSE(pg[2], 2.0f), "mse grad = 2*(pred-target)/N");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_softmax(void) {
    size_t sh[] = {3};
    float ad[] = {1.0f, 2.0f, 3.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, ad, 12);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* s = SNEPPX_softmax(tape, a, 0);
    SNEPPX_tape_backward(tape, s);
    float* ag = (float*)a->grad->data;
    float sum = ag[0] + ag[1] + ag[2];
    SX_ASSERT(SX_FLOAT_CLOSE(sum, 0.0f), "softmax grads sum to 0");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_cross_entropy(void) {
    size_t pred_sh[] = {2, 3}, tgt_sh[] = {2};
    float pd[] = {1.0f, 2.0f, 3.0f, 1.0f, 0.5f, 0.1f};
    float td[] = {1.0f, 0.0f};
    SNEPPXTensor* tp = SNEPPX_tensor_zeros(pred_sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tt = SNEPPX_tensor_zeros(tgt_sh, 1, SNEPPX_FLOAT32);
    memcpy(tp->data, pd, 24);
    memcpy(tt->data, td, 8);
    SNEPPXVariable* pred = SNEPPX_variable_create(tp, 1);
    SNEPPXVariable* target = SNEPPX_variable_create(tt, 0);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* loss = SNEPPX_cross_entropy(tape, pred, target);
    SNEPPX_tape_backward(tape, loss);
    SX_ASSERT(pred->grad != NULL, "cross_entropy grad exists");
    SX_ASSERT(pred->grad->size == 6, "cross_entropy grad size matches pred");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_embedding(void) {
    size_t wsh[] = {4, 2};
    SNEPPXTensor* tw = SNEPPX_tensor_zeros(wsh, 2, SNEPPX_FLOAT32);
    for (size_t i = 0; i < 8; i++) ((float*)tw->data)[i] = (float)(i + 1);
    size_t ish[] = {3};
    SNEPPXTensor* ti = SNEPPX_tensor_zeros(ish, 1, SNEPPX_INT32);
    size_t idx_vals[] = {0, 2, 1};
    SNEPPXVariable* w = SNEPPX_variable_create(tw, 1);
    SNEPPXVariable* idx = SNEPPX_variable_create(ti, 0);
    /* index tensor must carry integer values (read_index honors INT32 via
       dtype) instead of size_t bit-patterns memcpy'd into a float buffer. */
    int32_t* iv = (int32_t*)ti->data;
    iv[0] = (int32_t)idx_vals[0]; iv[1] = (int32_t)idx_vals[1]; iv[2] = (int32_t)idx_vals[2];
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* e = SNEPPX_embedding(tape, w, idx);
    SNEPPX_tape_backward(tape, e);
    SX_ASSERT(w->grad != NULL, "embedding grad exists");
    float* wg = (float*)w->grad->data;
    SX_ASSERT(SX_FLOAT_CLOSE(wg[0], 1.0f) && SX_FLOAT_CLOSE(wg[1], 1.0f) && SX_FLOAT_CLOSE(wg[2], 1.0f), "embedding grad at used indices");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_layer_norm(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tg = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(sh, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* g = SNEPPX_variable_create(tg, 1);
    SNEPPXVariable* beta = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* ln = SNEPPX_layer_norm(tape, a, g, beta, 1e-5f);
    SNEPPX_tape_backward(tape, ln);
    SX_ASSERT(a->grad != NULL, "ln a grad");
    SX_ASSERT(g->grad != NULL, "ln gamma grad");
    SX_ASSERT(beta->grad != NULL, "ln beta grad");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_multiple_consumers(void) {
    size_t sh[] = {2};
    float ad[] = {2.0f, 3.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, ad, 8);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* b = SNEPPX_mul(tape, a, a);
    SNEPPXVariable* c = SNEPPX_exp(tape, a);
    SNEPPXVariable* d = SNEPPX_add(tape, b, c);
    SNEPPXVariable* loss = SNEPPX_sum(tape, d, 0);
    SNEPPX_tape_backward(tape, loss);
    float* ag = (float*)a->grad->data;
    SX_ASSERT(SX_FLOAT_CLOSE(ag[0], 4.0f + 7.389f), "multi-consumer grad da = 2*a + exp(a)");
    SX_ASSERT(SX_FLOAT_CLOSE(ag[1], 6.0f + 20.085f), "multi-consumer grad da = 2*a + exp(a)");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_no_grad_leaves_zero(void) {
    size_t sh[] = {2};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 0);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_add(tape, a, b);
    SNEPPX_tape_backward(tape, c);
    SX_ASSERT(a->grad == NULL, "no_grad a has no grad");
    SX_ASSERT(b->grad != NULL, "no_grad b has grad");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_transpose(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 2, SNEPPX_FLOAT32);
    for (size_t i = 0; i < 6; i++) ((float*)ta->data)[i] = (float)(i + 1);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* t = SNEPPX_transpose(tape, a, 0, 1);
    SNEPPX_tape_backward(tape, t);
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[0], 1.0f), "transpose grad preserved");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[2], 1.0f), "transpose grad preserved");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_dropout(void) {
    size_t sh[] = {10};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* d = SNEPPX_dropout(tape, a, 0.5f, 42);
    SNEPPX_tape_backward(tape, d);
    SX_ASSERT(a->grad != NULL, "dropout grad exists");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_neg(void) {
    size_t sh[] = {2};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    ((float*)ta->data)[0] = 5.0f; ((float*)ta->data)[1] = -3.0f;
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* n = SNEPPX_neg(tape, a);
    SNEPPX_tape_backward(tape, n);
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[0], -1.0f), "neg grad = -1");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[1], -1.0f), "neg grad = -1");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_sum_mean(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 2, SNEPPX_FLOAT32);
    for (size_t i = 0; i < 6; i++) ((float*)ta->data)[i] = (float)(i+1);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* m = SNEPPX_mean(tape, a, 1);
    SNEPPX_tape_backward(tape, m);
    SX_ASSERT(a->grad != NULL, "sum+mean grad exists");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_sqrt_abs(void) {
    size_t sh[] = {2};
    float ad[] = {4.0f, 9.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, ad, 8);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* r = SNEPPX_sqrt(tape, a);
    SNEPPX_tape_backward(tape, r);
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[0], 0.25f), "sqrt grad = 0.5/sqrt(x): 0.25");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[1], 1.0f/6.0f), "sqrt grad = 0.5/sqrt(9) = 1/6");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_var_std(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    ((float*)ta->data)[0] = 1; ((float*)ta->data)[1] = 2; ((float*)ta->data)[2] = 3;
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* v = SNEPPX_var(tape, a, 0);
    SNEPPX_tape_backward(tape, v);
    SX_ASSERT(a->grad != NULL, "var grad exists");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_log_softmax(void) {
    size_t sh[] = {3};
    float ad[] = {1.0f, 2.0f, 3.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, ad, 12);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* l = SNEPPX_log_softmax(tape, a, 0);
    SNEPPX_tape_backward(tape, l);
    SX_ASSERT(a->grad != NULL, "log_softmax grad exists");
    float sum = 0;
    for (size_t i = 0; i < 3; i++) sum += ((float*)a->grad->data)[i];
    SX_ASSERT(SX_FLOAT_CLOSE(sum, 0.0f), "log_softmax grads sum to 0");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_concat(void) {
    size_t sh1[] = {2, 2}, sh2[] = {2, 3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh1, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(sh2, 2, SNEPPX_FLOAT32);
    SNEPPXVariable *a = SNEPPX_variable_create(ta, 1), *b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* vars[] = {a, b};
    SNEPPXVariable* c = SNEPPX_concat(tape, vars, 2, 1);
    SNEPPXVariable* s = SNEPPX_sum(tape, c, 0);
    SNEPPX_tape_backward(tape, s);
    SX_ASSERT(a->grad != NULL && b->grad != NULL, "concat grad exists");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_silu_gelu(void) {
    size_t sh[] = {2};
    float ad[] = {0.0f, 1.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, ad, 8);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* s = SNEPPX_silu(tape, a);
    SNEPPX_tape_backward(tape, s);
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[0], 0.5f), "silu grad at 0 = 0.5");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_numerical_stability(void) {
    size_t sh[] = {3};
    float ad[] = {0.0f, -1.0f, 1e-10f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, ad, 12);
    ((float*)tb->data)[0] = 2.0f; ((float*)tb->data)[1] = 2.0f; ((float*)tb->data)[2] = 2.0f;
    SNEPPXVariable *a = SNEPPX_variable_create(ta, 1), *b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* d = SNEPPX_div(tape, a, b);
    (void)d;
    SNEPPXVariable* l = SNEPPX_log(tape, SNEPPX_abs(tape, a));
    (void)l;
    SNEPPXVariable* s = SNEPPX_sqrt(tape, SNEPPX_abs(tape, a));
    (void)s;
    SNEPPX_tape_backward(tape, s);
    SX_ASSERT(a->grad != NULL, "numerical stability: grad exists");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_tape_clear(void) {
    size_t sh[] = {2};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPX_tape_record(tape, a);
    SNEPPX_tape_record(tape, b);
    SNEPPXVariable* c = SNEPPX_add(tape, a, b);
    SNEPPX_tape_backward(tape, c);
    SX_ASSERT(a->grad != NULL, "grad exists before clear");
    SNEPPX_tape_zero_grad(tape);
    SX_ASSERT(a->grad == NULL, "grad cleared");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_global_norm(void) {
    size_t sh[] = {2};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    ((float*)ta->data)[0] = 2; ((float*)ta->data)[1] = 3;
    ((float*)tb->data)[0] = 0; ((float*)tb->data)[1] = 4;
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPX_tape_record(tape, a);
    SNEPPX_tape_record(tape, b);
    SNEPPXVariable* c = SNEPPX_add(tape, a, b);
    SNEPPX_tape_backward(tape, c);
    if (c->grad) { SNEPPX_tensor_destroy(c->grad); c->grad = NULL; }
    float gn = SNEPPX_tape_global_norm(tape);
    SX_ASSERT(SX_FLOAT_CLOSE(gn, sqrtf(4)), "global norm = 2");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_clip_grad_norm(void) {
    size_t sh[] = {2};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    ((float*)ta->data)[0] = 10; ((float*)ta->data)[1] = 0;
    ((float*)tb->data)[0] = 0; ((float*)tb->data)[1] = 0;
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPX_tape_record(tape, a);
    SNEPPX_tape_record(tape, b);
    SNEPPXVariable* c = SNEPPX_add(tape, a, b);
    SNEPPX_tape_backward(tape, c);
    if (c->grad) { SNEPPX_tensor_destroy(c->grad); c->grad = NULL; }
    SNEPPX_tape_clip_grad_norm(tape, 1.0f);
    float gn = SNEPPX_tape_global_norm(tape);
    SX_ASSERT(SX_FLOAT_CLOSE(gn, 1.0f), "clipped norm = 1");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_minimum(void) {
    size_t sh[] = {3};
    float ad[] = {1.0f, 5.0f, 3.0f};
    float bd[] = {4.0f, 2.0f, 6.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, ad, 12); memcpy(tb->data, bd, 12);
    SNEPPXVariable *a = SNEPPX_variable_create(ta, 1), *b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_minimum(tape, a, b);
    SNEPPX_tape_backward(tape, c);
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[0], 1.0f), "min a[0]=1 (< b[0])");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[1], 0.0f), "min a[1]=0 (> b[1])");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[2], 1.0f), "min a[2]=1 (< b[2])");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)b->grad->data)[0], 0.0f), "min b[0]=0 (> a[0])");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)b->grad->data)[1], 1.0f), "min b[1]=1 (< a[1])");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)b->grad->data)[2], 0.0f), "min b[2]=0 (> a[2])");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_maximum(void) {
    size_t sh[] = {3};
    float ad[] = {1.0f, 5.0f, 3.0f};
    float bd[] = {4.0f, 2.0f, 6.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, ad, 12); memcpy(tb->data, bd, 12);
    SNEPPXVariable *a = SNEPPX_variable_create(ta, 1), *b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_maximum(tape, a, b);
    SNEPPX_tape_backward(tape, c);
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[0], 0.0f), "max a[0]=0 (< b[0])");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[1], 1.0f), "max a[1]=1 (> b[1])");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)a->grad->data)[2], 0.0f), "max a[2]=0 (< b[2])");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)b->grad->data)[0], 1.0f), "max b[0]=1 (> a[0])");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)b->grad->data)[1], 0.0f), "max b[1]=0 (< a[1])");
    SX_ASSERT(SX_FLOAT_CLOSE(((float*)b->grad->data)[2], 1.0f), "max b[2]=1 (> a[2])");
    SNEPPX_tape_destroy(tape);
}

static void test_grad_conv2d(void) {
    size_t ishape[] = {1, 1, 4, 4};
    size_t kshape[] = {1, 1, 2, 2};
    SNEPPXTensor* input = SNEPPX_tensor_zeros(ishape, 4, SNEPPX_FLOAT32);
    SNEPPXTensor* kernel = SNEPPX_tensor_zeros(kshape, 4, SNEPPX_FLOAT32);
    float* xd = (float*)input->data;
    float* kd = (float*)kernel->data;
    for (size_t i = 0; i < 16; i++) xd[i] = (float)((i * 7 + 3) % 13) / 13.0f;
    for (size_t i = 0; i < 4; i++) kd[i] = (float)((i * 13 + 7) % 11) / 11.0f;

    SNEPPXVariable *va = SNEPPX_variable_create(input, 1);
    SNEPPXVariable *vk = SNEPPX_variable_create(kernel, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* conv = SNEPPX_conv2d(tape, va, vk, 1, 1, 0, 0);
    size_t N_conv = conv->data->size;
    SNEPPXTensor* tgt_data = SNEPPX_tensor_zeros(conv->data->shape, conv->data->ndim, SNEPPX_FLOAT32);
    SNEPPXVariable* target = SNEPPX_variable_create(tgt_data, 0);
    SNEPPXVariable* loss = SNEPPX_mse_loss(tape, conv, target);
    SNEPPX_tape_backward(tape, loss);

    for (size_t idx = 0; idx < 16; idx++) {
        float orig = xd[idx];
        float eps = 1e-4f;
        xd[idx] = orig + eps;
        SNEPPXTensor* out_p = SNEPPX_tensor_conv2d(input, kernel, 1, 1, 0, 0);
        float lp = 0;
        for (size_t i = 0; i < out_p->size; i++) { float v = ((float*)out_p->data)[i]; lp += v * v / (float)N_conv; }
        SNEPPX_tensor_destroy(out_p);
        xd[idx] = orig - eps;
        SNEPPXTensor* out_m = SNEPPX_tensor_conv2d(input, kernel, 1, 1, 0, 0);
        float lm = 0;
        for (size_t i = 0; i < out_m->size; i++) { float v = ((float*)out_m->data)[i]; lm += v * v / (float)N_conv; }
        SNEPPX_tensor_destroy(out_m);
        xd[idx] = orig;
        float fd = (lp - lm) / (2.0f * eps);
        float ad = ((float*)va->grad->data)[idx];
        float ratio = fabsf(fd) > 1e-6f ? fabsf(ad - fd) / (fabsf(fd) + 1e-6f) : fabsf(ad - fd);
        SX_ASSERT(ratio < 0.15f, "conv2d input grad finite-diff match");
    }

    for (size_t idx = 0; idx < 4; idx++) {
        float orig = kd[idx];
        float eps = 1e-4f;
        kd[idx] = orig + eps;
        SNEPPXTensor* out_p = SNEPPX_tensor_conv2d(input, kernel, 1, 1, 0, 0);
        float lp = 0;
        for (size_t i = 0; i < out_p->size; i++) { float v = ((float*)out_p->data)[i]; lp += v * v / (float)N_conv; }
        SNEPPX_tensor_destroy(out_p);
        kd[idx] = orig - eps;
        SNEPPXTensor* out_m = SNEPPX_tensor_conv2d(input, kernel, 1, 1, 0, 0);
        float lm = 0;
        for (size_t i = 0; i < out_m->size; i++) { float v = ((float*)out_m->data)[i]; lm += v * v / (float)N_conv; }
        SNEPPX_tensor_destroy(out_m);
        kd[idx] = orig;
        float fd = (lp - lm) / (2.0f * eps);
        float ad = ((float*)vk->grad->data)[idx];
        float ratio = fabsf(fd) > 1e-6f ? fabsf(ad - fd) / (fabsf(fd) + 1e-6f) : fabsf(ad - fd);
        SX_ASSERT(ratio < 0.15f, "conv2d kernel grad finite-diff match");
    }

    SNEPPX_variable_destroy(target);
    SNEPPX_tape_destroy(tape);
}


TEST(test_gradient_check, add_grad) { test_grad_add(); }
TEST(test_gradient_check, mul_grad) { test_grad_mul(); }
TEST(test_gradient_check, matmul_grad) { test_grad_matmul(); }
TEST(test_gradient_check, div_grad) { test_grad_div(); }
TEST(test_gradient_check, relu_grad) { test_grad_relu(); }
TEST(test_gradient_check, sigmoid_grad) { test_grad_sigmoid(); }
TEST(test_gradient_check, tanh_grad) { test_grad_tanh(); }
TEST(test_gradient_check, exp_log_grad) { test_grad_exp_log(); }
TEST(test_gradient_check, pow_grad) { test_grad_pow(); }
TEST(test_gradient_check, sin_cos_grad) { test_grad_sin_cos(); }
TEST(test_gradient_check, chain_shared_input) { test_grad_chain_shared_input(); }
TEST(test_gradient_check, mse_loss_grad) { test_grad_mse(); }
TEST(test_gradient_check, softmax_grad) { test_grad_softmax(); }
TEST(test_gradient_check, cross_entropy_grad) { test_grad_cross_entropy(); }
TEST(test_gradient_check, embedding_grad) { test_grad_embedding(); }
TEST(test_gradient_check, layer_norm_grad) { test_grad_layer_norm(); }
TEST(test_gradient_check, multi_consumer_grad) { test_grad_multiple_consumers(); }
TEST(test_gradient_check, no_grad_leaves_zero) { test_grad_no_grad_leaves_zero(); }
TEST(test_gradient_check, transpose_grad) { test_grad_transpose(); }
TEST(test_gradient_check, dropout_grad) { test_grad_dropout(); }
TEST(test_gradient_check, neg_grad) { test_grad_neg(); }
TEST(test_gradient_check, sum_mean_grad) { test_grad_sum_mean(); }
TEST(test_gradient_check, sqrt_abs_grad) { test_grad_sqrt_abs(); }
TEST(test_gradient_check, var_std_grad) { test_grad_var_std(); }
TEST(test_gradient_check, log_softmax_grad) { test_grad_log_softmax(); }
TEST(test_gradient_check, concat_grad) { test_grad_concat(); }
TEST(test_gradient_check, silu_gelu_grad) { test_grad_silu_gelu(); }
TEST(test_gradient_check, numerical_stability) { test_grad_numerical_stability(); }
TEST(test_gradient_check, tape_zero_grad) { test_grad_tape_clear(); }
TEST(test_gradient_check, global_norm) { test_grad_global_norm(); }
TEST(test_gradient_check, clip_grad_norm) { test_grad_clip_grad_norm(); }
TEST(test_gradient_check, min_grad) { test_grad_minimum(); }
TEST(test_gradient_check, max_grad) { test_grad_maximum(); }
TEST(test_gradient_check, conv2d_grad) { test_grad_conv2d(); }
