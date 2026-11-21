#include "automatic_differentiation_framework.h"
#include "test_gtest.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Test Autodiff
 *
 * WHAT
 *   Test Autodiff.
 *
 * CONCEPT
 *   Provides the Test Autodiff.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */




static int float_close(float a, float b) { return fabsf(a - b) < 1e-3f; }

static void test_variable_create(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT(t != NULL, "tensor created");
    SNEPPXVariable* v = SNEPPX_variable_create(t, 1);
    SX_ASSERT(v != NULL, "var created");
    SX_ASSERT(v->data == t, "data matches");
    SX_ASSERT(v->requires_grad == 1, "requires_grad set");
    SX_ASSERT(v->grad == NULL, "grad is NULL");
    SX_ASSERT(v->backward_fn == NULL, "backward_fn NULL");
    SNEPPX_variable_destroy(v);
}

static void test_add_forward(void) {
    size_t shape[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_add(tape, a, b);
    SX_ASSERT(c != NULL, "add result");
    float* cd = (float*)c->data->data;
    SX_ASSERT(float_close(cd[0], 2.0f) && float_close(cd[1], 2.0f) && float_close(cd[2], 2.0f), "add values");
    SNEPPX_tape_destroy(tape);
}

static void test_add_backward(void) {
    size_t shape[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_add(tape, a, b);
    SNEPPX_tape_backward(tape, c);
    SX_ASSERT(a->grad != NULL, "a has grad");
    SX_ASSERT(b->grad != NULL, "b has grad");
    float* ag = (float*)a->grad->data;
    float* bg = (float*)b->grad->data;
    SX_ASSERT(float_close(ag[0], 1.0f) && float_close(ag[1], 1.0f) && float_close(ag[2], 1.0f), "a grad = 1");
    SX_ASSERT(float_close(bg[0], 1.0f) && float_close(bg[1], 1.0f) && float_close(bg[2], 1.0f), "b grad = 1");
    SNEPPX_tape_destroy(tape);
}

static void test_mul_backward(void) {
    size_t shape[] = {3};
    float adata[] = {2.0f, 3.0f, 4.0f};
    float bdata[] = {5.0f, 6.0f, 7.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, adata, 3 * sizeof(float));
    memcpy(tb->data, bdata, 3 * sizeof(float));
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_mul(tape, a, b);
    SNEPPX_tape_backward(tape, c);
    float* ag = (float*)a->grad->data;
    float* bg = (float*)b->grad->data;
    SX_ASSERT(float_close(ag[0], 5.0f) && float_close(ag[1], 6.0f) && float_close(ag[2], 7.0f), "a grad = b");
    SX_ASSERT(float_close(bg[0], 2.0f) && float_close(bg[1], 3.0f) && float_close(bg[2], 4.0f), "b grad = a");
    SNEPPX_tape_destroy(tape);
}

static void test_matmul_backward(void) {
    size_t sha[] = {2, 3}, shb[] = {3, 2};
    float adata[] = {1,2,3,4,5,6};
    float bdata[] = {7,8,9,10,11,12};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(sha, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(shb, 2, SNEPPX_FLOAT32);
    memcpy(ta->data, adata, 6 * sizeof(float));
    memcpy(tb->data, bdata, 6 * sizeof(float));
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_matmul(tape, a, b);
    SNEPPX_tape_backward(tape, c);
    SX_ASSERT(a->grad != NULL, "a grad exists");
    SX_ASSERT(b->grad != NULL, "b grad exists");
    float* ag = (float*)a->grad->data;
    float* bg = (float*)b->grad->data;
    SX_ASSERT(float_close(ag[0], 15.0f) && float_close(ag[5], 23.0f), "matmul grad a");
    SX_ASSERT(float_close(bg[0], 5.0f) && float_close(bg[5], 9.0f), "matmul grad b");
    SNEPPX_tape_destroy(tape);
}

static void test_mse_loss_backward(void) {
    size_t shape[] = {3};
    float pdata[] = {1.0f, 2.0f, 3.0f};
    float tdata[] = {0.0f, 0.0f, 0.0f};
    SNEPPXTensor* tp = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tt = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    memcpy(tp->data, pdata, 3 * sizeof(float));
    memcpy(tt->data, tdata, 3 * sizeof(float));
    SNEPPXVariable* pred = SNEPPX_variable_create(tp, 1);
    SNEPPXVariable* target = SNEPPX_variable_create(tt, 0);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* loss = SNEPPX_mse_loss(tape, pred, target);
    SX_ASSERT(loss != NULL, "mse loss");
    SNEPPX_tape_backward(tape, loss);
    SX_ASSERT(pred->grad != NULL, "pred has grad");
    float* pg = (float*)pred->grad->data;
    SX_ASSERT(float_close(pg[0], 2.0f/3.0f), "mse grad");
    SNEPPX_tape_destroy(tape);
}

static void test_relu_backward(void) {
    size_t shape[] = {4};
    float adata[] = {-2.0f, 0.0f, 3.0f, 5.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, adata, 4 * sizeof(float));
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* r = SNEPPX_relu(tape, a);
    SNEPPX_tape_backward(tape, r);
    float* ag = (float*)a->grad->data;
    SX_ASSERT(float_close(ag[0], 0.0f) && float_close(ag[1], 0.0f), "relu grad neg=0");
    SX_ASSERT(float_close(ag[2], 1.0f) && float_close(ag[3], 1.0f), "relu grad pos=1");
    SNEPPX_tape_destroy(tape);
}

static void test_sigmoid_backward(void) {
    size_t shape[] = {2};
    float adata[] = {0.0f, 1.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* s = SNEPPX_sigmoid(tape, a);
    SNEPPX_tape_backward(tape, s);
    SX_ASSERT(a->grad != NULL, "sigmoid grad");
    float* ag = (float*)a->grad->data;
    SX_ASSERT(float_close(ag[0], 0.25f), "sigmoid grad at 0 = 0.25");
    SX_ASSERT(float_close(ag[1], 0.1966f), "sigmoid grad at 1 ~ 0.1966");
    SNEPPX_tape_destroy(tape);
}

static void test_tanh_backward(void) {
    size_t shape[] = {2};
    float adata[] = {0.0f, 1.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* h = SNEPPX_tanh(tape, a);
    SNEPPX_tape_backward(tape, h);
    SX_ASSERT(a->grad != NULL, "tanh grad");
    float* ag = (float*)a->grad->data;
    SX_ASSERT(float_close(ag[0], 1.0f), "tanh grad at 0 = 1");
    SX_ASSERT(float_close(ag[1], 0.4199f), "tanh grad at 1 ~ 0.4199");
    SNEPPX_tape_destroy(tape);
}

static void test_softmax_backward(void) {
    size_t shape[] = {3};
    float adata[] = {1.0f, 2.0f, 3.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, adata, 3 * sizeof(float));
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* s = SNEPPX_softmax(tape, a, 0);
    SX_ASSERT(s != NULL, "softmax ok");
    SNEPPX_tape_backward(tape, s);
    SX_ASSERT(a->grad != NULL, "softmax grad");
    float* ag = (float*)a->grad->data;
    float sum = 0;
    for (size_t i = 0; i < 3; i++) sum += ag[i];
    SX_ASSERT(float_close(sum, 0.0f), "softmax grads sum to 0");
    SNEPPX_tape_destroy(tape);
}

static void test_no_grad(void) {
    size_t shape[] = {2};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPX_no_grad_enter();
    SNEPPXVariable* c = SNEPPX_add(tape, a, b);
    SNEPPX_no_grad_exit();
    SX_ASSERT(c != NULL, "add in no_grad");
    SX_ASSERT(c->backward_fn == NULL, "no backward_fn in no_grad");
    SX_ASSERT(tape->num_vars == 0, "no tape record in no_grad");
    SNEPPX_variable_destroy(c);
    SNEPPX_tape_destroy(tape);
}

static void test_exp_backward(void) {
    size_t shape[] = {2};
    float adata[] = {0.0f, 1.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* e = SNEPPX_exp(tape, a);
    SNEPPX_tape_backward(tape, e);
    SX_ASSERT(a->grad != NULL, "exp grad");
    float* ag = (float*)a->grad->data;
    SX_ASSERT(float_close(ag[0], 1.0f), "exp grad at 0 = 1");
    SX_ASSERT(float_close(ag[1], 2.71828f), "exp grad at 1 = e");
    SNEPPX_tape_destroy(tape);
}

static void test_sum_backward(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* s = SNEPPX_sum(tape, a, 0);
    SNEPPX_tape_backward(tape, s);
    SX_ASSERT(a->grad != NULL, "sum grad");
    float* ag = (float*)a->grad->data;
    for (size_t i = 0; i < 6; i++) SX_ASSERT(float_close(ag[i], 1.0f), "sum grad=1 expanded");
    SNEPPX_tape_destroy(tape);
}

static void test_chain_rule(void) {
    size_t shape[] = {2};
    float adata[] = {2.0f, 3.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    memset(tb->data, 0, 2 * sizeof(float));
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_add(tape, a, b);
    SNEPPXVariable* d = SNEPPX_mul(tape, c, a);
    SNEPPXVariable* loss = SNEPPX_sum(tape, d, 0);
    SNEPPX_tape_backward(tape, loss);
    SX_ASSERT(a->grad != NULL, "chain a grad");
    float* ag = (float*)a->grad->data;
    SX_ASSERT(float_close(ag[0], 4.0f), "chain grad a[0]");
    SX_ASSERT(float_close(ag[1], 6.0f), "chain grad a[1]");
    SNEPPX_tape_destroy(tape);
}

static void test_dropout_backward(void) {
    size_t shape[] = {4};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* d = SNEPPX_dropout(tape, a, 0.5f, 42);
    SX_ASSERT(d != NULL, "dropout result");
    SX_ASSERT(d->backward_fn != NULL, "dropout has backward");
    SNEPPX_tape_backward(tape, d);
    SX_ASSERT(a->grad != NULL, "dropout grad exists");
    SNEPPX_tape_destroy(tape);
}

static void test_layer_norm_backward(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tg = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* g = SNEPPX_variable_create(tg, 1);
    SNEPPXVariable* beta = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* ln = SNEPPX_layer_norm(tape, a, g, beta, 1e-5f);
    SX_ASSERT(ln != NULL, "layer_norm result");
    SNEPPX_tape_backward(tape, ln);
    SX_ASSERT(a->grad != NULL, "ln grad a");
    SX_ASSERT(g->grad != NULL, "ln grad gamma");
    SNEPPX_tape_destroy(tape);
}

static void test_sub_backward(void) {
    size_t shape[] = {3};
    float adata[] = {5.0f, 6.0f, 7.0f};
    float bdata[] = {1.0f, 2.0f, 3.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, adata, 3 * sizeof(float));
    memcpy(tb->data, bdata, 3 * sizeof(float));
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_sub(tape, a, b);
    SNEPPX_tape_backward(tape, c);
    SX_ASSERT(a->grad != NULL, "sub a grad");
    SX_ASSERT(b->grad != NULL, "sub b grad");
    float* ag = (float*)a->grad->data;
    float* bg = (float*)b->grad->data;
    SX_ASSERT(float_close(ag[0], 1.0f), "sub da/dc=1");
    SX_ASSERT(float_close(bg[0], -1.0f), "sub db/dc=-1");
    SNEPPX_tape_destroy(tape);
}

static void test_div_backward(void) {
    size_t shape[] = {2};
    float adata[] = {8.0f, 9.0f};
    float bdata[] = {2.0f, 3.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    memcpy(tb->data, bdata, 2 * sizeof(float));
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_div(tape, a, b);
    SNEPPX_tape_backward(tape, c);
    SX_ASSERT(a->grad != NULL, "div a grad");
    SX_ASSERT(b->grad != NULL, "div b grad");
    float* ag = (float*)a->grad->data;
    float* bg = (float*)b->grad->data;
    SX_ASSERT(float_close(ag[0], 0.5f), "div da=1/b");
    SX_ASSERT(float_close(bg[0], -2.0f), "div db=-a/b^2");
    SNEPPX_tape_destroy(tape);
}

static void test_pow_backward(void) {
    size_t shape[] = {2};
    float adata[] = {2.0f, 3.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    ((float*)tb->data)[0] = 2.0f; ((float*)tb->data)[1] = 2.0f;
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_pow(tape, a, b);
    SNEPPX_tape_backward(tape, c);
    SX_ASSERT(a->grad != NULL, "pow a grad");
    float* ag = (float*)a->grad->data;
    SX_ASSERT(float_close(ag[0], 4.0f), "pow d(x^2)/dx=2x -> 4");
    SX_ASSERT(float_close(ag[1], 6.0f), "pow d(x^2)/dx=2x -> 6");
    SNEPPX_tape_destroy(tape);
}

static void test_neg_backward(void) {
    size_t shape[] = {3};
    float adata[] = {1.0f, -2.0f, 3.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, adata, 3 * sizeof(float));
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* n = SNEPPX_neg(tape, a);
    SNEPPX_tape_backward(tape, n);
    SX_ASSERT(a->grad != NULL, "neg grad");
    float* ag = (float*)a->grad->data;
    SX_ASSERT(float_close(ag[0], -1.0f), "neg grad = -1");
    SNEPPX_tape_destroy(tape);
}

static void test_gelu_backward(void) {
    size_t shape[] = {2};
    float adata[] = {0.0f, 1.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* g = SNEPPX_gelu(tape, a);
    SNEPPX_tape_backward(tape, g);
    SX_ASSERT(a->grad != NULL, "gelu grad");
    float* ag = (float*)a->grad->data;
    SX_ASSERT(float_close(ag[0], 0.5f), "gelu d/dx at 0 = 0.5");
    SNEPPX_tape_destroy(tape);
}

static void test_silu_backward(void) {
    size_t shape[] = {2};
    float adata[] = {0.0f, 1.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* s = SNEPPX_silu(tape, a);
    SNEPPX_tape_backward(tape, s);
    SX_ASSERT(a->grad != NULL, "silu grad");
    float* ag = (float*)a->grad->data;
    SX_ASSERT(float_close(ag[0], 0.5f), "silu grad at 0 = 0.5");
    SNEPPX_tape_destroy(tape);
}

static void test_log_backward(void) {
    size_t shape[] = {2};
    float adata[] = {1.0f, 2.0f};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* l = SNEPPX_log(tape, a);
    SNEPPX_tape_backward(tape, l);
    SX_ASSERT(a->grad != NULL, "log grad");
    float* ag = (float*)a->grad->data;
    SX_ASSERT(float_close(ag[0], 1.0f), "log grad at 1 = 1");
    SX_ASSERT(float_close(ag[1], 0.5f), "log grad at 2 = 0.5");
    SNEPPX_tape_destroy(tape);
}

static void test_mean_backward(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* m = SNEPPX_mean(tape, a, 0);
    SNEPPX_tape_backward(tape, m);
    SX_ASSERT(a->grad != NULL, "mean grad");
    float* ag = (float*)a->grad->data;
    SX_ASSERT(float_close(ag[0], 0.5f), "mean grad = 1/N");
    SNEPPX_tape_destroy(tape);
}

static void test_transpose_backward(void) {
    size_t shape[] = {2, 3};
    float adata[] = {1,2,3,4,5,6};
    SNEPPXTensor* ta = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    memcpy(ta->data, adata, 6 * sizeof(float));
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* t = SNEPPX_transpose(tape, a, 0, 1);
    SNEPPX_tape_backward(tape, t);
    SX_ASSERT(a->grad != NULL, "transpose grad");
    float* ag = (float*)a->grad->data;
    SX_ASSERT(float_close(ag[0], 1.0f), "transpose grad matches");
    SX_ASSERT(float_close(ag[3], 1.0f), "transpose grad matches");
    SNEPPX_tape_destroy(tape);
}

static void test_concat_backward(void) {
    size_t sh1[] = {2, 2}, sh2[] = {2, 3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh1, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(sh2, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* vars[] = {a, b};
    SNEPPXVariable* c = SNEPPX_concat(tape, vars, 2, 1);
    SNEPPX_tape_backward(tape, c);
    SX_ASSERT(a->grad != NULL, "concat a grad");
    SX_ASSERT(b->grad != NULL, "concat b grad");
    float* ag = (float*)a->grad->data;
    for (size_t i = 0; i < 4; i++) SX_ASSERT(float_close(ag[i], 1.0f), "concat grad ones");
    SNEPPX_tape_destroy(tape);
}


TEST(test_autodiff, variable_create) { test_variable_create(); }
TEST(test_autodiff, add_forward) { test_add_forward(); }
TEST(test_autodiff, add_backward) { test_add_backward(); }
TEST(test_autodiff, mul_backward) { test_mul_backward(); }
TEST(test_autodiff, matmul_backward) { test_matmul_backward(); }
TEST(test_autodiff, mse_loss_backward) { test_mse_loss_backward(); }
TEST(test_autodiff, relu_backward) { test_relu_backward(); }
TEST(test_autodiff, sigmoid_backward) { test_sigmoid_backward(); }
TEST(test_autodiff, tanh_backward) { test_tanh_backward(); }
TEST(test_autodiff, softmax_backward) { test_softmax_backward(); }
TEST(test_autodiff, no_grad) { test_no_grad(); }
TEST(test_autodiff, exp_backward) { test_exp_backward(); }
TEST(test_autodiff, sum_backward) { test_sum_backward(); }
TEST(test_autodiff, chain_rule) { test_chain_rule(); }
TEST(test_autodiff, dropout_backward) { test_dropout_backward(); }
TEST(test_autodiff, layer_norm_backward) { test_layer_norm_backward(); }
TEST(test_autodiff, sub_backward) { test_sub_backward(); }
TEST(test_autodiff, div_backward) { test_div_backward(); }
TEST(test_autodiff, pow_backward) { test_pow_backward(); }
TEST(test_autodiff, neg_backward) { test_neg_backward(); }
TEST(test_autodiff, gelu_backward) { test_gelu_backward(); }
TEST(test_autodiff, silu_backward) { test_silu_backward(); }
TEST(test_autodiff, log_backward) { test_log_backward(); }
TEST(test_autodiff, mean_backward) { test_mean_backward(); }
TEST(test_autodiff, transpose_backward) { test_transpose_backward(); }
TEST(test_autodiff, concat_backward) { test_concat_backward(); }
