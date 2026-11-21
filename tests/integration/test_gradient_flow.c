#include "automatic_differentiation_framework.h"
#include "test_gtest.h"
#include "gradient_optimization_suite.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Gradient Flow
 *
 * WHAT
 *   Test Gradient Flow.
 *
 * CONCEPT
 *   Provides the Test Gradient Flow.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






static void test_mul_gradient(void) {
    size_t sh[] = {2};
    float xd[] = {2.0f, 3.0f};
    float wd[] = {0.5f, 2.0f};
    SNEPPXTensor* tx = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tw = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(tx->data, xd, 8);
    memcpy(tw->data, wd, 8);
    SNEPPXVariable* x = SNEPPX_variable_create(tx, 1);
    SNEPPXVariable* w = SNEPPX_variable_create(tw, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* y = SNEPPX_mul(tape, x, w);
    SNEPPXVariable* loss = SNEPPX_sum(tape, y, 0);
    SNEPPX_tape_backward(tape, loss);
    float* xg = (float*)x->grad->data;
    float* wg = (float*)w->grad->data;
    SX_ASSERT_NEAR(xg[0], 0.5f, 1e-5f, "d(sum(x*w))/dx[0] = w[0]");
    SX_ASSERT_NEAR(xg[1], 2.0f, 1e-5f, "d(sum(x*w))/dx[1] = w[1]");
    SX_ASSERT_NEAR(wg[0], 2.0f, 1e-5f, "d(sum(x*w))/dw[0] = x[0]");
    SX_ASSERT_NEAR(wg[1], 3.0f, 1e-5f, "d(sum(x*w))/dw[1] = x[1]");
    SNEPPX_tape_destroy(tape);
}

static void test_relu_gradient(void) {
    size_t sh[] = {4};
    float xd[] = {-2.0f, -1.0f, 0.0f, 3.0f};
    SNEPPXTensor* tx = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(tx->data, xd, 16);
    SNEPPXVariable* x = SNEPPX_variable_create(tx, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* y = SNEPPX_relu(tape, x);
    SNEPPXVariable* loss = SNEPPX_sum(tape, y, 0);
    SNEPPX_tape_backward(tape, loss);
    float* xg = (float*)x->grad->data;
    SX_ASSERT_NEAR(xg[0], 0.0f, 1e-5f, "relu'(-2) = 0");
    SX_ASSERT_NEAR(xg[1], 0.0f, 1e-5f, "relu'(-1) = 0");
    SX_ASSERT_NEAR(xg[2], 0.0f, 1e-5f, "relu'(0) = 0");
    SX_ASSERT_NEAR(xg[3], 1.0f, 1e-5f, "relu'(3) = 1");
    SNEPPX_tape_destroy(tape);
}

static void test_mse_gradient(void) {
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
    SX_ASSERT_NEAR(pg[0], 2.0f/3.0f, 1e-5f, "mse grad[0] = 2*(1-0)/3");
    SX_ASSERT_NEAR(pg[1], 4.0f/3.0f, 1e-5f, "mse grad[1] = 2*(2-0)/3");
    SX_ASSERT_NEAR(pg[2], 2.0f, 1e-5f, "mse grad[2] = 2*(3-0)/3");
    SNEPPX_tape_destroy(tape);
}

static void test_chain_gradient(void) {
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
    SX_ASSERT_NEAR(((float*)a->grad->data)[0], 4.0f, 1e-5f, "chain a[0] = 4");
    SX_ASSERT_NEAR(((float*)a->grad->data)[1], 6.0f, 1e-5f, "chain a[1] = 6");
    SNEPPX_tape_destroy(tape);
}

static void test_optimizer_step(void) {
    size_t sh[] = {3};
    float wd[] = {1.0f, 1.0f, 1.0f};
    SNEPPXTensor* tw = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    memcpy(tw->data, wd, 12);
    SNEPPXVariable* w = SNEPPX_variable_create(tw, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXTensor* tx = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    ((float*)tx->data)[0] = 1.0f; ((float*)tx->data)[1] = 0.5f; ((float*)tx->data)[2] = 0.25f;
    SNEPPXVariable* x = SNEPPX_variable_create(tx, 0);
    SNEPPXVariable* y = SNEPPX_mul(tape, w, x);
    SNEPPXVariable* loss = SNEPPX_sum(tape, y, 0);
    SNEPPX_tape_backward(tape, loss);
    SX_ASSERT(w->grad != NULL, "w has gradient");
    float old_w0 = ((float*)w->data->data)[0];
    SNEPPXOptimizerConfig opt_cfg = SNEPPX_optimizer_config_default();
    opt_cfg.learning_rate = 0.1f;
    SNEPPXOptimizer* opt = SNEPPX_optimizer_create(&opt_cfg);
    SX_ASSERT(opt != NULL, "optimizer created");
    SNEPPXTensor* params[] = {w->data};
    SNEPPXTensor* grads[] = {w->grad};
    SNEPPX_optimizer_step(opt, params, grads, 1);
    float new_w0 = ((float*)w->data->data)[0];
    SX_ASSERT(new_w0 != old_w0, "weight changed after step");
    SNEPPX_optimizer_destroy(opt);
    SNEPPX_tape_destroy(tape);
}


TEST(test_gradient_flow, mul_gradient) { test_mul_gradient(); }
TEST(test_gradient_flow, relu_gradient) { test_relu_gradient(); }
TEST(test_gradient_flow, mse_gradient) { test_mse_gradient(); }
TEST(test_gradient_flow, chain_gradient) { test_chain_gradient(); }
TEST(test_gradient_flow, optimizer_step) { test_optimizer_step(); }
