#include "test_gtest.h"
#include "automatic_differentiation_framework.h"

/*
 * SNEPPX - Test Autodiff Edge
 *
 * WHAT
 *   Test Autodiff Edge.
 *
 * CONCEPT
 *   Provides the Test Autodiff Edge.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_variable_null_data(void) {
    SNEPPXVariable* v = SNEPPX_variable_create(NULL, 1);
    SX_ASSERT_NOT_NULL(v, "var with NULL data is allowed");
    SX_ASSERT_NULL(v->data, "data is NULL");
    SX_ASSERT_EQ(v->requires_grad, 1, "requires_grad set");
    v->data = NULL;
    SNEPPX_variable_destroy(v);
}

static void test_variable_no_grad(void) {
    size_t sh[] = {3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* v = SNEPPX_variable_create(t, 0);
    SX_ASSERT_NOT_NULL(v, "var requires_grad=0");
    SX_ASSERT_EQ(v->requires_grad, 0, "requires_grad == 0");
    SX_ASSERT_NULL(v->grad, "grad is NULL");
    SNEPPX_variable_destroy(v);
}

static void test_tape_destroy_null(void) {
    SNEPPX_tape_destroy(NULL);
}

static void test_variable_destroy_null(void) {
    SNEPPX_variable_destroy(NULL);
}

static void test_add_null_vars(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_add(tape, a, NULL);
    SX_ASSERT_NULL(c, "add with NULL b");
    c = SNEPPX_add(tape, NULL, a);
    SX_ASSERT_NULL(c, "add with NULL a");
    c = SNEPPX_add(tape, NULL, NULL);
    SX_ASSERT_NULL(c, "add NULL NULL");
    SNEPPX_tape_destroy(tape);
    a->data = NULL;
    SNEPPX_variable_destroy(a);
}

static void test_sub_null_vars(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_sub(tape, a, NULL);
    SX_ASSERT_NULL(c, "sub with NULL");
    SNEPPX_tape_destroy(tape);
    a->data = NULL;
    SNEPPX_variable_destroy(a);
}

static void test_mul_null_vars(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_mul(tape, a, NULL);
    SX_ASSERT_NULL(c, "mul with NULL");
    SNEPPX_tape_destroy(tape);
    a->data = NULL;
    SNEPPX_variable_destroy(a);
}

static void test_matmul_null_vars(void) {
    size_t sh[] = {2, 3};
    size_t sh2[] = {3, 4};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(sh2, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_matmul(tape, a, NULL);
    SX_ASSERT_NULL(c, "matmul NULL b");
    c = SNEPPX_matmul(tape, NULL, b);
    SX_ASSERT_NULL(c, "matmul NULL a");
    SNEPPX_tape_destroy(tape);
    a->data = NULL; b->data = NULL;
    SNEPPX_variable_destroy(a); SNEPPX_variable_destroy(b);
}

static void test_matmul_mismatched_dims(void) {
    size_t sh1[] = {2, 3};
    size_t sh2[] = {5, 4};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh1, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(sh2, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_matmul(tape, a, b);
    SX_ASSERT_NULL(c, "matmul mismatched inner");
    SNEPPX_tape_destroy(tape);
    a->data = NULL; b->data = NULL;
    SNEPPX_variable_destroy(a); SNEPPX_variable_destroy(b);
}

static void test_mse_loss_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_mse_loss(tape, a, NULL);
    SX_ASSERT_NULL(c, "mse_loss NULL target");
    c = SNEPPX_mse_loss(tape, NULL, a);
    SX_ASSERT_NULL(c, "mse_loss NULL pred");
    SNEPPX_tape_destroy(tape);
    a->data = NULL;
    SNEPPX_variable_destroy(a);
}

static void test_mse_loss_mismatched(void) {
    size_t sh1[] = {3};
    size_t sh2[] = {4};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh1, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(sh2, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_mse_loss(tape, a, b);
    SX_ASSERT_NULL(c, "mse_loss mismatched sizes");
    SNEPPX_tape_destroy(tape);
    a->data = NULL; b->data = NULL;
    SNEPPX_variable_destroy(a); SNEPPX_variable_destroy(b);
}

static void test_relu_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_relu(tape, NULL);
    SX_ASSERT_NULL(c, "relu NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_gelu_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_gelu(tape, NULL);
    SX_ASSERT_NULL(c, "gelu NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_sigmoid_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_sigmoid(tape, NULL);
    SX_ASSERT_NULL(c, "sigmoid NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_silu_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_silu(tape, NULL);
    SX_ASSERT_NULL(c, "silu NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_div_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_div(tape, a, NULL);
    SX_ASSERT_NULL(c, "div NULL b");
    c = SNEPPX_div(tape, NULL, a);
    SX_ASSERT_NULL(c, "div NULL a");
    SNEPPX_tape_destroy(tape);
    a->data = NULL;
    SNEPPX_variable_destroy(a);
}

static void test_pow_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_pow(tape, a, NULL);
    SX_ASSERT_NULL(c, "pow NULL b");
    c = SNEPPX_pow(tape, NULL, a);
    SX_ASSERT_NULL(c, "pow NULL a");
    SNEPPX_tape_destroy(tape);
    a->data = NULL;
    SNEPPX_variable_destroy(a);
}

static void test_neg_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_neg(tape, NULL);
    SX_ASSERT_NULL(c, "neg NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_tanh_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_tanh(tape, NULL);
    SX_ASSERT_NULL(c, "tanh NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_softmax_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_softmax(tape, NULL, 0);
    SX_ASSERT_NULL(c, "softmax NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_exp_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_exp(tape, NULL);
    SX_ASSERT_NULL(c, "exp NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_log_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_log(tape, NULL);
    SX_ASSERT_NULL(c, "log NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_sum_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_sum(tape, NULL, 0);
    SX_ASSERT_NULL(c, "sum NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_mean_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_mean(tape, NULL, 0);
    SX_ASSERT_NULL(c, "mean NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_transpose_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_transpose(tape, NULL, 0, 1);
    SX_ASSERT_NULL(c, "transpose NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_dropout_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_dropout(tape, NULL, 0.5f, 42);
    SX_ASSERT_NULL(c, "dropout NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_layer_norm_null(void) {
    size_t sh[] = {4};
    SNEPPXTensor* tg = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* g = SNEPPX_variable_create(tg, 0);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 0);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_layer_norm(tape, NULL, g, b, 1e-5f);
    SX_ASSERT_NULL(c, "layer_norm NULL input");
    SNEPPX_tape_destroy(tape);
    g->data = NULL; b->data = NULL;
    SNEPPX_variable_destroy(g); SNEPPX_variable_destroy(b);
}

static void test_concat_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_concat(tape, NULL, 0, 0);
    SX_ASSERT_NULL(c, "concat NULL array");
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* arr[] = {a, NULL};
    c = SNEPPX_concat(tape, arr, 2, 0);
    SX_ASSERT_NULL(c, "concat with NULL var");
    SNEPPX_tape_destroy(tape);
    a->data = NULL;
    SNEPPX_variable_destroy(a);
}

static void test_tape_backward_constant(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 0);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPX_tape_backward(tape, a);
    SX_ASSERT_NOT_NULL(a->grad, "backward sets grad (grad=ones)");
    if (a->grad) {
        float* gd = (float*)a->grad->data;
        SX_ASSERT_EQ(gd[0], 1.0f, "grad == 1");
    }
    SNEPPX_tape_destroy(tape);
    a->data = NULL; a->grad = NULL;
    SNEPPX_variable_destroy(a);
}

static void test_tape_record_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPX_tape_record(tape, NULL);
    SNEPPX_tape_destroy(tape);
}

static void test_no_grad_scoping(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPX_no_grad_enter();
    SNEPPXVariable* c = SNEPPX_add(tape, a, b);
    SX_ASSERT_NOT_NULL(c, "add in no_grad mode");
    SX_ASSERT(c->requires_grad == 0 || c->requires_grad == 0, "result requires_grad=0");
    SX_ASSERT_EQ(SNEPPX_no_grad_is_active(), 1, "no_grad active");
    SNEPPX_no_grad_exit();
    SX_ASSERT_EQ(SNEPPX_no_grad_is_active(), 0, "no_grad inactive after exit");
    SNEPPX_tape_destroy(tape);
    a->data = NULL; b->data = NULL;
    if (c) { c->data = NULL; c->grad = NULL; SNEPPX_variable_destroy(c); }
    SNEPPX_variable_destroy(a); SNEPPX_variable_destroy(b);
}


TEST(test_autodiff_edge, variable_null_data) { test_variable_null_data(); }
TEST(test_autodiff_edge, variable_no_grad) { test_variable_no_grad(); }
TEST(test_autodiff_edge, tape_destroy_null) { test_tape_destroy_null(); }
TEST(test_autodiff_edge, variable_destroy_null) { test_variable_destroy_null(); }
TEST(test_autodiff_edge, add_null_vars) { test_add_null_vars(); }
TEST(test_autodiff_edge, sub_null_vars) { test_sub_null_vars(); }
TEST(test_autodiff_edge, mul_null_vars) { test_mul_null_vars(); }
TEST(test_autodiff_edge, matmul_null_vars) { test_matmul_null_vars(); }
TEST(test_autodiff_edge, matmul_mismatched_dims) { test_matmul_mismatched_dims(); }
TEST(test_autodiff_edge, mse_loss_null) { test_mse_loss_null(); }
TEST(test_autodiff_edge, mse_loss_mismatched) { test_mse_loss_mismatched(); }
TEST(test_autodiff_edge, relu_null) { test_relu_null(); }
TEST(test_autodiff_edge, gelu_null) { test_gelu_null(); }
TEST(test_autodiff_edge, sigmoid_null) { test_sigmoid_null(); }
TEST(test_autodiff_edge, tanh_null) { test_tanh_null(); }
TEST(test_autodiff_edge, softmax_null) { test_softmax_null(); }
TEST(test_autodiff_edge, exp_null) { test_exp_null(); }
TEST(test_autodiff_edge, log_null) { test_log_null(); }
TEST(test_autodiff_edge, sum_null) { test_sum_null(); }
TEST(test_autodiff_edge, mean_null) { test_mean_null(); }
TEST(test_autodiff_edge, transpose_null) { test_transpose_null(); }
TEST(test_autodiff_edge, dropout_null) { test_dropout_null(); }
TEST(test_autodiff_edge, layer_norm_null) { test_layer_norm_null(); }
TEST(test_autodiff_edge, concat_null) { test_concat_null(); }
TEST(test_autodiff_edge, silu_null) { test_silu_null(); }
TEST(test_autodiff_edge, div_null) { test_div_null(); }
TEST(test_autodiff_edge, pow_null) { test_pow_null(); }
TEST(test_autodiff_edge, neg_null) { test_neg_null(); }
TEST(test_autodiff_edge, tape_backward_constant) { test_tape_backward_constant(); }
TEST(test_autodiff_edge, tape_record_null) { test_tape_record_null(); }
TEST(test_autodiff_edge, no_grad_scoping) { test_no_grad_scoping(); }
