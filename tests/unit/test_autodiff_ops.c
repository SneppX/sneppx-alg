#include "automatic_differentiation_framework.h"
#include "test_gtest.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Autodiff Ops
 *
 * WHAT
 *   Test Autodiff Ops.
 *
 * CONCEPT
 *   Provides the Test Autodiff Ops.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






static void test_tape_create_destroy(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SX_ASSERT(tape != NULL, "tape created");
    SNEPPX_tape_destroy(tape);
}

static void test_variable_create(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT(t != NULL, "tensor created");
    SNEPPXVariable* v = SNEPPX_variable_create(t, 1);
    SX_ASSERT(v != NULL, "variable created");
    SX_ASSERT(v->requires_grad == 1, "requires_grad set");
    SNEPPX_variable_destroy(v);
}

static void test_tape_record_and_backward(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    size_t shape[] = {2, 2};
    SNEPPXTensor* a_t = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(a_t, 1);
    SNEPPX_tape_record(tape, a);

    SNEPPXTensor* b_t = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* b = SNEPPX_variable_create(b_t, 1);
    SNEPPX_tape_record(tape, b);

    SNEPPXVariable* c = SNEPPX_add(tape, a, b);
    SX_ASSERT(c != NULL, "add op created");

    SNEPPXVariable* d = SNEPPX_mul(tape, c, a);
    SX_ASSERT(d != NULL, "mul op created");

    SNEPPX_tape_backward(tape, d);
    SX_ASSERT(a->grad != NULL, "gradient computed for a");
    SX_ASSERT(b->grad != NULL, "gradient computed for b");

    /* tape owns all recorded variables (a,b,c,d); destroy via tape only */
    SNEPPX_tape_destroy(tape);
}

static void test_tape_ops(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    size_t shape[] = {3, 1};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* v = SNEPPX_variable_create(t, 1);
    SNEPPX_tape_record(tape, v);

    SNEPPXVariable* r1 = SNEPPX_relu(tape, v);
    SX_ASSERT(r1 != NULL, "relu op");
    SNEPPXVariable* r2 = SNEPPX_sigmoid(tape, v);
    SX_ASSERT(r2 != NULL, "sigmoid op");
    SNEPPXVariable* r3 = SNEPPX_exp(tape, v);
    SX_ASSERT(r3 != NULL, "exp op");
    SNEPPXVariable* r4 = SNEPPX_log(tape, v);
    SX_ASSERT(r4 != NULL, "log op");
    SNEPPXVariable* r5 = SNEPPX_neg(tape, v);
    SX_ASSERT(r5 != NULL, "neg op");

    SNEPPX_tape_backward(tape, r5);
    SX_ASSERT(v->grad != NULL, "grad computed after sequence");

    /* tape owns all recorded variables (v, r1..r5) */
    SNEPPX_tape_destroy(tape);
}


TEST(test_autodiff_ops, tape_create_destroy) { test_tape_create_destroy(); }
TEST(test_autodiff_ops, variable_create) { test_variable_create(); }
TEST(test_autodiff_ops, tape_record_and_backward) { test_tape_record_and_backward(); }
TEST(test_autodiff_ops, tape_ops) { test_tape_ops(); }
