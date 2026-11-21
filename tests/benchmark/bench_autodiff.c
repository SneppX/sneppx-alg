#include "bench_common.h"
#include "automatic_differentiation_framework.h"

/*
 * SNEPPX - Bench Autodiff
 *
 * WHAT
 *   Bench Autodiff.
 *
 * CONCEPT
 *   Provides the Bench Autodiff.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void bench_variable_ops(void) {
    BENCH_INIT(bs);
    size_t sh[] = {256, 256};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    printf("  Variable ops [256,256]:\n");

    BENCH_START(bs, 10000, 500, {
        SNEPPXVariable* v = SNEPPX_variable_create(t, 1);
        v->data = NULL;
        SNEPPX_variable_destroy(v);
    });
    bench_print("variable_create/destroy", &bs);

    SNEPPX_tensor_destroy(t);
}

static void bench_add_backward(void) {
    BENCH_INIT(bs);
    size_t sh[] = {256, 256};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    printf("  Add backward [256,256]:\n");

    BENCH_START(bs, 500, 50, {
        SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
        SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
        SNEPPXTape* tape = SNEPPX_tape_create();
        SNEPPXVariable* c = SNEPPX_add(tape, a, b);
        SNEPPX_tape_backward(tape, c);
        a->data = NULL; a->grad = NULL;
        b->data = NULL; b->grad = NULL;
        c->data = NULL; c->grad = NULL;
        SNEPPX_tape_destroy(tape);
    });
    bench_print("add + backward", &bs);

    SNEPPX_tensor_destroy(ta); SNEPPX_tensor_destroy(tb);
}

static void bench_mul_backward(void) {
    BENCH_INIT(bs);
    size_t sh[] = {256, 256};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    printf("  Mul backward [256,256]:\n");

    BENCH_START(bs, 500, 50, {
        SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
        SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
        SNEPPXTape* tape = SNEPPX_tape_create();
        SNEPPXVariable* c = SNEPPX_mul(tape, a, b);
        SNEPPX_tape_backward(tape, c);
        a->data = NULL; a->grad = NULL;
        b->data = NULL; b->grad = NULL;
        c->data = NULL; c->grad = NULL;
        SNEPPX_tape_destroy(tape);
    });
    bench_print("mul + backward", &bs);

    SNEPPX_tensor_destroy(ta); SNEPPX_tensor_destroy(tb);
}

static void bench_matmul_backward(void) {
    BENCH_INIT(bs);
    size_t sh1[] = {64, 128};
    size_t sh2[] = {128, 64};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh1, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(sh2, 2, SNEPPX_FLOAT32);
    printf("  Matmul backward [64,128]x[128,64]:\n");

    BENCH_START(bs, 200, 20, {
        SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
        SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
        SNEPPXTape* tape = SNEPPX_tape_create();
        SNEPPXVariable* c = SNEPPX_matmul(tape, a, b);
        SNEPPX_tape_backward(tape, c);
        a->data = NULL; a->grad = NULL;
        b->data = NULL; b->grad = NULL;
        c->data = NULL; c->grad = NULL;
        SNEPPX_tape_destroy(tape);
    });
    bench_print("matmul + backward", &bs);

    SNEPPX_tensor_destroy(ta); SNEPPX_tensor_destroy(tb);
}

static void bench_chain_rule(void) {
    BENCH_INIT(bs);
    size_t sh[] = {64, 64};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    printf("  Chain rule [64,64]:\n");

    BENCH_START(bs, 200, 20, {
        SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
        SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
        SNEPPXTape* tape = SNEPPX_tape_create();
        SNEPPXVariable* c = SNEPPX_add(tape, a, b);
        SNEPPXVariable* d = SNEPPX_mul(tape, c, a);
        SNEPPXVariable* e = SNEPPX_relu(tape, d);
        SNEPPX_tape_backward(tape, e);
        a->data = NULL; a->grad = NULL;
        b->data = NULL; b->grad = NULL;
        c->data = NULL; c->grad = NULL;
        d->data = NULL; d->grad = NULL;
        e->data = NULL; e->grad = NULL;
        SNEPPX_tape_destroy(tape);
    });
    bench_print("add->mul->relu + backward", &bs);

    SNEPPX_tensor_destroy(ta); SNEPPX_tensor_destroy(tb);
}

static void bench_relu_sigmoid_backward(void) {
    BENCH_INIT(bs);
    size_t sh[] = {256, 256};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    printf("  Activation backward [256,256]:\n");

    BENCH_START(bs, 500, 50, {
        SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
        SNEPPXTape* tape = SNEPPX_tape_create();
        SNEPPXVariable* c = SNEPPX_relu(tape, a);
        SNEPPX_tape_backward(tape, c);
        a->data = NULL; a->grad = NULL;
        c->data = NULL; c->grad = NULL;
        SNEPPX_tape_destroy(tape);
    });
    bench_print("relu + backward", &bs);

    BENCH_START(bs, 500, 50, {
        SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
        SNEPPXTape* tape = SNEPPX_tape_create();
        SNEPPXVariable* c = SNEPPX_sigmoid(tape, a);
        SNEPPX_tape_backward(tape, c);
        a->data = NULL; a->grad = NULL;
        c->data = NULL; c->grad = NULL;
        SNEPPX_tape_destroy(tape);
    });
    bench_print("sigmoid + backward", &bs);

    SNEPPX_tensor_destroy(ta);
}

static void bench_mse_loss_backward(void) {
    BENCH_INIT(bs);
    size_t sh[] = {64, 64};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(sh, 2, SNEPPX_FLOAT32);
    printf("  MSE loss backward [64,64]:\n");

    BENCH_START(bs, 500, 50, {
        SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
        SNEPPXVariable* b = SNEPPX_variable_create(tb, 0);
        SNEPPXTape* tape = SNEPPX_tape_create();
        SNEPPXVariable* c = SNEPPX_mse_loss(tape, a, b);
        SNEPPX_tape_backward(tape, c);
        a->data = NULL; a->grad = NULL;
        c->data = NULL; c->grad = NULL;
        SNEPPX_tape_destroy(tape);
    });
    bench_print("mse_loss + backward", &bs);

    SNEPPX_tensor_destroy(ta); SNEPPX_tensor_destroy(tb);
}

int main(void) {
    printf("=== Autodiff Benchmarks ===\n");
    BENCH_RUN("Variable ops", bench_variable_ops);
    BENCH_RUN("Add backward", bench_add_backward);
    BENCH_RUN("Mul backward", bench_mul_backward);
    BENCH_RUN("Matmul backward", bench_matmul_backward);
    BENCH_RUN("Chain rule", bench_chain_rule);
    BENCH_RUN("Activation backward", bench_relu_sigmoid_backward);
    BENCH_RUN("MSE loss backward", bench_mse_loss_backward);
    BENCH_MAIN();
}
