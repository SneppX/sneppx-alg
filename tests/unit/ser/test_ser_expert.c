#include "sparse_expert_routing.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Ser Expert
 *
 * WHAT
 *   Test Ser Expert.
 *
 * CONCEPT
 *   Provides the Test Ser Expert.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_expert_create(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.input_dim = 64; cfg.expert_dim = 128; cfg.output_dim = 64;
    SNEPPXExpert* e = SNEPPX_expert_create(&cfg, 42, SNEPPX_ACT_RELU);
    SX_ASSERT(e != NULL, "expert not null");
    SX_ASSERT(e->w1->shape[0] == 128 && e->w1->shape[1] == 64, "w1 shape");
    SX_ASSERT(e->w2->shape[0] == 64 && e->w2->shape[1] == 128, "w2 shape");
    SX_ASSERT(e->b1->shape[0] == 128, "b1 shape");
    SX_ASSERT(e->b2->shape[0] == 64, "b2 shape");
    SNEPPX_expert_destroy(e);
}

static void test_expert_forward(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.input_dim = 16; cfg.expert_dim = 32; cfg.output_dim = 16;
    SNEPPXExpert* e = SNEPPX_expert_create(&cfg, 42, SNEPPX_ACT_RELU);
    SX_ASSERT(e != NULL, "expert not null");

    size_t shape_in[] = {4, 16};
    SNEPPXTensor* input = SNEPPX_tensor_randn(shape_in, 2, SNEPPX_FLOAT32);
    SX_ASSERT(input != NULL, "input not null");

    size_t shape_out[] = {4, 16};
    SNEPPXTensor* output = SNEPPX_tensor_zeros(shape_out, 2, SNEPPX_FLOAT32);
    SX_ASSERT(output != NULL, "output not null");

    SNEPPX_ser_expert_forward(e, input, output);
    float* od = (float*)output->data;
    int ok = 1;
    for (size_t i = 0; i < 4 * 16; i++) {
        if (!isfinite(od[i])) { ok = 0; break; }
    }
    SX_ASSERT(ok, "all finite");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(output);
    SNEPPX_expert_destroy(e);
}


TEST(test_ser_expert, test_expert_create) { test_expert_create(); }
TEST(test_ser_expert, test_expert_forward) { test_expert_forward(); }
