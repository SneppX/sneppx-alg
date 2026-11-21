#include "sparse_expert_routing.h"
#include "test_gtest.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Ser Layer
 *
 * WHAT
 *   Test Ser Layer.
 *
 * CONCEPT
 *   Provides the Test Ser Layer.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_layer_forward(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.num_experts = 4; cfg.num_active = 2; cfg.input_dim = 16;
    cfg.expert_dim = 32; cfg.output_dim = 16;
    SNEPPXSERLayer* layer = SNEPPX_ser_layer_create(&cfg, 42);
    SX_ASSERT(layer != NULL, "layer not null");

    size_t shape_in[] = {16, 16};
    SNEPPXTensor* input = SNEPPX_tensor_randn(shape_in, 2, SNEPPX_FLOAT32);
    SX_ASSERT(input != NULL, "input not null");

    SNEPPXTensor* output = NULL;
    SNEPPX_ser_forward(layer, input, &output);
    SX_ASSERT(output != NULL, "output not null");
    SX_ASSERT(output->shape[0] == 16, "output tokens == 16");
    SX_ASSERT(output->shape[1] == 16, "output dim == 16");

    float* od = (float*)output->data;
    int ok = 1;
    for (size_t i = 0; i < 16 * 16; i++) {
        if (!isfinite(od[i])) { ok = 0; break; }
    }
    SX_ASSERT(ok, "all finite");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(output);
    SNEPPX_ser_layer_destroy(layer);
}

static void test_layer_load_balance(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.num_experts = 4; cfg.num_active = 2; cfg.input_dim = 16;
    cfg.expert_dim = 32; cfg.output_dim = 16;
    SNEPPXSERLayer* layer = SNEPPX_ser_layer_create(&cfg, 42);
    SX_ASSERT(layer != NULL, "layer not null");

    size_t shape_in[] = {16, 16};
    SNEPPXTensor* input = SNEPPX_tensor_randn(shape_in, 2, SNEPPX_FLOAT32);
    SX_ASSERT(input != NULL, "input not null");

    SNEPPXTensor* gw = NULL;
    int* ei = NULL;
    SNEPPX_ser_route(layer, input, &gw, &ei);
    SX_ASSERT(gw != NULL, "gw not null");

    float loss = SNEPPX_ser_load_balance_loss(gw, ei, 16);
    SX_ASSERT(isfinite(loss), "loss is finite");
    SX_ASSERT(loss >= 0.0f, "loss >= 0");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(gw);
    SNEPPX_free(ei, 16 * 2 * sizeof(int));
    SNEPPX_ser_layer_destroy(layer);
}


TEST(test_ser_layer, test_layer_forward) { test_layer_forward(); }
TEST(test_ser_layer, test_layer_load_balance) { test_layer_load_balance(); }
