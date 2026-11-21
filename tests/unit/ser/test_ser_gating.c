#include "sparse_expert_routing.h"
#include "test_gtest.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Ser Gating
 *
 * WHAT
 *   Test Ser Gating.
 *
 * CONCEPT
 *   Provides the Test Ser Gating.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






static void test_ser_config_default(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    SX_ASSERT(cfg.num_experts == 8, "default has 8 experts");
    SX_ASSERT(cfg.num_active == 2, "default 2 active experts");
    SX_ASSERT(cfg.top_k_method == SNEPPX_TOPK_GREEDY, "default greedy top-k");
}

static void test_ser_load_balance_loss(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.num_experts = 4;
    cfg.num_active = 2;
    cfg.input_dim = 8;
    cfg.expert_dim = 16;
    cfg.output_dim = 8;
    SNEPPXSERLayer* layer = SNEPPX_ser_layer_create(&cfg, 42);
    SX_ASSERT(layer != NULL, "layer created");

    size_t shape_in[] = {4, cfg.input_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) d[i] = ((float)(i % 5) - 2.0f) * 0.1f;

    SNEPPXTensor* gate_weights = NULL;
    int* expert_indices = NULL;
    SNEPPX_ser_route(layer, input, &gate_weights, &expert_indices);
    SX_ASSERT(gate_weights != NULL, "gate weights computed");
    SX_ASSERT(expert_indices != NULL, "expert indices computed");

    float lb_loss = SNEPPX_ser_load_balance_loss(gate_weights, expert_indices, 4);
    SX_ASSERT(lb_loss >= 0.0f, "load balance loss >= 0");

    SNEPPX_tensor_destroy(gate_weights);
    SNEPPX_free(expert_indices, 4 * 2 * sizeof(int));
    SNEPPX_tensor_destroy(input);
    SNEPPX_ser_layer_destroy(layer);
}

static void test_ser_z_loss(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.num_experts = 4;
    cfg.num_active = 2;
    SNEPPXSERLayer* layer = SNEPPX_ser_layer_create(&cfg, 42);

    size_t shape_in[] = {2, cfg.input_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) d[i] = 0.5f;

    SNEPPXTensor* gate_weights = NULL;
    int* expert_indices = NULL;
    SNEPPX_ser_route(layer, input, &gate_weights, &expert_indices);

    float zl = SNEPPX_ser_z_loss(gate_weights);
    SX_ASSERT(zl >= 0.0f, "z-loss >= 0");

    SNEPPX_tensor_destroy(gate_weights);
    SNEPPX_free(expert_indices, 2 * 2 * sizeof(int));
    SNEPPX_tensor_destroy(input);
    SNEPPX_ser_layer_destroy(layer);
}


TEST(test_ser_gating, ser_config_default) { test_ser_config_default(); }
TEST(test_ser_gating, ser_load_balance_loss) { test_ser_load_balance_loss(); }
TEST(test_ser_gating, ser_z_loss) { test_ser_z_loss(); }
