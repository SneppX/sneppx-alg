#include "sparse_expert_routing.h"
#include "test_gtest.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Ser Mlp Gater
 *
 * WHAT
 *   Test Ser Mlp Gater.
 *
 * CONCEPT
 *   Provides the Test Ser Mlp Gater.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_mlp_gater_config_default(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    SX_ASSERT(cfg.use_mlp_gater == 0, "default is linear gater");
    SX_ASSERT(cfg.gater_hidden_dim == 64, "default gater hidden dim 64");
}

static void test_mlp_gater_route(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.use_mlp_gater = 1;
    cfg.gater_hidden_dim = 16;
    cfg.num_experts = 4;
    cfg.num_active = 2;
    cfg.input_dim = 8;
    cfg.expert_dim = 16;
    cfg.output_dim = 8;
    SNEPPXSERLayer* layer = SNEPPX_ser_layer_create(&cfg, 42);
    SX_ASSERT(layer != NULL, "layer with mlp gater created");
    SX_ASSERT(layer->gater_w1 != NULL, "gater_w1 created");
    SX_ASSERT(layer->gater_b1 != NULL, "gater_b1 created");
    SX_ASSERT(layer->gater_w2 != NULL, "gater_w2 created");
    SX_ASSERT(layer->gater_b2 != NULL, "gater_b2 created");

    size_t shape_in[] = {4, cfg.input_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) d[i] = ((float)(i % 5) - 2.0f) * 0.1f;

    SNEPPXTensor* gate_weights = NULL;
    int* expert_indices = NULL;
    SNEPPX_ser_route_mlp_gated(layer, input, &gate_weights, &expert_indices);
    SX_ASSERT(gate_weights != NULL, "gate weights computed");
    SX_ASSERT(expert_indices != NULL, "expert indices computed");
    SX_ASSERT(gate_weights->shape[0] == 4, "batch dim correct");
    SX_ASSERT(gate_weights->shape[1] == 2, "active dim correct");
    SX_ASSERT(expert_indices[0] >= 0 && expert_indices[0] < 4, "valid expert index");

    float* gw = (float*)gate_weights->data;
    float row_sum = gw[0] + gw[1];
    SX_ASSERT_NEAR(row_sum, 1.0f, 1e-5f, "gate weights sum to 1");

    SNEPPX_tensor_destroy(gate_weights);
    SNEPPX_free(expert_indices, 4 * 2 * sizeof(int));
    SNEPPX_tensor_destroy(input);
    SNEPPX_ser_layer_destroy(layer);
}

static void test_mlp_gater_linear_vs_mlp_interface(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.use_mlp_gater = 0;
    cfg.num_experts = 4;
    cfg.num_active = 2;
    cfg.input_dim = 8;
    cfg.expert_dim = 16;
    cfg.output_dim = 8;
    SNEPPXSERLayer* linear_layer = SNEPPX_ser_layer_create(&cfg, 42);
    SX_ASSERT(linear_layer->gater_w1 == NULL, "linear layer has no gater_w1");

    cfg.use_mlp_gater = 1;
    cfg.gater_hidden_dim = 16;
    SNEPPXSERLayer* mlp_layer = SNEPPX_ser_layer_create(&cfg, 42);
    SX_ASSERT(mlp_layer->gater_w1 != NULL, "mlp layer has gater_w1");

    size_t shape_in[] = {2, cfg.input_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) d[i] = 0.1f;

    SNEPPXTensor* gw1 = NULL; int* ei1 = NULL;
    SNEPPX_ser_route(linear_layer, input, &gw1, &ei1);
    SX_ASSERT(gw1 != NULL, "linear route works");

    SNEPPXTensor* gw2 = NULL; int* ei2 = NULL;
    SNEPPX_ser_route_mlp_gated(mlp_layer, input, &gw2, &ei2);
    SX_ASSERT(gw2 != NULL, "mlp gated route works");

    SNEPPX_tensor_destroy(gw1);
    SNEPPX_free(ei1, 2 * 2 * sizeof(int));
    SNEPPX_tensor_destroy(gw2);
    SNEPPX_free(ei2, 2 * 2 * sizeof(int));
    SNEPPX_tensor_destroy(input);
    SNEPPX_ser_layer_destroy(linear_layer);
    SNEPPX_ser_layer_destroy(mlp_layer);
}


TEST(test_ser_mlp_gater, MLP_gater_config_defaults) { test_mlp_gater_config_default(); }
TEST(test_ser_mlp_gater, MLP_gater_route_output) { test_mlp_gater_route(); }
TEST(test_ser_mlp_gater, MLP_gater_vs_linear_interface) { test_mlp_gater_linear_vs_mlp_interface(); }
