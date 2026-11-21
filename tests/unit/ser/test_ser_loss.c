#include "sparse_expert_routing.h"
#include "test_gtest.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Ser Loss
 *
 * WHAT
 *   Test Ser Loss.
 *
 * CONCEPT
 *   Provides the Test Ser Loss.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






static void test_ser_importance_loss(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.num_experts = 4;
    SNEPPXSERLayer* layer = SNEPPX_ser_layer_create(&cfg, 42);
    size_t shape[] = {2, cfg.input_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) d[i] = 1.0f;
    SNEPPXTensor* gate_weights = NULL;
    int* expert_indices = NULL;
    SNEPPX_ser_route(layer, input, &gate_weights, &expert_indices);
    float imp_loss = SNEPPX_ser_importance_loss(gate_weights, expert_indices, 4);
    SX_ASSERT(imp_loss >= 0.0f, "importance loss >= 0");
    SNEPPX_tensor_destroy(gate_weights);
    SNEPPX_free(expert_indices, 2 * 2 * sizeof(int));
    SNEPPX_tensor_destroy(input);
    SNEPPX_ser_layer_destroy(layer);
}

static void test_ser_aux_loss_balanced(void) {
    size_t shape_batch[] = {4, 2};
    float data[8] = {1.0f, 0.0f, 0.5f, 0.5f, 0.0f, 1.0f, 0.3f, 0.7f};
    SNEPPXTensor* gate_weights = SNEPPX_tensor_create(shape_batch, 2, SNEPPX_FLOAT32);
    for (size_t i = 0; i < 8; i++) ((float*)gate_weights->data)[i] = data[i];
    int expert_indices[] = {0, 1, 0, 0};
    float aux = SNEPPX_ser_aux_load_balance(gate_weights, expert_indices, 4, 2);
    SX_ASSERT(aux >= 0.0f, "aux loss >= 0");
    SNEPPX_tensor_destroy(gate_weights);
}


TEST(test_ser_loss, ser_importance_loss) { test_ser_importance_loss(); }
TEST(test_ser_loss, ser_aux_loss_balanced) { test_ser_aux_loss_balanced(); }
