#include "sparse_expert_routing.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Ser Model
 *
 * WHAT
 *   Test Ser Model.
 *
 * CONCEPT
 *   Provides the Test Ser Model.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_model_create(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.num_experts = 4; cfg.num_active = 2; cfg.input_dim = 16;
    cfg.expert_dim = 32; cfg.output_dim = 16;
    SNEPPXSERModel* model = SNEPPX_ser_model_create(&cfg, 42, 2);
    SX_ASSERT(model != NULL, "model not null");
    SX_ASSERT(model->num_layers == 2, "num_layers == 2");
    SX_ASSERT(model->layers[0] != NULL, "layer 0 not null");
    SX_ASSERT(model->layers[1] != NULL, "layer 1 not null");
    SX_ASSERT(model->layers[0]->experts[0] != NULL, "expert 0 not null");
    SNEPPX_ser_model_destroy(model);
}

static void test_model_forward(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.num_experts = 4; cfg.num_active = 2; cfg.input_dim = 16;
    cfg.expert_dim = 32; cfg.output_dim = 16;
    SNEPPXSERModel* model = SNEPPX_ser_model_create(&cfg, 42, 1);
    SX_ASSERT(model != NULL, "model not null");

    size_t shape_in[] = {4, 32, 16};
    SNEPPXTensor* input = SNEPPX_tensor_randn(shape_in, 3, SNEPPX_FLOAT32);
    SX_ASSERT(input != NULL, "input not null");

    SNEPPXTensor* output = NULL;
    size_t num_tokens = 4 * 32;
    SNEPPXTensor flat_input;
    size_t flat_shape[] = {num_tokens, 16};
    flat_input.data = input->data;
    flat_input.shape = flat_shape;
    flat_input.ndim = 2;
    flat_input.size = num_tokens * 16;
    flat_input.item_size = sizeof(float);
    flat_input.dtype = SNEPPX_FLOAT32;
    flat_input.strides = NULL;

    SNEPPX_ser_forward(model->layers[0], &flat_input, &output);
    SX_ASSERT(output != NULL, "output not null");
    SX_ASSERT(output->shape[0] == num_tokens, "output tokens");
    SX_ASSERT(output->shape[1] == 16, "output dim == 16");

    float* od = (float*)output->data;
    int ok = 1;
    for (size_t i = 0; i < output->size; i++) {
        if (!isfinite(od[i])) { ok = 0; break; }
    }
    SX_ASSERT(ok, "all finite");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(output);
    SNEPPX_ser_model_destroy(model);
}


TEST(test_ser_model, test_model_create) { test_model_create(); }
TEST(test_ser_model, test_model_forward) { test_model_forward(); }
