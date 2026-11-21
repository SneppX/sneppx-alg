#include "hierarchical_state_space.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Test Hss Model
 *
 * WHAT
 *   Test Hss Model.
 *
 * CONCEPT
 *   Provides the Test Hss Model.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_model_create(void) {
    SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    cfg.num_layers = 2;
    SNEPPXHSSModel* model = SNEPPX_hss_model_create(&cfg, 42);
    SX_ASSERT(model != NULL, "model not null");
    SX_ASSERT(model->layers != NULL, "layers array not null");
    SX_ASSERT(model->layers[0] != NULL, "layer 0 not null");
    SX_ASSERT(model->layers[1] != NULL, "layer 1 not null");
    SX_ASSERT(model->config.num_layers == 2, "num_layers == 2");
    SX_ASSERT(model->norm_gamma[0] != NULL, "norm_gamma[0] not null");
    SX_ASSERT(model->norm_beta[0] != NULL, "norm_beta[0] not null");
    SNEPPX_hss_model_destroy(model);
}

static void test_forward_small(void) {
    SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    cfg.num_layers = 1;
    cfg.use_hierarchical = 0;
    SNEPPXHSSModel* model = SNEPPX_hss_model_create(&cfg, 42);
    SX_ASSERT(model != NULL, "model not null");

    size_t shape_in[] = {4, 8, 8};
    SNEPPXTensor* input = SNEPPX_tensor_randn(shape_in, 3, SNEPPX_FLOAT32);
    SX_ASSERT(input != NULL, "input not null");

    SNEPPXTensor* output = NULL;
    int ret = SNEPPX_hss_forward(model, input, &output);
    SX_ASSERT(ret == 0, "forward returned 0");
    SX_ASSERT(output != NULL, "output not null");
    SX_ASSERT(output->shape[0] == 4, "output batch == 4");
    SX_ASSERT(output->shape[1] == 8, "output seq_len == 8");
    SX_ASSERT(output->shape[2] == 8, "output dim == 8");

    float* od = (float*)output->data;
    int has_nan = 0;
    for (size_t i = 0; i < output->size; i++) {
        if (!isfinite(od[i])) { has_nan = 1; break; }
    }
    SX_ASSERT(!has_nan, "no NaN/inf in output");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(output);
    SNEPPX_hss_model_destroy(model);
}

static void test_forward_single_seq(void) {
    SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    cfg.num_layers = 1;
    cfg.use_hierarchical = 0;
    SNEPPXHSSModel* model = SNEPPX_hss_model_create(&cfg, 42);
    SX_ASSERT(model != NULL, "model not null");

    size_t shape_in[] = {16, 8};
    SNEPPXTensor* input = SNEPPX_tensor_randn(shape_in, 2, SNEPPX_FLOAT32);
    SX_ASSERT(input != NULL, "input not null");

    SNEPPXTensor* output = NULL;
    int ret = SNEPPX_hss_forward(model, input, &output);
    SX_ASSERT(ret == 0, "forward returned 0");
    SX_ASSERT(output != NULL, "output not null");

    float* od = (float*)output->data;
    int has_nan = 0;
    for (size_t i = 0; i < output->size; i++) {
        if (!isfinite(od[i])) { has_nan = 1; break; }
    }
    SX_ASSERT(!has_nan, "no NaN/inf in output");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(output);
    SNEPPX_hss_model_destroy(model);
}


TEST(test_hss_model, test_model_create) { test_model_create(); }
TEST(test_hss_model, test_forward_small) { test_forward_small(); }
TEST(test_hss_model, test_forward_single_seq) { test_forward_single_seq(); }
