#include "hierarchical_state_space.h"
#include "test_gtest.h"
#include "sparse_expert_routing.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Hss Ser Stack
 *
 * WHAT
 *   Test Hss Ser Stack.
 *
 * CONCEPT
 *   Provides the Test Hss Ser Stack.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */




static void test_hss_ser_stack(void) {
    SNEPPXHSSConfig hss_cfg = SNEPPX_hss_config_default();
    hss_cfg.state_dim = 16; hss_cfg.input_dim = 32; hss_cfg.output_dim = 32;
    hss_cfg.num_layers = 1; hss_cfg.use_hierarchical = 0;

    SNEPPXHSSModel* hss = SNEPPX_hss_model_create(&hss_cfg, 42);
    SX_ASSERT(hss != NULL, "hss model not null");

    SNEPPXSERConfig ser_cfg = SNEPPX_ser_config_default();
    ser_cfg.num_experts = 4; ser_cfg.num_active = 2; ser_cfg.input_dim = 32;
    ser_cfg.expert_dim = 64; ser_cfg.output_dim = 32;

    SNEPPXSERModel* ser = SNEPPX_ser_model_create(&ser_cfg, 99, 1);
    SX_ASSERT(ser != NULL, "ser model not null");

    size_t shape_in[] = {4, 16, 32};
    SNEPPXTensor* input = SNEPPX_tensor_randn(shape_in, 3, SNEPPX_FLOAT32);
    SX_ASSERT(input != NULL, "input not null");

    SNEPPXTensor* hss_out = NULL;
    int ret = SNEPPX_hss_forward(hss, input, &hss_out);
    SX_ASSERT(ret == 0, "hss forward ok");
    SX_ASSERT(hss_out != NULL, "hss output not null");
    SX_ASSERT(hss_out->shape[0] == 4 && hss_out->shape[1] == 16 && hss_out->shape[2] == 32, "hss output shape");

    size_t merged = 4 * 16;
    SNEPPXTensor flat_input;
    size_t flat_shape[] = {merged, 32};
    flat_input.data = hss_out->data;
    flat_input.shape = flat_shape;
    flat_input.ndim = 2;
    flat_input.size = merged * 32;
    flat_input.item_size = sizeof(float);
    flat_input.dtype = SNEPPX_FLOAT32;
    flat_input.strides = NULL;

    SNEPPXTensor* ser_out = NULL;
    SNEPPX_ser_forward(ser->layers[0], &flat_input, &ser_out);
    SX_ASSERT(ser_out != NULL, "ser output not null");
    SX_ASSERT(ser_out->shape[0] == merged, "ser output tokens");
    SX_ASSERT(ser_out->shape[1] == 32, "ser output dim");

    float* od = (float*)ser_out->data;
    int ok = 1;
    for (size_t i = 0; i < ser_out->size; i++) {
        if (!isfinite(od[i])) { ok = 0; break; }
    }
    SX_ASSERT(ok, "all finite in ser output");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(hss_out);
    SNEPPX_tensor_destroy(ser_out);
    SNEPPX_hss_model_destroy(hss);
    SNEPPX_ser_model_destroy(ser);
}


TEST(test_hss_ser_stack, test_hss_ser_stack) { test_hss_ser_stack(); }
