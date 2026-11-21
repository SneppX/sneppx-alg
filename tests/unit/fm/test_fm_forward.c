#include "fractal_memory_orchestrator.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Fm Forward
 *
 * WHAT
 *   Test Fm Forward.
 *
 * CONCEPT
 *   Provides the Test Fm Forward.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






static void test_fm_forward_simple(void) {
    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    cfg.memory_dim = 8;
    cfg.memory_capacity = 16;
    SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);
    SX_ASSERT(ctrl != NULL, "controller created");

    size_t shape[] = {2, cfg.memory_dim};
    SNEPPXTensor* input = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* output = NULL;
    int ret = SNEPPX_fm_forward(ctrl, 0, input, &output);
    SX_ASSERT(ret == 0, "forward returned 0");
    SX_ASSERT(output != NULL, "forward output created");
    SX_ASSERT(output->shape[0] == input->shape[0], "forward batch dim matches");
    SX_ASSERT(output->shape[1] == input->shape[1], "forward feat dim matches");

    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(input);
    SNEPPX_fm_controller_destroy(ctrl);
}

static void test_fm_forward_memory_write(void) {
    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    cfg.memory_dim = 4;
    cfg.memory_capacity = 4;
    SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);

    size_t shape[] = {1, cfg.memory_dim};
    SNEPPXTensor* input = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* output1 = NULL;
    int ret = SNEPPX_fm_forward(ctrl, 0, input, &output1);
    SX_ASSERT(ret == 0, "first forward returned 0");
    SX_ASSERT(output1 != NULL, "first forward");

    SNEPPXTensor* input2 = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* output2 = NULL;
    ret = SNEPPX_fm_forward(ctrl, 0, input2, &output2);
    SX_ASSERT(ret == 0, "second forward returned 0");
    SX_ASSERT(output2 != NULL, "second forward");

    SNEPPX_tensor_destroy(output2);
    SNEPPX_tensor_destroy(input2);
    SNEPPX_tensor_destroy(output1);
    SNEPPX_tensor_destroy(input);
    SNEPPX_fm_controller_destroy(ctrl);
}

static void test_fm_forward_batch(void) {
    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    cfg.memory_dim = 4;
    cfg.memory_capacity = 8;
    SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);

    size_t shape[] = {3, cfg.memory_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < 12; i++) d[i] = (float)(i % 4) * 0.25f;

    SNEPPXTensor* output = NULL;
    int ret = SNEPPX_fm_forward(ctrl, 0, input, &output);
    SX_ASSERT(ret == 0, "batch forward returned 0");
    SX_ASSERT(output != NULL, "batch forward created");
    SX_ASSERT(output->shape[0] == 3, "batch dim preserved");

    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(input);
    SNEPPX_fm_controller_destroy(ctrl);
}


TEST(test_fm_forward, fm_forward_simple) { test_fm_forward_simple(); }
TEST(test_fm_forward, fm_forward_memory_write) { test_fm_forward_memory_write(); }
TEST(test_fm_forward, fm_forward_batch) { test_fm_forward_batch(); }
