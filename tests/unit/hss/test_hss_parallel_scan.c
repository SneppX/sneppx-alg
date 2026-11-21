#include "hierarchical_state_space.h"
#include "test_gtest.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/*
 * SNEPPX - Test Hss Parallel Scan
 *
 * WHAT
 *   Test Hss Parallel Scan.
 *
 * CONCEPT
 *   Provides tensor/expert parallelism.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






static void test_parallel_vs_sequential(void) {
    SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
    cfg.state_dim = 8;
    cfg.input_dim = 16;
    cfg.output_dim = 16;
    cfg.num_layers = 1;
    cfg.seq_len = 32;
    cfg.use_hierarchical = 0;
    cfg.use_parallel_scan = 1;

    SNEPPXHSSModel* model = SNEPPX_hss_model_create(&cfg, 42);
    SX_ASSERT(model != NULL, "model created");

    size_t shape_in[] = {1, cfg.seq_len, cfg.input_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 3, SNEPPX_FLOAT32);
    SX_ASSERT(input != NULL, "input tensor created");

    float* data = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) data[i] = ((float)(i % 7) - 3.0f) * 0.5f;

    SNEPPXTensor* output = NULL;
    int ret = SNEPPX_hss_forward(model, input, &output);
    SX_ASSERT(ret == 0, "forward pass with parallel scan succeeded");
    SX_ASSERT(output != NULL, "output tensor created");
    SX_ASSERT(output->size == cfg.seq_len * cfg.output_dim, "output size correct");

    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(input);
    SNEPPX_hss_model_destroy(model);
}

static void test_parallel_scan_disabled_fallback(void) {
    SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
    cfg.state_dim = 8;
    cfg.input_dim = 16;
    cfg.output_dim = 16;
    cfg.num_layers = 1;
    cfg.seq_len = 32;
    cfg.use_hierarchical = 0;
    cfg.use_parallel_scan = 0;

    SNEPPXHSSModel* model = SNEPPX_hss_model_create(&cfg, 42);
    SX_ASSERT(model != NULL, "model created");

    size_t shape_in[] = {1, cfg.seq_len, cfg.input_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 3, SNEPPX_FLOAT32);
    SX_ASSERT(input != NULL, "input tensor created");

    float* data = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) data[i] = ((float)(i % 7) - 3.0f) * 0.5f;

    SNEPPXTensor* output = NULL;
    int ret = SNEPPX_hss_forward(model, input, &output);
    SX_ASSERT(ret == 0, "forward pass with sequential scan succeeded");
    SX_ASSERT(output != NULL, "output tensor created");

    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(input);
    SNEPPX_hss_model_destroy(model);
}

static void test_parallel_scan_matches_sequential(void) {
    SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    cfg.num_layers = 2;
    cfg.seq_len = 16;
    cfg.use_hierarchical = 0;

    SNEPPXHSSModel* model_seq = SNEPPX_hss_model_create(&cfg, 100);
    SX_ASSERT(model_seq != NULL, "seq model created");
    SNEPPXHSSModel* model_par = SNEPPX_hss_model_create(&cfg, 100);
    SX_ASSERT(model_par != NULL, "par model created");

    model_par->config.use_parallel_scan = 1;

    size_t shape_in[] = {1, cfg.seq_len, cfg.input_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 3, SNEPPX_FLOAT32);
    SX_ASSERT(input != NULL, "input tensor created");

    float* data = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) data[i] = sinf((float)i * 0.1f);

    SNEPPXTensor* out_seq = NULL;
    SNEPPX_hss_forward(model_seq, input, &out_seq);
    SX_ASSERT(out_seq != NULL, "seq output");

    SNEPPXTensor* out_par = NULL;
    SNEPPX_hss_forward(model_par, input, &out_par);
    SX_ASSERT(out_par != NULL, "par output");

    float* seq_data = (float*)out_seq->data;
    float* par_data = (float*)out_par->data;
    for (size_t i = 0; i < out_seq->size; i++) {
        SX_ASSERT_NEAR(seq_data[i], par_data[i], 1e-3f, "parallel scan matches sequential");
    }

    SNEPPX_tensor_destroy(out_seq);
    SNEPPX_tensor_destroy(out_par);
    SNEPPX_tensor_destroy(input);
    SNEPPX_hss_model_destroy(model_seq);
    SNEPPX_hss_model_destroy(model_par);
}


TEST(test_hss_parallel_scan, parallel_vs_sequential) { test_parallel_vs_sequential(); }
TEST(test_hss_parallel_scan, parallel_scan_disabled_fallback) { test_parallel_scan_disabled_fallback(); }
TEST(test_hss_parallel_scan, parallel_scan_matches_sequential) { test_parallel_scan_matches_sequential(); }
