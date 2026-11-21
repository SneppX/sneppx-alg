#include "hierarchical_state_space.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Test Hss Layer
 *
 * WHAT
 *   Test Hss Layer.
 *
 * CONCEPT
 *   Provides the Test Hss Layer.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_layer_create(void) {
    SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    SNEPPXHSSLayer* layer = SNEPPX_hss_layer_create(&cfg, 42);
    SX_ASSERT(layer != NULL, "layer not null");
    SX_ASSERT(layer->A != NULL, "A not null");
    SX_ASSERT(layer->B != NULL, "B not null");
    SX_ASSERT(layer->C != NULL, "C not null");
    SX_ASSERT(layer->D != NULL, "D not null");
    SX_ASSERT(layer->dt != NULL, "dt not null");
    SX_ASSERT(layer->h != NULL, "h not null");
    SX_ASSERT(layer->x_proj != NULL, "x_proj not null");
    SX_ASSERT(layer->x_proj_bias != NULL, "x_proj_bias not null");
    SX_ASSERT(layer->A->shape[0] == 4, "A rows == 4");
    SX_ASSERT(layer->A->shape[1] == 4, "A cols == 4");
    SX_ASSERT(layer->B->shape[0] == 4, "B rows == 4");
    SX_ASSERT(layer->B->shape[1] == 8, "B cols == 8");
    SX_ASSERT(layer->C->shape[0] == 8, "C rows == 8");
    SX_ASSERT(layer->C->shape[1] == 4, "C cols == 4");
    SX_ASSERT(layer->dt->shape[0] == 4, "dt size == 4");
    SNEPPX_hss_layer_destroy(layer);
}

static void test_discretize(void) {
    SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    SNEPPXHSSLayer* layer = SNEPPX_hss_layer_create(&cfg, 42);
    SX_ASSERT(layer != NULL, "layer not null");
    SNEPPX_hss_discretize(layer);
    SX_ASSERT(layer->A_bar != NULL, "A_bar not null");
    SX_ASSERT(layer->B_bar != NULL, "B_bar not null");
    SX_ASSERT(layer->A_bar->shape[0] == 4, "A_bar rows == 4");
    SX_ASSERT(layer->A_bar->shape[1] == 4, "A_bar cols == 4");
    SX_ASSERT(layer->B_bar->shape[0] == 4, "B_bar rows == 4");
    SX_ASSERT(layer->B_bar->shape[1] == 8, "B_bar cols == 8");
    SNEPPX_hss_layer_destroy(layer);
}

static void test_step(void) {
    SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    SNEPPXHSSLayer* layer = SNEPPX_hss_layer_create(&cfg, 42);
    SX_ASSERT(layer != NULL, "layer not null");
    SNEPPX_hss_discretize(layer);

    size_t shape_x[] = {8};
    size_t shape_h[] = {4};
    SNEPPXTensor* x = SNEPPX_tensor_zeros(shape_x, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* h_next = SNEPPX_tensor_zeros(shape_h, 1, SNEPPX_FLOAT32);
    SX_ASSERT(x != NULL && h_next != NULL, "tensors not null");
    ((float*)x->data)[0] = 1.0f;

    SNEPPX_hss_step(layer, x, h_next);
    float* hd = (float*)h_next->data;
    int finite = 1;
    for (size_t i = 0; i < 4; i++) {
        if (!isfinite(hd[i])) finite = 0;
    }
    SX_ASSERT(finite, "h_next values are finite");
    SNEPPX_tensor_destroy(x);
    SNEPPX_tensor_destroy(h_next);
    SNEPPX_hss_layer_destroy(layer);
}

static void test_scan(void) {
    SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    cfg.seq_len = 10;
    SNEPPXHSSLayer* layer = SNEPPX_hss_layer_create(&cfg, 42);
    SX_ASSERT(layer != NULL, "layer not null");
    SNEPPX_hss_discretize(layer);

    size_t shape_xs[] = {10, 8};
    size_t shape_hs[] = {10, 4};
    size_t shape_ys[] = {10, 8};
    SNEPPXTensor* x_seq = SNEPPX_tensor_randn(shape_xs, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* h_seq = SNEPPX_tensor_zeros(shape_hs, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* y_seq = SNEPPX_tensor_zeros(shape_ys, 2, SNEPPX_FLOAT32);
    SX_ASSERT(x_seq && h_seq && y_seq, "scan tensors not null");

    SNEPPX_hss_scan(layer, x_seq, h_seq, y_seq);

    SX_ASSERT(h_seq->shape[0] == 10, "h_seq rows == 10");
    SX_ASSERT(h_seq->shape[1] == 4, "h_seq cols == 4");
    SX_ASSERT(y_seq->shape[0] == 10, "y_seq rows == 10");
    SX_ASSERT(y_seq->shape[1] == 8, "y_seq cols == 8");

    SNEPPX_tensor_destroy(x_seq);
    SNEPPX_tensor_destroy(h_seq);
    SNEPPX_tensor_destroy(y_seq);
    SNEPPX_hss_layer_destroy(layer);
}


TEST(test_hss_layer, test_layer_create) { test_layer_create(); }
TEST(test_hss_layer, test_discretize) { test_discretize(); }
TEST(test_hss_layer, test_step) { test_step(); }
TEST(test_hss_layer, test_scan) { test_scan(); }
