#include "fractal_memory_orchestrator.h"
#include "test_gtest.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Fm Privacy
 *
 * WHAT
 *   Test Fm Privacy.
 *
 * CONCEPT
 *   Provides the Test Fm Privacy.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_fm_config_default(void) {
    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    SX_ASSERT(cfg.num_nodes == 4, "default 4 nodes");
    SX_ASSERT(cfg.sync_method == SNEPPX_SYNC_ALL_REDUCE, "default all-reduce sync");
    SX_ASSERT(cfg.privacy_epsilon > 0, "privacy epsilon > 0");
}

static void test_fm_controller_create_destroy(void) {
    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    cfg.memory_dim = 16;
    cfg.memory_capacity = 32;
    SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);
    SX_ASSERT(ctrl != NULL, "controller created");
    SX_ASSERT(ctrl->config.num_nodes == 4, "controller has 4 nodes");
    SNEPPX_fm_controller_destroy(ctrl);
}

static void test_fm_add_privacy_noise(void) {
    size_t shape[] = {2, 4};
    SNEPPXTensor* data = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    SNEPPX_fm_add_privacy_noise(data, 1.0f);
    float* d = (float*)data->data;
    float sum = 0.0f;
    for (size_t i = 0; i < 8; i++) sum += d[i];
    /* With epsilon=1.0, noise is added so data likely changed */
    SX_ASSERT(d[0] != 0.0f || d[1] != 0.0f || d[2] != 0.0f || d[3] != 0.0f, "privacy noise added");
    SNEPPX_tensor_destroy(data);
}

static void test_fm_compress_gradients(void) {
    size_t shape[] = {4, 4};
    SNEPPXTensor* grads = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    float* d = (float*)grads->data;
    for (size_t i = 0; i < 16; i++) d[i] = (float)(i % 4) * 0.5f;

    SNEPPXTensor* compressed = SNEPPX_fm_compress_gradients(grads, 0.5f);
    SX_ASSERT(compressed != NULL, "compressed gradients created");

    float* cd = (float*)compressed->data;
    int nonzero = 0;
    for (size_t i = 0; i < 16; i++) if (cd[i] != 0.0f) nonzero++;
    SX_ASSERT(nonzero > 0 && nonzero <= 8, "compression reduced nonzeros");

    SNEPPX_tensor_destroy(compressed);
    SNEPPX_tensor_destroy(grads);
}


TEST(test_fm_privacy, fm_config_default) { test_fm_config_default(); }
TEST(test_fm_privacy, fm_controller_create_destroy) { test_fm_controller_create_destroy(); }
TEST(test_fm_privacy, fm_add_privacy_noise) { test_fm_add_privacy_noise(); }
TEST(test_fm_privacy, fm_compress_gradients) { test_fm_compress_gradients(); }
