#include "fractal_memory_orchestrator.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Fm Controller
 *
 * WHAT
 *   Test Fm Controller.
 *
 * CONCEPT
 *   Provides the Test Fm Controller.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_controller_create(void) {
    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    cfg.num_nodes = 4;
    cfg.memory_dim = 8;
    cfg.memory_capacity = 16;

    SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);
    SX_ASSERT(ctrl != NULL, "ctrl not null");
    SX_ASSERT(ctrl->nodes != NULL, "nodes not null");
    SX_ASSERT(ctrl->config.num_nodes == 4, "4 nodes");
    for (size_t i = 0; i < 4; i++) {
        SX_ASSERT(ctrl->nodes[i] != NULL, "node allocated");
        SX_ASSERT(ctrl->nodes[i]->node_id == i, "node id");
        SX_ASSERT(ctrl->nodes[i]->is_online == 1, "online");
    }
    SX_ASSERT(ctrl->sync_state.global_memory != NULL, "global memory");
    SX_ASSERT(ctrl->sync_state.sync_round == 0, "sync round 0");
    SNEPPX_fm_controller_destroy(ctrl);
}

static void test_controller_forward(void) {
    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    cfg.num_nodes = 2;
    cfg.memory_dim = 16;
    cfg.memory_capacity = 8;
    cfg.sync_interval = 1000;

    SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);
    SX_ASSERT(ctrl != NULL, "ctrl");

    size_t sh[] = {8, 16};
    SNEPPXTensor* input = SNEPPX_tensor_zeros(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* output = NULL;

    int r = SNEPPX_fm_forward(ctrl, 0, input, &output);
    SX_ASSERT(r == 0, "forward ok");
    SX_ASSERT(output != NULL, "output not null");
    SX_ASSERT(output->shape[0] == 8, "output rows 8");
    SX_ASSERT(output->shape[1] == 16, "output cols 16");

    int has_nan = 0, has_inf = 0;
    float* od = (float*)output->data;
    for (size_t i = 0; i < output->size; i++) {
        if (isnan(od[i])) has_nan = 1;
        if (isinf(od[i])) has_inf = 1;
    }
    SX_ASSERT(!has_nan, "no nan");
    SX_ASSERT(!has_inf, "no inf");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(output);
    SNEPPX_fm_controller_destroy(ctrl);
}


TEST(test_fm_controller, test_controller_create) { test_controller_create(); }
TEST(test_fm_controller, test_controller_forward) { test_controller_forward(); }
